import copy
import io

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, append_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
    recursive_merge,
)
from aiida_shell import ShellJob


class MultipolePreprocessing(ShellJob):
    """Subclass for correct process labeling."""
    pass

class MultipolePostprocessing(ShellJob):
    """Subclass for correct process labeling."""
    pass

class MultipoleAnalysis(ShellJob):
    """Subclass for symmetry analysis step."""
    pass

class QuadrupoleWorkChain(ProtocolMixin, WorkChain):
    """
    Workchain to calculate dynamical quadrupoles.
    It orchestrates:
    1. Pre-processing (multipole.py -e) with MultipolePreprocessing
    2. Multiple Phonon calculations via PhBaseWorkChain (separate folders)
    3. Post-processing (multipole.py -f) with MultipolePostprocessing, collecting results
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Define Inputs
        spec.expose_inputs(
            PhBaseWorkChain,
            namespace="ph_base",
            exclude=(
                "clean_workdir",
                "ph.parent_folder",
                "qpoints",
                "qpoints_distance",
            ),
            namespace_options={
                "help": "Inputs for the `PhBaseWorkChain` that does the `ph.x` calculation."
           },
        )

        spec.input("structure", valid_type=orm.StructureData)
        spec.input(
            "clean_workdir", valid_type=orm.Bool, default=lambda: orm.Bool(False)
        )

        # Scf parent for data - NOW OPTIONAL
        spec.input('scf_parent_folder', valid_type=orm.RemoteData, required=False,
                   help='Optional SCF parent folder. If not provided, an SCF calculation will be run.')

        # SCF protocol for when no parent folder is provided
        spec.input('scf_protocol', valid_type=orm.Str, default=lambda: orm.Str('stringent'),
                   help='Protocol to use for SCF calculation when no parent folder is provided.')

        # Expose PwBaseWorkChain inputs for optional SCF calculation
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="pw_base",
            exclude=(
                "clean_workdir",
                "pw.structure",
            ),
            namespace_options={
                "help": "Inputs for the `PwBaseWorkChain` for SCF calculation when no parent folder is provided.",
                "required": False,
            },
        )

        # Multipole script
        spec.input(
            "multipole_script", valid_type=(orm.RemoteData, orm.SinglefileData)
        )

        # Shell code for running python3 (use aiida-shell)
        spec.input(
            "shell_code", valid_type=orm.AbstractCode,
            help="Code for running shell commands (python3) via aiida-shell."
        )

        spec.input('nq', valid_type=orm.Int, required=False)
        spec.input('seed', valid_type=orm.Int, required=False)


        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            if_(cls.should_run_analysis)(
                cls.run_symmetry_analysis,
            ),
            cls.run_preprocessing,
            cls.run_phonon_loop,
            cls.run_postprocessing,
            cls.results,
        )

        spec.output('quadrupole_fmt', valid_type=orm.SinglefileData)
        spec.output('quadrupole_remote', valid_type=orm.RemoteData,
                    help='Remote folder containing quadrupole.fmt file.')
        spec.exit_code(300, 'ERROR_PHONON_FAILED', message='One of the ph.x calculations failed.')
        spec.exit_code(301, 'ERROR_NO_INPUT_FILES', message='Preprocessing did not generate any ph.in.XX files.')
        spec.exit_code(302, 'ERROR_SCF_FAILED', message='The SCF calculation failed.')
        spec.exit_code(
            303,
            'ERROR_PREPROCESSING_FAILED',
            message='The preprocessing step failed.',
        )
        spec.exit_code(
            304,
            'ERROR_ANALYSIS_PARSING_FAILED',
            message='Failed to parse the multipole symmetry-analysis output.',
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / "quadrupole.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        codes,
        structure,
        multipole_script,
        scf_parent_folder=None,
        protocol='moderate',
        scf_protocol='stringent',
        overrides=None,
        **kwargs
    ):
        """
        Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param codes: Dictionary of codes. Required: 'ph'. Optional: 'pw' (needed if no scf_parent_folder).
        :param structure: The structure to calculate.
        :param multipole_script: The multipole.py script.
        :param scf_parent_folder: Optional SCF parent folder. If not provided, an SCF calculation will be run.
        :param protocol: Protocol for phonon calculation (default: 'moderate').
        :param scf_protocol: Protocol for SCF calculation if no parent folder is provided (default: 'stringent').
        :param overrides: Optional dictionary of inputs to override the defaults.
        """
        overrides = copy.deepcopy(overrides) if overrides else {}
        pw_base_overrides = overrides.pop("pw_base", {})
        ph_base_overrides = overrides.pop("ph_base", None)

        # 1. Get base inputs from protocol
        inputs = cls.get_protocol_inputs(protocol, overrides)

        # 2. Create the builder
        builder = cls.get_builder()

        # 3. Assign Mandatory Inputs
        builder.structure = structure
        builder.multipole_script = multipole_script

        # 4. Optionally assign SCF parent folder
        if scf_parent_folder is not None:
            builder.scf_parent_folder = scf_parent_folder
            # Since scf_parent_folder is provided, we don't need pw_base.
            # Pop it to avoid validation errors due to dummy fields being partially filled by get_builder()
            builder.pop('pw_base', None)
        else:
            # Need to configure pw_base for SCF calculation
            if 'pw' not in codes:
                raise ValueError("'pw' code is required when scf_parent_folder is not provided.")

            builder.scf_protocol = orm.Str(scf_protocol)

            # Get PwBaseWorkChain builder from protocol
            pw_base_overrides = recursive_merge(
                inputs.get("pw_base", {}), pw_base_overrides
            )
            # Ensure 'pw' key exists as PwBaseWorkChain expects it in overrides
            if 'pw' not in pw_base_overrides:
                pw_base_overrides['pw'] = {}

            # Quadrupole/lmultipole calculations ONLY work with insulators (fixed occupation)
            # Metal (with smearing) will cause "lmultipole does not work with metal" error
            from aiida_quantumespresso.common.types import ElectronicType

            pw_base_kwargs = {
                'code': codes['pw'],
                'structure': structure,
                'protocol': scf_protocol,
                'overrides': pw_base_overrides,
                'electronic_type': ElectronicType.INSULATOR,  # Required for lmultipole
                **kwargs
            }

            pw_base = PwBaseWorkChain.get_builder_from_protocol(**pw_base_kwargs)
            pw_base.pop("clean_workdir", None)
            pw_base.pw.pop("structure", None)
            builder.pw_base = pw_base

        args = (codes["ph"], None, protocol)
        ph_base_overrides = recursive_merge(
            inputs.get("ph_base", {}), ph_base_overrides or {}
        )
        ph_base = PhBaseWorkChain.get_builder_from_protocol(
            *args, overrides=ph_base_overrides or None, **kwargs
        )
        ph_base.pop("clean_workdir", None)
        ph_base.pop("qpoints_distance")

        builder.ph_base = ph_base

        return builder

    def setup(self):
        """Define constant parameters and template."""
        # Get ASE atoms object from StructureData
        ase_atoms = self.inputs.structure.get_ase()

        # Get lattice constant 'a' in Angstrom (first element of cell lengths)
        a_angstrom = ase_atoms.cell.lengths()[0]

        # Convert to Bohr (1 Angstrom = 1.889726 Bohr)
        # We format it as '8.237B' to match the script's expected string format
        self.ctx.alat_str = f"{a_angstrom:.4f}A"

        # self.report(f"Extracted alat from structure: {a_angstrom:.4f} A -> {self.ctx.alat_str}")
        self.ctx.order = 5
        # self.ctx.epsil_order = 4

        # Template for ph.in, to be used by multipole.py to generate ph.in.XX
        # We write this to a SinglefileData to pass to ShellJob
        ph_in_template = (
            "&inputph\n"
            "outdir='./out/',\n"
            "prefix='aiida',\n"
            "tr2_ph=1.0d-18,\n"
            "lmultipole = .true.\n"
            "fildrho = 'drho.dat.{}'\n"
            "fildvscf = 'drhodv.dat.{}'\n"
            "/\n"
        )
        self.ctx.ph_in_template = orm.SinglefileData(io.BytesIO(ph_in_template.encode('utf-8')))

        if isinstance(self.inputs.multipole_script, orm.SinglefileData):
            self.ctx.multipole_filename = self.inputs.multipole_script.filename
            self.ctx.multipole_file_dir = self.inputs.multipole_script.filename
        else:
            self.ctx.multipole_filename = None
            self.ctx.multipole_file_dir = self.inputs.multipole_script.get_remote_path()

    def get_scf_remote_folder(self):
        """Get the SCF remote folder - either from input or from the SCF calculation."""
        if 'scf_parent_folder' in self.inputs:
            return self.inputs.scf_parent_folder
        else:
            return self.ctx.scf_remote_folder

    def should_run_scf(self):
        """Run SCF only if scf_parent_folder is not provided."""
        return 'scf_parent_folder' not in self.inputs

    def run_scf(self):
        """Run the SCF calculation using PwBaseWorkChain with stringent/precise protocol."""
        scf_protocol = self.inputs.scf_protocol.value
        self.report(f"No scf_parent_folder provided, running SCF with '{scf_protocol}' protocol...")

        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace="pw_base")
        )
        inputs.pw.structure = self.inputs.structure
        inputs.metadata.call_link_label = "scf"

        workchain_node = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"launching PwBaseWorkChain<{workchain_node.pk}> for SCF calculation")

        return ToContext(workchain_scf=workchain_node)

    def inspect_scf(self):
        """Verify that the SCF calculation finished successfully and set scf_parent_folder."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f"PwBaseWorkChain<{workchain.pk}> failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SCF_FAILED

        # Store the remote folder for use in subsequent calculations
        self.ctx.scf_remote_folder = workchain.outputs.remote_folder
        self.report(f"SCF calculation completed successfully. Using remote folder from <{workchain.pk}>")

    def should_run_analysis(self):
        """Run analysis only if nq is not specified."""
        return 'nq' not in self.inputs

    def run_symmetry_analysis(self):
        """
        Step 1a: Run multipole.py -e (analysis only) to get NQ'.
        """
        self.report("Running multipole.py symmetry analysis...")

        arguments = [
            self.ctx.multipole_file_dir,
            '-e',
            '--order', str(self.ctx.order),
            # '--epsil_order', str(self.ctx.epsil_order),
            '--alat', self.ctx.alat_str,
        ]

        inputs = {
            'code': self.inputs.shell_code,
            'arguments': orm.List(list=arguments),
            'nodes': {
                'scf_data': self.get_scf_remote_folder(),
            },
            'metadata': {
                'label': 'MultipoleAnalysis',
                'options': {
                    'output_filename': 'analysis.out',
                    'prepend_text': 'cp aiida.in scf.in\n',
                },
            },
        }

        if self.ctx.multipole_filename:
            inputs['nodes']['multipole_script_node'] = self.inputs.multipole_script
            if 'filenames' not in inputs:
                inputs['filenames'] = {}
            inputs['filenames']['multipole_script_node'] = self.ctx.multipole_filename

        future = self.submit(MultipoleAnalysis, **inputs)
        return ToContext(analysis=future)

    def run_preprocessing(self):
        """
        Step 1b: Run multipole.py -e -p to generate input files.
        Calculates nq if needed.
        """
        self.report("Running multipole.py pre-processing...")

        nq_val = None
        if 'nq' in self.inputs:
            nq_val = self.inputs.nq.value
        else:
            # Parse output from analysis
            analysis_node = self.ctx.analysis
            if not analysis_node.is_finished_ok:
                return self.exit_codes.ERROR_PREPROCESSING_FAILED

            # Read output from ShellJob stdout (stored as analysis_out SinglefileData)
            try:
                stdout_node = analysis_node.outputs.analysis_out
                content = stdout_node.get_content()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')

                # Regex for "Total number of free components: XXX"
                import re
                import math
                match = re.search(r'Total number of free components:\s+(\d+)', content)
                if match:
                    nq_prime = int(match.group(1))
                    ase_atoms = self.inputs.structure.get_ase()
                    natom = len(ase_atoms)

                    n_min_q = nq_prime / (2.0 * natom) # 3/6
                    nq_val = max(1, math.ceil(n_min_q))
                    self.report(f"Calculated N_minq: {nq_val} (NQ'={nq_prime}, na={natom})")
                else:
                    self.report("Could not find 'Total number of free components' in analysis output.")
                    return self.exit_codes.ERROR_ANALYSIS_PARSING_FAILED
            except Exception as e:
                self.report(f"Failed to parse analysis output: {e}")
                return self.exit_codes.ERROR_ANALYSIS_PARSING_FAILED

        # Arguments for multipole.py
        arguments = [
            self.ctx.multipole_file_dir,
            '-e',
            '--order', str(self.ctx.order),
            # '--epsil_order', str(self.ctx.epsil_order),
            '-p',
            '--alat', self.ctx.alat_str,
            '--ir_q'
        ]

        # Pass calculated or provided nq
        if nq_val:
            arguments.extend(['--nq', str(nq_val)])

        if 'seed' in self.inputs:
            arguments.extend(['--seed', str(self.inputs.seed.value)])

        # Inputs for ShellJob
        inputs = {
            'code': self.inputs.shell_code,
            'arguments': orm.List(list=arguments),
            'nodes': {
                'ph_in': self.ctx.ph_in_template,
                'scf_data': self.get_scf_remote_folder(),  # RemoteData copied to workdir
            },
            'filenames': {
                'ph_in': 'ph.in'
            },
            'metadata': {
                'label': 'MultipolePreprocessing',
                'options': {
                    'output_filename': 'preprocessing.out',
                    'prepend_text': 'cp aiida.in scf.in\n',
                },
                'call_link_label': 'multipole_preprocessing',
            },
            'outputs': orm.List(list=['ph.in.*']),  # Retrieve generated files
        }

        if self.ctx.multipole_filename:
            inputs['nodes']['multipole_script_node'] = self.inputs.multipole_script
            # filenames dict already exists in this scope
            inputs['filenames']['multipole_script_node'] = self.ctx.multipole_filename

        future = self.submit(MultipolePreprocessing, **inputs)
        return ToContext(preproc=future)

    def run_phonon_loop(self):
        """
        Step 2: Run ph.x for each generated ph.in.XX using PhBaseWorkchain.
        Each runs in its own directory, but uses the ph.in.XX content.
        """
        self.report("Checking preprocessing outputs and starting phonon loop...")

        # ShellJob outputs ph.in.XX files as separate SinglefileData nodes with names like ph_in_01
        preproc_outputs = self.ctx.preproc.outputs

        # Find all ph_in_XX output nodes
        ph_output_names = [
            name for name in preproc_outputs
            if name.startswith('ph_in_') and name != 'ph_in'
        ]
        ph_output_names.sort()

        if not ph_output_names:
            self.report("No ph.in.XX files found! Preprocessing might have failed.")
            return self.exit_codes.ERROR_NO_INPUT_FILES

        self.report(f"Found {len(ph_output_names)} input files: {ph_output_names}")

        # Use SCF parent folder for phonon calculations (PhCalculation needs a QE parent, not ShellJob)
        scf_parent = self.get_scf_remote_folder()

        for ph_output_name in ph_output_names:
            # Get the SinglefileData node
            ph_file_node = getattr(preproc_outputs, ph_output_name)

            # Read content of the input file
            ph_content = ph_file_node.get_content()
            if isinstance(ph_content, bytes):
                try:
                    ph_content = ph_content.decode('utf-8')
                except UnicodeDecodeError:
                    self.report(f"Failed to decode {ph_output_name}, skipping.")
                    continue

            # Prepend text to overwrite aiida.in with the ph.in.XX content
            prepend_text = f"cat > aiida.in <<EOF\n{ph_content}\nEOF\n"

            try:
                options = self.inputs.ph_base.metadata.options.get_dict()
            except AttributeError:
                options = {}

            metadata = {'options': options.copy()}
            metadata['options']['prepend_text'] = metadata['options'].get('prepend_text', '') + prepend_text

            # Extract suffix from ph_output_name (e.g. ph_in_01 -> 01)
            suffix = ph_output_name.split('_')[-1]

            # Minimal dummy parameters to satisfy PhBaseWorkChain validation
            # The actual input file is created by prepend_text which overwrites aiida.in
            ph_params = {
                'INPUTPH': {
                    'tr2_ph': 1.0e-18,
                }
            }

            # Request retrieval of drho/drhodv files
            # The key ADDITIONAL_RETRIEVE_LIST in settings is deprecated.
            # We use CalcJob.metadata.options.additional_retrieve_list instead.
            metadata['options']['additional_retrieve_list'] = [
                f'drho.dat.{suffix}',
                f'drhodv.dat.{suffix}'
            ]

            inputs = {
                'ph': {
                    'code': self.inputs.ph_base.ph.code,
                    'parameters': orm.Dict(dict=ph_params),
                    'parent_folder': scf_parent,
                    'metadata': metadata,
                },
                # Dummy qpoints to satisfy validation - actual q-point is in the ph.in.XX file
                'qpoints': orm.KpointsData(),
            }
            # Set a 1x1x1 mesh as dummy (actual q-point is in the ph.in.XX file)
            inputs['qpoints'].set_kpoints_mesh([1, 1, 1])

            future = self.submit(PhBaseWorkChain, **inputs)
            self.to_context(ph_calculations=append_(future))

    def run_postprocessing(self):
        """
        Step 3: Run multipole.py -f to collect results.
        We gather drho.dat.XX and drhodv.dat.XX from all phonon calculations.
        """
        self.report("Running multipole.py post-processing...")

        # Collect failure info
        for ph_wf in self.ctx.ph_calculations:
            if not ph_wf.is_finished_ok:
                return self.exit_codes.ERROR_PHONON_FAILED

        # Prepare nodes dict - include preprocessing folder (has scf.in, qpoints.dat, etc.)
        # and all phonon remote folders (have drho.dat.XX, drhodv.dat.XX)
        nodes = {
            'preproc_data': self.ctx.preproc.outputs.remote_folder,
        }
        filenames = {}

        # Gather retrieved files from phonon calculations
        for i, ph_wf in enumerate(self.ctx.ph_calculations):
            suffix = f"{i+1:02d}"  # 01, 02, ...

            # Check if files were retrieved
            retrieved = ph_wf.outputs.retrieved
            file_list = retrieved.list_object_names()

            # drho.dat.XX
            drho_name = f'drho.dat.{suffix}'
            if drho_name in file_list:
                content = retrieved.get_object_content(drho_name)
                # Create SinglefileData
                if isinstance(content, str): content = content.encode('utf-8')
                drho_node = orm.SinglefileData(io.BytesIO(content))
                node_key = f'drho_{suffix}'
                nodes[node_key] = drho_node
                filenames[node_key] = drho_name
            else:
                self.report(f"Warning: {drho_name} not found in retrieved files of ph calc {i+1}")

            # drhodv.dat.XX
            drhodv_name = f'drhodv.dat.{suffix}'
            if drhodv_name in file_list:
                content = retrieved.get_object_content(drhodv_name)
                # Create SinglefileData
                if isinstance(content, str): content = content.encode('utf-8')
                drhodv_node = orm.SinglefileData(io.BytesIO(content))
                node_key = f'drhodv_{suffix}'
                nodes[node_key] = drhodv_node
                filenames[node_key] = drhodv_name
            else:
                self.report(f"Warning: {drhodv_name} not found in retrieved files of ph calc {i+1}")

        # Arguments: multipole.py -f ...
        arguments = [
            self.ctx.multipole_file_dir,
            '-f',
            '--order', str(self.ctx.order),
            # '--epsil_order', str(self.ctx.epsil_order),
            '--alat', self.ctx.alat_str
        ]

        inputs = {
            'code': self.inputs.shell_code,
            'arguments': orm.List(list=arguments),
            'nodes': nodes,
            'filenames': filenames,
            'metadata': {
                'label': 'MultipolePostprocessing',
                'options': {
                    'output_filename': 'postprocessing.out',
                },
                'call_link_label': 'multipole_postprocessing',
            },
            'outputs': orm.List(list=['quadrupole.fmt']),  # Retrieve generated files
        }

        if self.ctx.multipole_filename:
            inputs['nodes']['multipole_script_node'] = self.inputs.multipole_script
            if 'filenames' not in inputs:
                inputs['filenames'] = {}
            inputs['filenames']['multipole_script_node'] = self.ctx.multipole_filename

        future = self.submit(MultipolePostprocessing, **inputs)
        return ToContext(postproc=future)

    def results(self):
        """Retrieve the final quadrupole.fmt file."""
        self.report("Workchain completed. Finalizing outputs...")

        try:
            # ShellJob outputs files specified in 'outputs' as separate SinglefileData nodes
            # The filename 'quadrupole.fmt' becomes output name 'quadrupole_fmt'
            if hasattr(self.ctx.postproc.outputs, 'quadrupole_fmt'):
                self.out('quadrupole_fmt', self.ctx.postproc.outputs.quadrupole_fmt)
                self.report("Successfully retrieved quadrupole.fmt")
            else:
                self.report("quadrupole_fmt not found in postprocessing outputs.")
                # List available outputs for debugging
                self.report(f"Available outputs: {list(self.ctx.postproc.outputs)}")

            # Also output the remote folder for use by MobilityWorkChain
            if hasattr(self.ctx.postproc.outputs, 'remote_folder'):
                self.out('quadrupole_remote', self.ctx.postproc.outputs.remote_folder)
        except Exception as e:
            self.report(f"Failed to retrieve results: {e}")

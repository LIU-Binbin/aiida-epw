from aiida import orm
from pathlib import Path
import io
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_, append_
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_shell import ShellJob


class MultipolePreprocessing(ShellJob):
    """Subclass for correct process labeling."""
    pass

class MultipolePostprocessing(ShellJob):
    """Subclass for correct process labeling."""
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
        
        # Scf parent for data
        spec.input('scf_parent_folder', valid_type=orm.RemoteData)
        
        # Multipole script
        spec.input(
            "multipole_script", valid_type=(orm.RemoteData, orm.SinglefileData)
        )
        
        # Shell code for running python3 (use aiida-shell)
        spec.input(
            "shell_code", valid_type=orm.AbstractCode,
            help="Code for running shell commands (python3) via aiida-shell."
        )

 
        spec.outline(
            cls.setup,
            cls.run_preprocessing,
            cls.run_phonon_loop,
            cls.run_postprocessing,
            cls.results,
        )

        spec.output('quadrupole_mft', valid_type=orm.SinglefileData)
        spec.exit_code(300, 'ERROR_PHONON_FAILED', message='One of the ph.x calculations failed.')
        spec.exit_code(301, 'ERROR_NO_INPUT_FILES', message='Preprocessing did not generate any ph.in.XX files.')

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
        scf_parent_folder,
        multipole_script,
        protocol='moderate',
        overrides=None,
        **kwargs
    ):
        """
        Return a builder prepopulated with inputs selected according to the chosen protocol.
        """
        from aiida.engine import ProcessBuilder
        
        # 1. Get base inputs from protocol
        inputs = cls.get_protocol_inputs(protocol, overrides)

        # 2. Create the builder
        builder = cls.get_builder()
        
        # 3. Assign Mandatory Inputs
        builder.structure = structure 
        builder.scf_parent_folder = scf_parent_folder
        builder.multipole_script = multipole_script

        args = (codes["ph"], None, protocol)
        ph_base = PhBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get("ph_base", None), **kwargs
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

        self.ctx.mesh = ['3', '3', '3']
        self.ctx.order = 3
        self.ctx.epsil_order = 4
        
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
        self.ctx.multipole_file_dir = self.inputs.multipole_script.get_remote_path()

    def run_preprocessing(self):
        """
        Step 1: Run multipole.py -e to generate input files.
        We use MultipolePreprocessing (ShellJob).
        """
        self.report("Running multipole.py pre-processing...")
        
        # Arguments for multipole.py
        # multipole.py -e --order 3 ...
        arguments = [
            self.ctx.multipole_file_dir,
            '-e',
            '--order', str(self.ctx.order),
            '--epsil_order', str(self.ctx.epsil_order),
            '-p',
            '--mesh', *self.ctx.mesh,
            '--mesh_step', '0.01',
            '--alat', self.ctx.alat_str,
            '--ir_q'
        ]

        # Inputs for ShellJob
        # Passing scf_parent_folder as a node ensures its content (including scf.in/aiida.in) is copied to workdir
        inputs = {
            'code': self.inputs.shell_code,
            'arguments': orm.List(list=arguments),
            'nodes': {
                'ph_in': self.ctx.ph_in_template,
                'scf_data': self.inputs.scf_parent_folder,  # RemoteData copied to workdir
            },
            'filenames': {
                'ph_in': 'ph.in'
            },
            'metadata': {
                'label': 'MultipolePreprocessing',
                'options': {
                    'output_filename': 'preprocessing.out',
                    'prepend_text': 'cp aiida.in scf.in 2>/dev/null || true\n',
                },
                'call_link_label': 'multipole_preprocessing',
            },
            'outputs': orm.List(list=['ph.in.*']),  # Retrieve generated files
        }
        
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
        scf_parent = self.inputs.scf_parent_folder

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
            '--epsil_order', str(self.ctx.epsil_order),
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
            'outputs': orm.List(list=['quadruple.mft']),  # Retrieve generated files
        }

        future = self.submit(MultipolePostprocessing, **inputs)
        return ToContext(postproc=future)

    def results(self):
        """Retrieve the final quadruple.mft file."""
        self.report("Workchain completed. Finalizing outputs...")
        
        try:
            retrieved = self.ctx.postproc.outputs.retrieved
            if 'quadruple.mft' in retrieved.list_object_names():
                content = retrieved.get_object_content('quadruple.mft')
                # Ensure bytes
                if isinstance(content, str):
                    content = content.encode('utf-8')
                self.out('quadrupole_mft', orm.SinglefileData(io.BytesIO(content)))
            else:
                 self.report("quadrupole.fmt not found in retrieved files.")
        except Exception as e:
            self.report(f"Failed to retrieve results: {e}")

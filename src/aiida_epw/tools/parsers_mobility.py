"""Manual parsing functions for mobility post-processing."""

import numpy as np
import re
import io

def parse_epw_transport_out(file_contents, filename_pattern="epw.out"):
    """
    Parse the transport properties (Conductivity, Mobility, Hall factor) from EPW output.
    
    The function extracts 5 matrices for each temperature and approximation method (SERTA/BTE):
    1. Conductivity tensor without magnetic field (sigma)
    2. Conductivity tensor with magnetic field (sigma_B)
    3. Mobility tensor without magnetic field (mu)
    4. Hall mobility (mu_Hall)
    5. Hall factor (Z_Hall)

    parameters:
        file_contents: a dictionary where keys are filenames and values are file content strings.
                       (Compatible with AiiDA retrieved folder or local dict).
        filename_pattern: string or regex to match the EPW output file (default 'epw.out' or 'epw2.out').

    returns:
        parsed_data: a dictionary with structure:
            {
                Temperature (float): {
                    'SERTA': {
                        'sigma': np.array, 'sigma_B': np.array, 
                        'mu': np.array, 'mu_Hall': np.array, 
                        'hall_factor': np.array
                    },
                    'BTE': { ... same structure ... }
                },
                ...
            }
    """
    parsed_data = {}
    
    # Identify the correct file from the dictionary
    content = None
    for fname, fcontent in file_contents.items():
        if filename_pattern in fname:
            content = fcontent
            break
            
    if content is None:
        return parsed_data

    lines = content.splitlines()
    
    # State tracking
    current_approx = None  # 'SERTA' or 'BTE'
    current_T = None
    
    # Helper to parse lines with "|" separator (used in Conductivity and Mobility sections)
    def parse_split_matrix_lines(line_idx, lines_list):
        mat_left = []
        mat_right = []
        for i in range(3):
            line = lines_list[line_idx + i]
            parts = line.split('|')
            # Left matrix row
            mat_left.append([float(x) for x in parts[0].split()])
            # Right matrix row
            mat_right.append([float(x) for x in parts[1].split()])
        return np.array(mat_left), np.array(mat_right)

    # Helper to parse standard matrix (Hall factor)
    def parse_single_matrix_lines(line_idx, lines_list):
        mat = []
        for i in range(3):
            line = lines_list[line_idx + i]
            mat.append([float(x) for x in line.split()])
        return np.array(mat)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 1. Detect Approximation Method Section
        # Check for SERTA
        if "BTE in the self-energy relaxation time approximation (SERTA)" in line:
            current_approx = "SERTA"
        # Check for Iterative BTE (Exact match logic to distinguish from sentences containing BTE)
        elif line == "BTE":
            current_approx = "BTE"
            
        # 2. Detect Temperature
        if line.startswith("Temperature:"):
            # Line format: "Temperature:   300.0000 K"
            match = re.search(r"Temperature:\s+([\d\.]+)\s+K", line)
            if match:
                current_T = float(match.group(1))
                if current_T not in parsed_data:
                    parsed_data[current_T] = {}
                if current_approx:
                    if current_approx not in parsed_data[current_T]:
                        parsed_data[current_T][current_approx] = {}

        # 3. Parse Matrices (Only if we are inside a valid Temp and Approx block)
        if current_T is not None and current_approx is not None:
            container = parsed_data[current_T][current_approx]
            
            # Case A: Conductivity tensors (Split by |)
            if "Conductivity tensor without magnetic field" in line:
                # Data starts at next line
                sigma, sigma_B = parse_split_matrix_lines(i + 1, lines)
                container['sigma'] = sigma
                container['sigma_B'] = sigma_B
                i += 3 # Skip the data lines
            
            # Case B: Mobility tensors (Split by |)
            elif "Mobility tensor without magnetic field" in line:
                mu, mu_Hall = parse_split_matrix_lines(i + 1, lines)
                container['mu'] = mu
                container['mu_Hall'] = mu_Hall
                i += 3 
                
            # Case C: Hall factor (Single matrix)
            elif "Hall factor" in line:
                hall_factor = parse_single_matrix_lines(i + 1, lines)
                container['hall_factor'] = hall_factor
                i += 3

        i += 1

    return parsed_data




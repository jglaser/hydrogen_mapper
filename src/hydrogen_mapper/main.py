import argparse
import numpy as np
import pandas as pd
from hydrogen_mapper import active_loop_lib as al

def main():
    parser = argparse.ArgumentParser(description="Run a full active learning simulation for neutron phasing.")
    parser.add_argument('--polarization_files_csv', type=str, required=True, help='CSV file mapping polarization phis to mtz file paths')
    parser.add_argument('--coverage_events', type=str, required=True, help='NPZ file with coverage events to define goniometer search space.')
    parser.add_argument("--mtz_file", type=str, required=True, help="A representative MTZ file for crystal info")
    parser.add_argument("--mtz_array", type=str, required=True, help="Name of the intensity array in the mtz files")
    parser.add_argument('--pdb_ref', type=str, required=True, help='Reference structure for heavy atoms')
    parser.add_argument('--h_only_file', type=str, help='Reference structure factor (.mtz) for validation')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of active learning steps to perform')
    parser.add_argument('--n_candidates', type=int, default=10, help='Number of candidates to score')
    parser.add_argument('--num_recon_iter', type=int, default=500, help='Number of iterations per map reconstruction')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--mtz_out', type=str, default=None, help='.mtz (structure factor) output filename prefix')
    parser.add_argument('--mrc_out', type=str, default=None, help='.mrc (H density map) output filename prefix')
    args = parser.parse_args()

    # --- Run the active learning loop from the library ---
    al.run_active_learning_loop(args)

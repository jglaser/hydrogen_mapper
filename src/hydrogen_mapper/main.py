import argparse
import numpy as np
import pandas as pd
# Import the new ActiveLearningLoop class
from hydrogen_mapper.active_learning_loop import ActiveLearningLoop
from iotbx.reflection_file_reader import any_reflection_file
from cctbx import miller
from cctbx.array_family import flex
import iotbx.mrcfile
import os # Keep os for calculate_validation_metrics

def calculate_validation_metrics(F_H_map_recon, f_h_only_ref, miller_set):
    """Calculates R-factor, Phase Correlation, and Density Correlation."""
    # This function is unchanged from the original
    f_h_only_ref, F_H_map_recon = f_h_only_ref.common_sets(other=F_H_map_recon)

    sg_info = miller_set.crystal_symmetry().space_group_info()
    log = open(os.devnull,'w')
    f_h_only_ref = f_h_only_ref.change_symmetry(space_group_info=sg_info, log=log)
    F_H_map_recon = F_H_map_recon.change_symmetry(space_group_info=sg_info, log=log)

    scale_factor = np.sum(np.real(F_H_map_recon.data() * np.conj(f_h_only_ref.data())))
    scale_factor /= np.sum(np.abs(F_H_map_recon.data())**2)

    mag_ref = np.abs(f_h_only_ref.data())
    mag_recon = np.abs(F_H_map_recon.data()) * np.abs(scale_factor)
    r_fact = np.sum(np.abs(mag_ref - mag_recon)) / np.sum(mag_ref)

    phi_ref = np.angle(f_h_only_ref.data())
    phi_recon = np.angle(F_H_map_recon.data())
    delta_phi = phi_ref - phi_recon
    phase_corr = np.abs(np.mean(np.exp(1j * delta_phi)))

    fft_map_recon = (F_H_map_recon * scale_factor).fft_map(resolution_factor=1./4)
    fft_map_recon.apply_sigma_scaling()
    map_data_recon = fft_map_recon.real_map_unpadded()

    fft_map_ref = f_h_only_ref.fft_map(resolution_factor=1./4)
    fft_map_ref.apply_sigma_scaling()
    map_data_ref = fft_map_ref.real_map_unpadded()

    density_corr = flex.linear_correlation(x=map_data_recon.as_1d(), y=map_data_ref.as_1d()).coefficient()

    print(f" --> Validation: R-factor={r_fact:.4f}, PhaseCorr={phase_corr:.4f}, DensityCorr={density_corr:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run a full active learning simulation for neutron phasing.")
    # --- Arguments are updated ---
    parser.add_argument('--polarization_files_csv', type=str, required=True, help='CSV file mapping polarization phis to mtz file paths')
    parser.add_argument('--instrument_mask', type=str, required=True, help='Nexus file (.nxs) with instrument coverage definition.')
    parser.add_argument("--mtz_file", type=str, required=True, help="A representative MTZ file for crystal info")
    parser.add_argument("--mtz_array", type=str, required=True, help="Name of the intensity array in the mtz files")
    parser.add_argument('--pdb_ref', type=str, required=True, help='Reference structure for heavy atoms')
    parser.add_argument('--h_only_file', type=str, help='Reference structure factor (.mtz) for validation')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of active learning steps to perform')

    # --- These arguments are no longer used by the loop, but kept for compatibility ---
    parser.add_argument('--n_candidates', type=int, default=20, help='Number of candidate orientations to score per step')
    parser.add_argument('--num_recon_iter', type=int, default=500, help='Number of iterations per map reconstruction')
    parser.add_argument('--oversampling_factor', type=float, default=1.0, help='How many times the density grid is oversampled')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--mtz_out', type=str, default=None, help='.mtz (structure factor) output filename prefix')
    parser.add_argument('--mrc_out', type=str, default=None, help='.mrc (H density map) output filename prefix')
    args = parser.parse_args()

    # --- Initialization ---
    # All the complex setup is now handled by the ActiveLearningLoop constructor.
    print("Initializing Active Learning Loop...")
    loop = ActiveLearningLoop(
        instrument_mask_file=args.instrument_mask,
        reflection_file=args.mtz_file,
        pdb_file=args.pdb_ref,
        polarization_files_csv=args.polarization_files_csv,
        mtz_array_label=args.mtz_array,
        h_only_file=args.h_only_file
    )

    # We can pass external parameters (like from argparse) into the loop if needed
    # (Note: These are not implemented in the loop, but shows how you would)
    # loop.num_recon_iter = args.num_recon_iter
    # loop.learning_rate = args.learning_rate
    # loop.n_candidates = args.n_candidates

    print("--- Starting Simulation ---")
    for step in range(args.num_steps):
        if loop.next_state is None:
            print("No more states to measure. Simulation finished early.")
            break

        print(f"--- Step {step+1}/{args.num_steps} ---")
        print(f"Measuring state: phi_pol={loop.next_state['phi_pol']}")
        # print(f"Rotation Matrix:\n{loop.next_state['rotation_matrix']}") # Uncomment for verbose output

        # Simulate measurement. This single call:
        # 1. Moves data from unmeasured_data to measured_data
        # 2. Triggers the next map reconstruction (al.phase_retrieval_adam_direct)
        # 3. Triggers the next uncertainty calculation (al.calculate_trace_of_covariance_direct_blocked)
        # 4. Triggers the next candidate scoring (loop._score_candidate_orientations)
        # 5. Sets loop.F_H, loop.current_uncertainty, and loop.next_state
        loop.add_measurements([])

        # Validation and reporting (using results from the calculation
        # that just ran inside add_measurements)
        print(f" --> Current Uncertainty: {loop.current_uncertainty:.6e}")

        if args.h_only_file:
            # We can access the loop's internal state for validation
            calculate_validation_metrics(loop.F_H, loop.f_h_only_ref, loop.miller_set)

        # Save intermediate map (using the newly calculated F_H)
        if args.mrc_out and loop.F_H:
            fft_map = loop.F_H.fft_map(resolution_factor=1./4)
            fft_map.apply_sigma_scaling()
            file_name = f"{args.mrc_out}_step{step+1}.mrc"
            iotbx.mrcfile.write_ccp4_map(
                file_name   = file_name,
                unit_cell   = loop.F_H.unit_cell(),
                space_group = loop.F_H.crystal_symmetry().space_group(),
                map_data    = fft_map.real_map_unpadded().as_double(),
                labels      = flex.std_string([args.polarization_files_csv]))

            print(f"Wrote H-map: {file_name}")

        # Save intermediate mtz (using the newly calculated F_H)
        if args.mtz_out and loop.F_H:
            file_name = f"{args.mtz_out}_step{step+1}.mtz"
            mtz_dataset = loop.F_H.as_mtz_dataset(column_root_label="F_H")
            mtz_dataset.mtz_object().write(file_name)
            print(f"Wrote H-factors: {file_name}")

    print("--- Simulation Complete ---")

if __name__ == '__main__':
    main()

import argparse
import numpy as np
import pandas as pd
from hydrogen_mapper import active_loop_lib as al
from iotbx.reflection_file_reader import any_reflection_file
from mmtbx.model import manager as model_manager
import iotbx.pdb
from cctbx import miller
from cctbx.array_family import flex

def calculate_validation_metrics(F_H_map_recon, f_h_only_ref, miller_set):
    """Calculates R-factor, Phase Correlation, and Density Correlation."""
    f_h_only_ref, F_H_map_recon = f_h_only_ref.common_sets(other=F_H_map_recon)

    sg_info = miller_set.crystal_symmetry().space_group_info()
    import os
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
    # (Argument parsing remains the same)
    parser.add_argument('--polarization_files_csv', type=str, required=True, help='CSV file mapping polarization phis to mtz file paths')
    parser.add_argument('--coverage_events', type=str, required=True, help='NPZ file with coverage events to define goniometer search space.')
    parser.add_argument("--mtz_file", type=str, required=True, help="A representative MTZ file for crystal info")
    parser.add_argument("--mtz_array", type=str, required=True, help="Name of the intensity array in the mtz files")
    parser.add_argument('--pdb_ref', type=str, required=True, help='Reference structure for heavy atoms')
    parser.add_argument('--h_only_file', type=str, help='Reference structure factor (.mtz) for validation')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of active learning steps to perform')
    parser.add_argument('--n_candidates', type=int, default=10, help='Number of candidates to score')
    parser.add_argument('--num_recon_iter', type=int, default=500, help='Number of iterations per map reconstruction')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for Adam optimizer')
    parser.add_argument('--oversampling_factor', type=float, default=1.0, help='How many times the density grid is oversampled')
    parser.add_argument('--mtz_out', type=str, default=None, help='.mtz (structure factor) output filename prefix')
    parser.add_argument('--mrc_out', type=str, default=None, help='.mrc (H density map) output filename prefix')
    args = parser.parse_args()

    # --- Initialization ---
    miller_set = any_reflection_file(args.mtz_file).as_miller_arrays(merge_equivalents=False)[0]
    pdb_inp = iotbx.pdb.input(file_name=args.pdb_ref)
    model = model_manager(model_input=pdb_inp)
    model.get_hierarchy().remove_hd()
    xrs = model.get_xray_structure()
    xrs.switch_to_neutron_scattering_dictionary()
    f_heavy_calc = xrs.structure_factors(d_min=miller_set.d_min(), algorithm='direct').f_calc()
    f_heavy_calc, miller_set = f_heavy_calc.common_sets(other=miller_set)
    p1_map = miller_set.expand_to_p1()
    f_heavy_map = f_heavy_calc.expand_to_p1()
    map_shape = p1_map.fft_map(resolution_factor=1./3).real_map_unpadded().all()
    n_voxels = np.prod(map_shape)
    p1_indices = list(p1_map.indices())
    
    f_h_only_ref = None
    if args.h_only_file:
        f_h_only_ref_native = any_reflection_file(args.h_only_file).as_miller_arrays()[0]
        f_h_only_ref = f_h_only_ref_native.expand_to_p1()
        f_h_only_ref, f_heavy_map = f_h_only_ref.common_sets(other=f_heavy_map)

    pool_data = al.load_full_dataset(args.polarization_files_csv, args.mtz_array)
    pool_data['hkl'] = pool_data[['h','k','l']].apply(tuple, axis=1)

    coverage_data = np.load(args.coverage_events)
    hkl_map_coverage = coverage_data['hkl_map']
    event_log = coverage_data['event_log']
    gonio_angles_to_search = np.unique(event_log[:, 0])
    hkl_to_fheavy_map = {tuple(hkl): amp for hkl, amp in zip(np.array(f_heavy_calc.indices()), np.array(f_heavy_calc.amplitudes().data()))}
    optimal_gonio_angle = al.find_optimal_initial_goniometer_angle(gonio_angles_to_search, hkl_map_coverage, event_log, hkl_to_fheavy_map)
    
    event_idx = 0
    covered_at_optimal = set()
    while event_idx < len(event_log) and event_log[event_idx, 0] <= optimal_gonio_angle:
        _, event_type, hkl_id = event_log[event_idx]
        hkl = tuple(hkl_map_coverage[int(hkl_id)])
        if event_type == 1:
            covered_at_optimal.add(hkl)
        else:
            covered_at_optimal.discard(hkl)
        event_idx += 1

    pool_data['goniometer_angle'] = -1.0 
    for gonio_angle in gonio_angles_to_search:
        event_idx = 0
        covered_hkls = set()
        while event_idx < len(event_log) and event_log[event_idx, 0] <= gonio_angle:
            _, event_type, hkl_id = event_log[event_idx]
            hkl = tuple(hkl_map_coverage[int(hkl_id)])
            if event_type == 1: covered_hkls.add(hkl)
            else: covered_hkls.discard(hkl)
            event_idx += 1
        pool_data.loc[pool_data['hkl'].isin(covered_hkls), 'goniometer_angle'] = gonio_angle
        
    first_phi_pol = sorted(pool_data['phi_pol'].unique())[0]
    seed_mask = (pool_data['hkl'].isin(covered_at_optimal)) & (pool_data['phi_pol'] == first_phi_pol)
    measured_data = pool_data[seed_mask].copy()
    unmeasured_data = pool_data.drop(measured_data.index)

    grid = None
    for step in range(args.num_steps):
        print(f"--- Step {step+1}/{args.num_steps} ---")

        # Correctly create miller_arrays from the measured_data DataFrame
        miller_arrays = []
        indices = miller_set.indices()
        for phi in measured_data['phi_pol'].unique():
            I_hkl = measured_data.groupby(['hkl','phi_pol'])['I'].mean()
            I = [ I_hkl.get((tuple(k), phi), 0) for k in indices ]
            sigI_hkl = measured_data.groupby(['hkl','phi_pol'])['sigI'].mean()
            sigI = [ sigI_hkl.get((tuple(k), phi), -1) for k in indices ]
            m = miller.array(miller_set, data=flex.double(I), sigmas=flex.double(sigI))
            miller_arrays.append(m)

        F_H, grid = al.phase_retrieval_adam_direct(miller_arrays, measured_data['phi_pol'].unique(), f_heavy_map, grid,
                                                   num_iterations=args.num_recon_iter, learning_rate=args.learning_rate,
                                                   oversampling_factor=args.oversampling_factor)

        if args.h_only_file:
            calculate_validation_metrics(F_H, f_h_only_ref, miller_set)

        current_uncertainty = al.calculate_trace_of_covariance_direct_blocked(measured_data, F_H, f_heavy_map, p1_indices, n_voxels)

        if len(unmeasured_data) == 0:
            break

        best_state = al.score_candidate_states_optimized(unmeasured_data, measured_data, F_H, f_heavy_map, p1_indices, n_voxels)

        if best_state is not None:
            gonio_angle_to_measure, phi_pol_to_measure = best_state['goniometer_angle'], best_state['phi_pol']
            batch_mask = (unmeasured_data['goniometer_angle'] == gonio_angle_to_measure) & (unmeasured_data['phi_pol'] == phi_pol_to_measure)
            batch_to_measure = unmeasured_data[batch_mask]
            measured_data = pd.concat([measured_data, batch_to_measure])
            unmeasured_data = unmeasured_data.drop(batch_to_measure.index)

if __name__ == '__main__':
    main()

import numpy as np
import argparse
import pandas as pd
import re
from iotbx.reflection_file_reader import any_reflection_file
from mmtbx.model import manager as model_manager
import iotbx.pdb
from cctbx.array_family import flex
from cctbx import miller
from cctbx import maptbx
from cctbx import crystal
import mrcfile
from skimage.restoration import denoise_tv_chambolle

# JAX for acceleration
import jax
import jax.numpy as jnp
from functools import partial

# --- Core Functions ---

def load_full_dataset(csv_path, mtz_array_label):
    """
    Loads a full dataset from a CSV file mapping polarization phis to mtz files.
    """
    df_map = pd.read_csv(csv_path)
    all_data = []
    print("Loading full dataset from MTZ files specified in CSV...")
    for _, row in df_map.iterrows():
        phi_pol, mtz_file = row['phi'], row['mtz_file']

        reader = any_reflection_file(mtz_file)
        miller_arrays = reader.as_miller_arrays(merge_equivalents=False)

        intensities = None
        available_labels = [a.info().label_string() for a in miller_arrays]
        for array in miller_arrays:
            if array.info().label_string() == mtz_array_label:
                if not array.is_xray_intensity_array():
                    array = array.as_intensity_array()
                intensities = array
                break

        if intensities is None:
            print(f"Error: Array '{mtz_array_label}' not found in {mtz_file}")
            print(f"Available arrays are: {available_labels}")
            raise ValueError(f"Array not found.")

        indices = intensities.indices()
        sigIs = intensities.sigmas()
        if sigIs is None:
            print(f"  - WARNING: No sigmas found in {mtz_file}. Using dummy sigmas of 1.0.")
            sigIs = flex.double(intensities.data().size(), 1.0)

        df_mtz = pd.DataFrame({
            'h': list(indices.as_vec3_double().parts()[0]),
            'k': list(indices.as_vec3_double().parts()[1]),
            'l': list(indices.as_vec3_double().parts()[2]),
            'phi_pol': phi_pol,
            'I': list(intensities.data()),
            'sigI': list(sigIs)
        })
        all_data.append(df_mtz)
        print(f"  - Loaded {len(df_mtz)} reflections for phi_pol={phi_pol}")
    return pd.concat(all_data, ignore_index=True)

def find_optimal_initial_goniometer_angle(gonio_angles, hkl_map, event_log, hkl_to_fheavy_map):
    """
    Finds the optimal initial goniometer angle by minimizing the average
    heavy-atom amplitude of covered reflections, using the event log.
    """
    best_gonio_angle = None
    best_score = float('inf')

    print("\nSearching for optimal initial goniometer setting using coverage events...")

    event_log = event_log[event_log[:, 0].argsort()] # Sort by goniometer angle

    for gonio_angle in np.sort(gonio_angles):
        # Determine covered hkls at this specific angle by replaying the log
        covered_hkls_at_angle = set()
        event_idx = 0
        while event_idx < len(event_log) and event_log[event_idx, 0] <= gonio_angle:
            _, event_type, hkl_id = event_log[event_idx]
            hkl = tuple(hkl_map[int(hkl_id)])
            if event_type == 1:
                covered_hkls_at_angle.add(hkl)
            else:
                covered_hkls_at_angle.discard(hkl)
            event_idx += 1

        if not covered_hkls_at_angle: continue

        amplitudes = [hkl_to_fheavy_map[hkl] for hkl in covered_hkls_at_angle if hkl in hkl_to_fheavy_map]
        if not amplitudes: continue

        score = np.mean(amplitudes)
        print(f"  - Goniometer Angle {gonio_angle:.2f}°: Score (Avg |F_heavy|): {score:.4f}, Covered: {len(amplitudes)}")

        if score < best_score:
            best_score, best_gonio_angle = score, gonio_angle

    return best_gonio_angle

def get_jacobian_constants_and_weights_direct(measured_data, F_H_map, f_heavy_map, p1_indices):
    """
    Computes complex constants 'c' for the Jacobian J = 2*Re(psi*c) and weights
    for each measurement based on the direct intensity model.
    """
    p1_hkl_to_idx = {hkl: i for i, hkl in enumerate(p1_indices)}
    const_list, weight_list, hkl_list = [], [], []

    F_H_np = F_H_map.data().as_numpy_array()
    f_heavy_np = f_heavy_map.data().as_numpy_array()

    for _, row in measured_data.iterrows():
        hkl = (row['h'], row['k'], row['l'])
        idx = p1_hkl_to_idx.get(hkl)
        if idx is None: continue

        phi_pol = row['phi_pol']
        F_H_q = F_H_np[idx]
        f_heavy_q = f_heavy_np[idx]

        F_tot_q = f_heavy_q + phi_pol * F_H_q

        c_q = phi_pol * np.conj(F_tot_q)
        w_q = 1.0 / (row['sigI']**2 + 1e-9)

        const_list.append(c_q)
        weight_list.append(w_q)
        hkl_list.append(hkl)

    return const_list, weight_list, hkl_list

def get_friedel_key(hkl):
    """Generates a canonical key for a Friedel pair."""
    h, k, l = hkl
    if h > 0 or (h == 0 and k > 0) or (h == 0 and k == 0 and l > 0):
        return hkl
    elif h == 0 and k == 0 and l == 0:
        return hkl
    else:
        return (-h, -k, -l)

@partial(jax.jit, static_argnums=(2,))
def _calculate_block_trace_jax(const_block, weight_block, n_voxels, hkls_in_block, epsilon):
    """
    JIT-compiled core for calculating the trace contribution from a single HKL block.
    """
    k = const_block.shape[0]
    G_block = jnp.zeros((k, k), dtype=jnp.complex64)

    hkls_outer = hkls_in_block[:, None, :]
    same_hkl_mask = jnp.all(hkls_outer == hkls_in_block, axis=-1)
    friedel_hkl_mask = jnp.all(hkls_outer == -hkls_in_block, axis=-1)

    const_outer_conj = const_block[:, None] * jnp.conj(const_block)
    const_outer = const_block[:, None] * const_block

    G_block = G_block.at[:,:].set(
        n_voxels * (same_hkl_mask * const_outer_conj + friedel_hkl_mask * const_outer)
    )
    G_block = jnp.real(G_block)

    W_block_inv = jnp.diag(1.0 / (weight_block + 1e-9))
    A_block = W_block_inv + (1.0 / epsilon) * G_block

    A_block_inv = jnp.linalg.inv(A_block)
    trace_contrib = jnp.trace(W_block_inv @ A_block_inv)
    return trace_contrib

def calculate_trace_of_covariance_direct_blocked(measured_data, F_H_map, f_heavy_map, p1_indices, n_voxels, epsilon=1.0):
    """
    Wrapper function that uses a block-diagonal approach for efficient uncertainty calculation.
    """
    const_list, weight_list, hkl_list = get_jacobian_constants_and_weights_direct(measured_data, F_H_map, f_heavy_map, p1_indices)
    if not const_list: return np.inf

    friedel_groups = {}
    for i, hkl in enumerate(hkl_list):
        key = get_friedel_key(hkl)
        if key not in friedel_groups:
            friedel_groups[key] = []
        friedel_groups[key].append(i)

    total_trace_contrib = 0.0

    for key, indices in friedel_groups.items():
        const_block = jnp.array([const_list[i] for i in indices])
        weight_block = jnp.array([weight_list[i] for i in indices])
        hkls_in_block = jnp.array([hkl_list[i] for i in indices])

        trace_contrib = _calculate_block_trace_jax(const_block, weight_block, n_voxels, hkls_in_block, epsilon)
        total_trace_contrib += trace_contrib

    M = len(const_list)
    trace_cov = ((n_voxels - M) / epsilon) + ((1.0 / epsilon) * total_trace_contrib)
    return float(trace_cov)

def to_numpy(F, sys_abs=None, return_sigma=False, return_mask=False):
    """convert Miller array to numpy fft array"""
    F = F.expand_to_p1()

    if not F.anomalous_flag():
        F = F.generate_bijvoet_mates()
    else:
        print("Data already includes Friedel pairs.")

    indices=np.array(F.indices())
    h_max=np.max(indices, axis=0)
    x,y,z=2*np.abs(h_max)+1
    a=np.zeros((x,y,z),dtype=np.complex64)

    if return_mask:
        mask=np.ones(shape=(x,y,z), dtype=np.float32)
        if sys_abs is not None:
            for k in zip(sys_abs.indices()):
                mask[tuple(k)]=0

    if return_sigma:
        sigma=np.full(shape=(x,y,z), fill_value=-1, dtype=np.float32)
        for k,v,s in zip(indices, F.data(), F.sigmas()):
            #use negative indexing to achieve np.fft.fftfreq convention
            a[tuple(k)]=v
            sigma[tuple(k)]=s

        if return_mask:
            return a, sigma, mask
        else:
            return a, sigma
    else:
        for k,v in zip(indices, F.data()):
            #use negative indexing to achieve np.fft.fftfreq convention
            a[tuple(k)]=v
        if return_mask:
            return a, mask
        else:
            return a

def cost_function_total(rho_H_grid, f_heavy, measured_phis, measured_Is, measured_sigIs):
    """
    Computes the scalar data-fidelity cost from a TOTAL real-space density grid.
    """
    # Derive the H-only map in Fourier space
    F_H_grid = jnp.fft.fftn(rho_H_grid)

    # Calculate cost based on the H-only structure factors
    F_tot_q_pred = jnp.stack([f_heavy + phi * F_H_grid for phi in measured_phis])
    I_pred = jnp.abs(F_tot_q_pred)**2
    residuals = jnp.where(measured_sigIs > 0, (I_pred - measured_Is)**2 / (measured_sigIs**2 + 1e-9), 0)
    return jnp.sum(residuals)/jnp.sum(jnp.where(measured_sigIs > 0, 1/measured_sigIs**2, 0))

@partial(jax.jit, static_argnames=('weight', 'max_iter'))
def _tv_prox_jax(input_grid, weight, max_iter=50):
    """
    JAX implementation of TV-denoising prox operator.
    """
    def body_fun(_, p):
        grad_div_p = jnp.zeros_like(input_grid)
        for i in range(input_grid.ndim):
            grad_div_p += jnp.roll(p[..., i], -1, axis=i) - p[..., i]
        g = input_grid - grad_div_p / weight
        grad_g = jnp.stack([g - jnp.roll(g, 1, axis=i) for i in range(input_grid.ndim)], axis=-1)
        norm = jnp.sqrt(jnp.sum(grad_g**2, axis=-1, keepdims=True))
        p_next = (p - 0.25 * weight * grad_g) / (1.0 + 0.25 * norm)
        return p_next

    p_initial = jnp.zeros(input_grid.shape + (input_grid.ndim,), dtype=input_grid.dtype)
    p_final = jax.lax.fori_loop(0, max_iter, body_fun, p_initial)
    final_grad_div_p = jnp.zeros_like(input_grid)
    for i in range(input_grid.ndim):
        final_grad_div_p += jnp.roll(p_final[..., i], -1, axis=i) - p_final[..., i]
    return input_grid - final_grad_div_p

def phase_retrieval_adam_direct(measured_data, phis, f_heavy_arr, num_iterations=500, tol=1e-6,
                                learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                lambda_tv=0.01, use_sysabs=False):
    """
    Performs phase retrieval using the Adam optimizer.
    """
    f_heavy, mask = to_numpy(f_heavy_arr, sys_abs=sys_abs if use_sysabs else None, return_mask=True)
    f_heavy = jnp.array(f_heavy)
    mask = jnp.array(mask)
    indices = f_heavy_arr.indices()

    nps = [to_numpy(a, return_sigma=True) for a in measured_data]
    measured_Is = jnp.stack([jnp.abs(a[0]) for a in nps])
    measured_sigIs = jnp.stack([a[1] for a in nps])

    # Define the gradient of the cost function
    grad_fun = jax.jit(jax.grad(cost_function_total, argnums=0))

    def adam_body_fun(carry, _):
        rho_H_grid, m, v, t = carry
        rho_H_prev_grid = rho_H_grid

        # 1. Calculate gradient
        grad_rho_H_grid = grad_fun(rho_H_grid, f_heavy, phis, measured_Is, measured_sigIs)
        grad_rho_H_grid_masked = grad_rho_H_grid * mask

        # 2. Update biased first and second moment estimates
        m_next = beta1 * m + (1 - beta1) * grad_rho_H_grid_masked
        v_next = beta2 * v + (1 - beta2) * jnp.square(grad_rho_H_grid_masked)

        # 3. Compute bias-corrected moment estimates
        t_next = t + 1
        m_hat = m_next / (1 - beta1**t_next)
        v_hat = v_next / (1 - beta2**t_next)

        # 4. Update parameters (density grid)
        update_step = learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        rho_H_intermediate = rho_H_grid - update_step

        # 5. Apply physical constraints (same as in FISTA)
        rho_H_intermediate = rho_H_intermediate.real
        rho_H_denoised = _tv_prox_jax(rho_H_intermediate, learning_rate * lambda_tv)
        rho_H_final = jnp.maximum(rho_H_denoised, 0)

        # Ensure type consistency for the next iteration's FFT
        rho_H_grid_next = jnp.array(rho_H_final, dtype=jnp.complex64)

        # Calculate convergence error
        error = jnp.linalg.norm(rho_H_grid_next - rho_H_prev_grid) / (jnp.linalg.norm(rho_H_prev_grid) + 1e-9)

        return (rho_H_grid_next, m_next, v_next, t_next), error

    # Initialize parameters and Adam state
    rho_H_grid_init = jnp.zeros_like(f_heavy, dtype=jnp.complex64)
    m_init = jnp.zeros_like(rho_H_grid_init)
    v_init = jnp.zeros_like(rho_H_grid_init)
    t_init = 0
    initial_carry = (rho_H_grid_init, m_init, v_init, t_init)

    # Run the optimization using jax.lax.scan
    final_carry, errors = jax.lax.scan(adam_body_fun, initial_carry, None, length=num_iterations)
    final_rho_H_grid = final_carry[0]

    if False:
        # Print progress and check for convergence
        errors_np = np.array(errors)
        for i, error in enumerate(errors_np):
            if i % 10 == 0:
                print(f"  - Iteration {i}: error = {error:.4g}")
            if error < tol and i > 0:
                print(f"Converged after {i+1} iterations.")
                break
        if i == num_iterations - 1:
            print(f"Reached max iterations ({num_iterations}).")

    # Reconstruct final structure factors
    final_F_H_grid_jax = jnp.fft.fftn(final_rho_H_grid)
    final_F_H_grid_np = np.array(final_F_H_grid_jax, dtype=np.complex128)

    Fh_1d = np.array([final_F_H_grid_np[tuple(k)] for k in indices])
    return miller.array(f_heavy_arr.set(), data=flex.complex_double(Fh_1d))

def prepare_friedel_groups(data, F_H_map, f_heavy_map, p1_indices):
    """
    Processes a DataFrame of measurements into a dictionary of Friedel groups,
    containing the necessary constants, weights, and HKLs for the trace calculation.
    """
    const_list, weight_list, hkl_list = get_jacobian_constants_and_weights_direct(
        data, F_H_map, f_heavy_map, p1_indices
    )
    friedel_groups = {}
    if not const_list:
        return friedel_groups

    for i, hkl in enumerate(hkl_list):
        key = get_friedel_key(hkl)
        if key not in friedel_groups:
            friedel_groups[key] = {'consts': [], 'weights': [], 'hkls': []}
        friedel_groups[key]['consts'].append(const_list[i])
        friedel_groups[key]['weights'].append(weight_list[i])
        friedel_groups[key]['hkls'].append(hkl_list[i])
    return friedel_groups

def score_candidate_states_optimized(unmeasured_data, measured_data, F_H_map, f_heavy_map, p1_indices, n_voxels, epsilon=1.0):
    """
    Scores candidate states by efficiently calculating the change in the trace of
    the covariance matrix, leveraging JAX for the core numerical update.
    """
    # 1. Pre-process base data and calculate base trace contributions
    base_groups = prepare_friedel_groups(measured_data, F_H_map, f_heavy_map, p1_indices)

    base_trace_contributions = {}
    total_trace_contrib_base = 0.0
    M_base = 0

    for key, group_data in base_groups.items():
        const_block = jnp.array(group_data['consts'])
        weight_block = jnp.array(group_data['weights'])
        hkls_in_block = jnp.array(group_data['hkls'])
        M_base += len(const_block)

        trace_contrib = _calculate_block_trace_jax(const_block, weight_block, n_voxels, hkls_in_block, epsilon)
        base_trace_contributions[key] = trace_contrib
        total_trace_contrib_base += trace_contrib

    trace_base = ((n_voxels - M_base) / epsilon) + ((1.0 / epsilon) * total_trace_contrib_base)

    if np.isinf(trace_base) or M_base == 0:
        print("      - Baseline trace is infinite or no data measured. Cannot score candidates.")
        # Fallback: return a random candidate if scoring isn't possible
        candidate_states = unmeasured_data[['goniometer_angle', 'phi_pol']].drop_duplicates()
        if candidate_states.empty: return None
        return candidate_states.sample(n=1).iloc[0]

    print(f"      - Baseline Trace: {trace_base:.5g}")
    best_state = None
    min_predicted_trace = trace_base

    # 2. Iterate through candidates and score them by updating the trace
    candidate_states = unmeasured_data[['goniometer_angle', 'phi_pol']].drop_duplicates()
    states_to_check = candidate_states.sample(n=min(args.n_candidates, len(candidate_states)))

    for _, state in states_to_check.iterrows():
        gonio_angle, phi_pol = state['goniometer_angle'], state['phi_pol']
        batch_to_add = unmeasured_data[(unmeasured_data['goniometer_angle'] == gonio_angle) & (unmeasured_data['phi_pol'] == phi_pol)]
        if batch_to_add.empty: continue

        candidate_groups = prepare_friedel_groups(batch_to_add, F_H_map, f_heavy_map, p1_indices)

        predicted_trace_contrib_total = total_trace_contrib_base
        M_candidate = M_base + len(batch_to_add)

        # Loop over only the few groups affected by the candidate batch
        for key, cand_group_data in candidate_groups.items():
            old_trace_contrib = base_trace_contributions.get(key, 0.0)

            # Combine base and candidate data for this specific group
            if key in base_groups:
                base_group_data = base_groups[key]
                combined_consts = base_group_data['consts'] + cand_group_data['consts']
                combined_weights = base_group_data['weights'] + cand_group_data['weights']
                combined_hkls = base_group_data['hkls'] + cand_group_data['hkls']
            else: # This is a completely new Friedel group
                combined_consts = cand_group_data['consts']
                combined_weights = cand_group_data['weights']
                combined_hkls = cand_group_data['hkls']

            # Recalculate trace contribution for only this single updated group
            const_block_new = jnp.array(combined_consts)
            weight_block_new = jnp.array(combined_weights)
            hkls_in_block_new = jnp.array(combined_hkls)
            new_trace_contrib = _calculate_block_trace_jax(const_block_new, weight_block_new, n_voxels, hkls_in_block_new, epsilon)

            # Update the total contribution with the delta
            predicted_trace_contrib_total += (new_trace_contrib - old_trace_contrib)

        predicted_trace = ((n_voxels - M_candidate) / epsilon) + ((1.0 / epsilon) * predicted_trace_contrib_total)

        improvement = trace_base - predicted_trace
        print(f"      - Testing ({gonio_angle:.2f}°, {phi_pol:.2f}): Predicted Trace={predicted_trace:.5g}, Improvement={improvement:.4g}")

        if predicted_trace < min_predicted_trace:
            min_predicted_trace = predicted_trace
            best_state = state

    if best_state is None:
        print("    - No improvement found. Choosing a random candidate to expand coverage.")
        return states_to_check.iloc[0]

    return best_state

def score_candidate_states(unmeasured_data, measured_data, F_H_map, f_heavy_map, p1_indices, n_voxels):
    trace_base = calculate_trace_of_covariance_direct_blocked(measured_data, F_H_map, f_heavy_map, p1_indices, n_voxels)
    if np.isinf(trace_base):
        print("      - Baseline trace is infinite. Cannot score candidates.")
        return None 

    best_state = None
    min_predicted_trace = trace_base
    print(f"      - Baseline Trace: {trace_base:.5g}")

    candidate_states = unmeasured_data[['goniometer_angle', 'phi_pol']].drop_duplicates()
    states_to_check = candidate_states.sample(n=min(10, len(candidate_states)))

    for _, state in states_to_check.iterrows():
        gonio_angle, phi_pol = state['goniometer_angle'], state['phi_pol']
        batch_to_add = unmeasured_data[(unmeasured_data['goniometer_angle'] == gonio_angle) & (unmeasured_data['phi_pol'] == phi_pol)]
        if batch_to_add.empty: continue

        hypothetical_data = pd.concat([measured_data, batch_to_add])

        predicted_trace = calculate_trace_of_covariance_direct_blocked(hypothetical_data, F_H_map, f_heavy_map, p1_indices, n_voxels)

        improvement = trace_base - predicted_trace
        print(f"      - Testing ({gonio_angle:.2f}°, {phi_pol:.2f}): Predicted Trace={predicted_trace:.5g}, Improvement={improvement:.4g}")

        if predicted_trace < min_predicted_trace:
            min_predicted_trace = predicted_trace
            best_state = state

    if best_state is None:
        print("    - No improvement found. Choosing a random candidate to expand coverage.")
        return states_to_check.iloc[0]

    return best_state

def calculate_validation_metrics(F_H_map_recon, f_h_only_ref):
    """Calculates R-factor, Phase Correlation, and Density Correlation."""
    # Ensure common set of reflections
    f_h_only_ref, F_H_map_recon = f_h_only_ref.common_sets(other=F_H_map_recon)

    sg_info = miller_set.crystal_symmetry().space_group_info()
    import os
    log = open(os.devnull,'w')
    f_h_only_ref = f_h_only_ref.change_symmetry(space_group_info=sg_info, log=log)
    F_H_map_recon = F_H_map_recon.change_symmetry(space_group_info=sg_info, log=log)

    scale_factor = np.sum(np.real(F_H_map_recon.data() * np.conj(f_h_only_ref.data())))
    scale_factor /= np.sum(np.abs(F_H_map_recon.data())**2)

    # R-factor
    mag_ref = np.abs(f_h_only_ref.data())
    mag_recon = np.abs(F_H_map_recon.data()) * np.abs(scale_factor)
    r_fact = np.sum(np.abs(mag_ref - mag_recon)) / np.sum(mag_ref)

    # Phase Correlation
    phi_ref = np.angle(f_h_only_ref.data())
    phi_recon = np.angle(F_H_map_recon.data())
    delta_phi = phi_ref - phi_recon
    phase_corr = np.abs(np.mean(np.exp(1j * delta_phi)))

    # Density Correlation
    fft_map_recon = (F_H_map_recon * scale_factor).fft_map(resolution_factor=1./4)
    fft_map_recon.apply_sigma_scaling()
    map_data_recon = fft_map_recon.real_map_unpadded()

    fft_map_ref = f_h_only_ref.fft_map(resolution_factor=1./4)
    fft_map_ref.apply_sigma_scaling()
    map_data_ref = fft_map_ref.real_map_unpadded()

    density_corr = flex.linear_correlation(x=map_data_recon.as_1d(), y=map_data_ref.as_1d()).coefficient()

    print(f" --> Validation: R-factor={r_fact:.4f}, PhaseCorr={phase_corr:.4f}, DensityCorr={density_corr:.4f}")

def write_mtz(f_pol, fn):
    mtz_dataset = f_pol.as_mtz_dataset(column_root_label="f_pol")
    mtz_object = mtz_dataset.mtz_object()
    mtz_object.write(file_name=fn)
    print(f"Wrote reconstructed F_H to {fn}")

def write_mrc(f_pol, fn, label):
    fft_map = f_pol.fft_map(resolution_factor=1./4)
    fft_map.apply_sigma_scaling()
    map_data = fft_map.real_map_unpadded()
    iotbx.mrcfile.write_ccp4_map(file_name=fn,
        unit_cell=f_pol.unit_cell(),
        space_group=f_pol.crystal_symmetry().space_group(),
        map_data=map_data.as_double(),
        labels=flex.std_string([label]))
    print(f"Wrote reconstructed density map to {fn}")

# --- Main Script ---
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

# --- 1. Initialization ---
print("--- 1. Initialization ---")
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

# get the systematic absences
ms_base = miller_set.customized_copy(crystal_symmetry=crystal.symmetry(
        unit_cell=miller_set.unit_cell(),
        space_group=miller_set.space_group().build_derived_point_group()
        )
)
ms_all = ms_base.complete_set()
sys_abs = ms_all.lone_set(other=ms_base)

map_shape = p1_map.fft_map(resolution_factor=1./3).real_map_unpadded().all()
n_voxels = np.prod(map_shape)
p1_indices = list(p1_map.indices())
print(f"Number of voxels in map: {n_voxels}")

if args.h_only_file:
    print(f"Loading H-only reference from {args.h_only_file} for validation.")
    f_h_only_ref_native = any_reflection_file(args.h_only_file).as_miller_arrays()[0]
    # FIX: Expand H-only map to P1 to match the heavy-atom map before finding common sets
    f_h_only_ref = f_h_only_ref_native.expand_to_p1()
    f_h_only_ref, f_heavy_map = f_h_only_ref.common_sets(other=f_heavy_map)

# --- 2. Load and Prepare Simulation Data ---
print("\n--- 2. Loading and Preparing Simulation Data ---")
pool_data = load_full_dataset(args.polarization_files_csv, args.mtz_array)
pool_data['hkl'] = pool_data[['h','k','l']].apply(tuple, axis=1)
print(f"Created a global pool of {len(pool_data)} possible measurements.")

# --- 3. Active Learning Loop ---
print("\n--- 3. Starting Active Learning Loop ---")

coverage_data = np.load(args.coverage_events)
hkl_map_coverage = coverage_data['hkl_map']
event_log = coverage_data['event_log']
gonio_angles_to_search = np.unique(event_log[:, 0])

hkls_heavy = np.array(f_heavy_calc.indices())
f_heavy_abs = np.array(f_heavy_calc.amplitudes().data())
hkl_to_fheavy_map = {tuple(hkl): amp for hkl, amp in zip(hkls_heavy, f_heavy_abs)}
optimal_gonio_angle = find_optimal_initial_goniometer_angle(gonio_angles_to_search, hkl_map_coverage, event_log, hkl_to_fheavy_map)
print(f"\nOptimal goniometer angle for seeding: {optimal_gonio_angle:.2f} degrees")

event_idx = 0
covered_at_optimal = set()
while event_idx < len(event_log) and event_log[event_idx, 0] <= optimal_gonio_angle:
    _, event_type, hkl_id = event_log[event_idx]
    hkl = tuple(hkl_map_coverage[int(hkl_id)])
    if event_type == 1: covered_at_optimal.add(hkl)
    else: covered_at_optimal.discard(hkl)
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

# Using the first polarizations in the file
first_phi_pol = sorted(pool_data['phi_pol'].unique())[0]
seed_mask = (pool_data['hkl'].isin(covered_at_optimal)) & (pool_data['phi_pol'] == first_phi_pol)
measured_data = pool_data[seed_mask].copy()
unmeasured_data = pool_data.drop(measured_data.index)
print(f"Initialized with {len(measured_data)} measurements at optimal goniometer angle {optimal_gonio_angle:.2f}° and phi_pol = {first_phi_pol:.2f}")

# uncomment to load full dataset
#measured_data = pd.concat([measured_data, unmeasured_data])
#unmeasured_data = []

#print(len(measured_data))

for step in range(args.num_steps):
    print(f"\n--- Step {step+1}/{args.num_steps} ---")
    print(f" --> Total measurements: {len(measured_data)}")

    hkl_counts = measured_data.groupby('hkl').size()
    n_single = np.sum(hkl_counts == 1)
    n_double = np.sum(hkl_counts == 2)
    n_triple_plus = np.sum(hkl_counts >= 3)
    print(f" --> Constraint Status: {n_triple_plus} HKLs (≥3 pts), {n_double} HKLs (2 pts), {n_single} HKLs (1 pt)")


    print(" (a) Reconstructing map...")
    phis = measured_data['phi_pol'].unique()
    miller_arrays = []
    indices = miller_set.indices()
    for phi in phis:
        I_hkl = measured_data.groupby(['hkl','phi_pol'])['I'].mean()
        I = [ I_hkl[(tuple(k), phi)] if (tuple(k), phi) in I_hkl else 0 for k in indices ]
        sigI_hkl = measured_data.groupby(['hkl','phi_pol'])['sigI'].mean()
        sigI = [ sigI_hkl[(tuple(k), phi)] if (tuple(k), phi) in sigI_hkl else -1 for k in indices ]
        m = miller.array(miller_set, data=flex.double(I), sigmas=flex.double(sigI))
        miller_arrays.append(m)
    F_H = phase_retrieval_adam_direct(miller_arrays, phis, f_heavy_map, num_iterations=args.num_recon_iter,
                                      learning_rate=args.learning_rate)

    if args.h_only_file:
        # The validation function needs the P1 version of the H-only map
        calculate_validation_metrics(F_H, f_h_only_ref)

    if args.mtz_out:
        write_mtz(F_H, args.mtz_out + f"_{step}.mtz")

    if args.mrc_out:
        write_mrc(F_H, args.mrc_out + f"_{step}.mrc", label=args.polarization_files_csv)

    print(" (b) Estimating uncertainty...")
    current_uncertainty = calculate_trace_of_covariance_direct_blocked(measured_data, F_H, f_heavy_map, p1_indices, n_voxels)

    print(f" --> Total Uncertainty (Tr(Cov)): {current_uncertainty:.5g}")

    print(" (c) Scoring candidate states...")
    if len(unmeasured_data) == 0:
        print("All possible data has been measured. Stopping.")
        break

#    best_state = score_candidate_states(unmeasured_data, measured_data, F_H, f_heavy_map, p1_indices, n_voxels)
    best_state = score_candidate_states_optimized(unmeasured_data, measured_data, F_H, f_heavy_map, p1_indices, n_voxels)


    if best_state is None:
        print("Scoring failed, selecting a random state.")
        candidate_states = unmeasured_data[['goniometer_angle', 'phi_pol']].drop_duplicates()
        if candidate_states.empty: break
        best_state = candidate_states.sample(n=1).iloc[0]

    gonio_angle_to_measure, phi_pol_to_measure = best_state['goniometer_angle'], best_state['phi_pol']

    batch_mask = (unmeasured_data['goniometer_angle'] == gonio_angle_to_measure) & (unmeasured_data['phi_pol'] == phi_pol_to_measure)
    batch_to_measure = unmeasured_data[batch_mask]
    
    measured_data = pd.concat([measured_data, batch_to_measure])
    unmeasured_data = unmeasured_data.drop(batch_to_measure.index)

    print(f" (d) 'Measuring' {len(batch_to_measure)} reflections at goniometer angle {gonio_angle_to_measure:.2f}° and phi_pol={phi_pol_to_measure:.2f}")

print("\n--- Active Learning Simulation Complete ---")



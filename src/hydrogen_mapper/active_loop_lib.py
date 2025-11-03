import numpy as np
import pandas as pd
from iotbx.reflection_file_reader import any_reflection_file
from mmtbx.model import manager as model_manager
import iotbx.pdb
from cctbx.array_family import flex
from cctbx import miller
from cctbx import crystal
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
        for array in miller_arrays:
            if array.info().label_string() == mtz_array_label:
                if not array.is_xray_intensity_array():
                    array = array.as_intensity_array()
                intensities = array
                break
        if intensities is None:
            raise ValueError(f"Array '{mtz_array_label}' not found in {mtz_file}")
        indices = intensities.indices()
        sigIs = intensities.sigmas() if intensities.sigmas() is not None else flex.double(intensities.data().size(), 1.0)
        df_mtz = pd.DataFrame({
            'h': list(indices.as_vec3_double().parts()[0]),
            'k': list(indices.as_vec3_double().parts()[1]),
            'l': list(indices.as_vec3_double().parts()[2]),
            'phi_pol': phi_pol,
            'I': list(intensities.data()),
            'sigI': list(sigIs)
        })
        all_data.append(df_mtz)
    return pd.concat(all_data, ignore_index=True)

def find_optimal_initial_goniometer_angle(gonio_angles, hkl_map, event_log, hkl_to_fheavy_map):
    """
    Finds the optimal initial goniometer angle by minimizing the average
    heavy-atom amplitude of covered reflections.
    """
    best_gonio_angle = None
    best_score = float('inf')
    event_log = event_log[event_log[:, 0].argsort()]
    for gonio_angle in np.sort(gonio_angles):
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
        if score < best_score:
            best_score, best_gonio_angle = score, gonio_angle
    return best_gonio_angle

def get_jacobian_constants_and_weights_direct(measured_data, F_H_map, f_heavy_map, p1_indices):
    """
    Computes complex constants 'c' for the Jacobian and weights for each measurement.
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
    """JIT-compiled core for calculating the trace contribution from a single HKL block."""
    k = const_block.shape[0]
    hkls_outer = hkls_in_block[:, None, :]
    same_hkl_mask = jnp.all(hkls_outer == hkls_in_block, axis=-1)
    friedel_hkl_mask = jnp.all(hkls_outer == -hkls_in_block, axis=-1)
    const_outer_conj = const_block[:, None] * jnp.conj(const_block)
    const_outer = const_block[:, None] * const_block
    G_block = jnp.real(n_voxels * (same_hkl_mask * const_outer_conj + friedel_hkl_mask * const_outer))
    W_block_inv = jnp.diag(1.0 / (weight_block + 1e-9))
    A_block = W_block_inv + (1.0 / epsilon) * G_block
    A_block_inv = jnp.linalg.inv(A_block)
    return jnp.trace(W_block_inv @ A_block_inv)

def calculate_trace_of_covariance_direct_blocked(measured_data, F_H_map, f_heavy_map, p1_indices, n_voxels, epsilon=1.0):
    """Wrapper function that uses a block-diagonal approach for efficient uncertainty calculation."""
    const_list, weight_list, hkl_list = get_jacobian_constants_and_weights_direct(measured_data, F_H_map, f_heavy_map, p1_indices)
    if not const_list: return np.inf
    friedel_groups = {}
    for i, hkl in enumerate(hkl_list):
        key = get_friedel_key(hkl)
        if key not in friedel_groups:
            friedel_groups[key] = []
        friedel_groups[key].append(i)
    total_trace_contrib = 0.0
    for _, indices in friedel_groups.items():
        const_block = jnp.array([const_list[i] for i in indices])
        weight_block = jnp.array([weight_list[i] for i in indices])
        hkls_in_block = jnp.array([hkl_list[i] for i in indices])
        total_trace_contrib += _calculate_block_trace_jax(const_block, weight_block, n_voxels, hkls_in_block, epsilon)
    M = len(const_list)
    return float(((n_voxels - M) / epsilon) + ((1.0 / epsilon) * total_trace_contrib))

def to_numpy(F, sys_abs=None, return_sigma=False, return_mask=False):
    """Convert Miller array to numpy fft array"""
    F = F.expand_to_p1()
    if not F.anomalous_flag():
        F = F.generate_bijvoet_mates()
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
            a[tuple(k)]=v
            sigma[tuple(k)]=s
        return (a, sigma, mask) if return_mask else (a, sigma)
    else:
        for k,v in zip(indices, F.data()):
            a[tuple(k)]=v
        return (a, mask) if return_mask else a

def cost_function_total(rho_H_grid, f_heavy, measured_phis, measured_Is, measured_sigIs):
    """Computes the scalar data-fidelity cost from a TOTAL real-space density grid."""
    F_H_grid = jnp.fft.fftn(rho_H_grid)
    F_tot_q_pred = jnp.stack([f_heavy + phi * F_H_grid for phi in measured_phis])
    I_pred = jnp.abs(F_tot_q_pred)**2
    residuals = jnp.where(measured_sigIs > 0, (I_pred - measured_Is)**2 / (measured_sigIs**2 + 1e-9), 0)
    return jnp.sum(residuals)/jnp.sum(jnp.where(measured_sigIs > 0, 1/measured_sigIs**2, 0))

@partial(jax.jit, static_argnames=('weight', 'max_iter'))
def _tv_prox_jax(input_grid, weight, max_iter=50):
    """JAX implementation of TV-denoising prox operator."""
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
    'measured_data' is now a list of miller.array objects.
    """
    f_heavy, mask = to_numpy(f_heavy_arr, sys_abs=f_heavy_arr.crystal_symmetry().space_group_info() if use_sysabs else None, return_mask=True)
    f_heavy = jnp.array(f_heavy)
    mask = jnp.array(mask)
    indices = f_heavy_arr.indices()

    # The first argument 'measured_data' is now the list of miller_arrays
    nps = [to_numpy(a, return_sigma=True) for a in measured_data]
    measured_Is = jnp.stack([jnp.abs(a[0]) for a in nps])
    measured_sigIs = jnp.stack([a[1] for a in nps])

    grad_fun = jax.jit(jax.grad(cost_function_total, argnums=0))

    # ... (the rest of the Adam optimizer logic is the same)
    def adam_body_fun(carry, _):
        rho_H_grid, m, v, t = carry
        rho_H_prev_grid = rho_H_grid
        grad_rho_H_grid = grad_fun(rho_H_grid, f_heavy, phis, measured_Is, measured_sigIs)
        grad_rho_H_grid_masked = grad_rho_H_grid * mask
        m_next = beta1 * m + (1 - beta1) * grad_rho_H_grid_masked
        v_next = beta2 * v + (1 - beta2) * jnp.square(grad_rho_H_grid_masked)
        t_next = t + 1
        m_hat = m_next / (1 - beta1**t_next)
        v_hat = v_next / (1 - beta2**t_next)
        update_step = learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        rho_H_intermediate = rho_H_grid - update_step
        rho_H_intermediate = rho_H_intermediate.real # real constraint
        rho_H_denoised = _tv_prox_jax(rho_H_intermediate, learning_rate * lambda_tv)
        rho_H_grid_next = jnp.array(rho_H_denoised, dtype=jnp.complex64)
        error = jnp.linalg.norm(rho_H_grid_next - rho_H_prev_grid) / (jnp.linalg.norm(rho_H_prev_grid) + 1e-9)
        return (rho_H_grid_next, m_next, v_next, t_next), error

    rho_H_grid_init = jnp.zeros_like(f_heavy, dtype=jnp.complex64)
    m_init, v_init, t_init = jnp.zeros_like(rho_H_grid_init), jnp.zeros_like(rho_H_grid_init), 0
    initial_carry = (rho_H_grid_init, m_init, v_init, t_init)

    final_carry, _ = jax.lax.scan(adam_body_fun, initial_carry, None, length=num_iterations)
    final_rho_H_grid = final_carry[0]

    final_F_H_grid_jax = jnp.fft.fftn(final_rho_H_grid)
    final_F_H_grid_np = np.array(final_F_H_grid_jax, dtype=np.complex128)
    Fh_1d = np.array([final_F_H_grid_np[tuple(k)] for k in indices])
    return miller.array(f_heavy_arr.set(), data=flex.complex_double(Fh_1d))

def prepare_friedel_groups(data, F_H_map, f_heavy_map, p1_indices):
    """Processes a DataFrame of measurements into a dictionary of Friedel groups."""
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

def score_candidate_states_optimized(unmeasured_data, measured_data, F_H_map, f_heavy_map, p1_indices, n_voxels, epsilon=1.0, n_candidates=10):
    """Scores candidate states by efficiently calculating the change in the trace of the covariance matrix."""
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
        candidate_states = unmeasured_data[['goniometer_angle', 'phi_pol']].drop_duplicates()
        return None if candidate_states.empty else candidate_states.sample(n=1).iloc[0]
    best_state = None
    min_predicted_trace = trace_base
    candidate_states = unmeasured_data[['goniometer_angle', 'phi_pol']].drop_duplicates()
    states_to_check = candidate_states.sample(n=min(n_candidates, len(candidate_states)))
    for _, state in states_to_check.iterrows():
        gonio_angle, phi_pol = state['goniometer_angle'], state['phi_pol']
        batch_to_add = unmeasured_data[(unmeasured_data['goniometer_angle'] == gonio_angle) & (unmeasured_data['phi_pol'] == phi_pol)]
        if batch_to_add.empty: continue
        candidate_groups = prepare_friedel_groups(batch_to_add, F_H_map, f_heavy_map, p1_indices)
        predicted_trace_contrib_total = total_trace_contrib_base
        M_candidate = M_base + len(batch_to_add)
        for key, cand_group_data in candidate_groups.items():
            old_trace_contrib = base_trace_contributions.get(key, 0.0)
            if key in base_groups:
                base_group_data = base_groups[key]
                combined_consts = base_group_data['consts'] + cand_group_data['consts']
                combined_weights = base_group_data['weights'] + cand_group_data['weights']
                combined_hkls = base_group_data['hkls'] + cand_group_data['hkls']
            else:
                combined_consts = cand_group_data['consts']
                combined_weights = cand_group_data['weights']
                combined_hkls = cand_group_data['hkls']
            const_block_new = jnp.array(combined_consts)
            weight_block_new = jnp.array(combined_weights)
            hkls_in_block_new = jnp.array(combined_hkls)
            new_trace_contrib = _calculate_block_trace_jax(const_block_new, weight_block_new, n_voxels, hkls_in_block_new, epsilon)
            predicted_trace_contrib_total += (new_trace_contrib - old_trace_contrib)
        predicted_trace = ((n_voxels - M_candidate) / epsilon) + ((1.0 / epsilon) * predicted_trace_contrib_total)
        if predicted_trace < min_predicted_trace:
            min_predicted_trace = predicted_trace
            best_state = state
    if best_state is None:
        return states_to_check.iloc[0]
    return best_state

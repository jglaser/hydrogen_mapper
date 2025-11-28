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
        columns = []
        for array in miller_arrays:
            column_name = array.info().label_string()
            columns += [column_name]
            if column_name == mtz_array_label:
                if not array.is_xray_intensity_array():
                    array = array.as_intensity_array()
                intensities = array
                break
        if intensities is None:
            raise ValueError(f"Array '{mtz_array_label}' not found in {mtz_file}, Available columns: {columns}")
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
        df_mtz['hkl'] = list(zip(df_mtz['h'], df_mtz['k'], df_mtz['l']))
        all_data.append(df_mtz)
    return pd.concat(all_data, ignore_index=True)

def get_jacobian_constants_and_weights_direct(measured_data, F_H_map, f_heavy_map, p1_indices):
    """
    Computes complex constants 'c' for the Jacobian and weights for each measurement.
    """
    p1_hkl_to_idx = {hkl: i for i, hkl in enumerate(p1_indices)}
    const_list, weight_list, hkl_list = [], [], []

    F_H_np = F_H_map.data().as_numpy_array()
    f_heavy_np = f_heavy_map.data().as_numpy_array()

    # Ensure hkl column exists and is tuple
    if 'hkl' not in measured_data.columns and not measured_data.empty:
        measured_data['hkl'] = measured_data[['h','k','l']].apply(tuple, axis=1)

    for _, row in measured_data.iterrows():
        hkl = row['hkl'] # Use the tuple directly
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
    A_block_inv = jnp.linalg.pinv(A_block)
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

def to_numpy(F, return_sigma=False):
    """Convert Miller array to numpy fft array"""
    F = F.expand_to_p1()
    if not F.anomalous_flag():
        F = F.generate_bijvoet_mates()
    indices=np.array(F.indices())
    h_max=np.max(indices, axis=0)
    x,y,z=2*np.abs(h_max)+1
    a=np.zeros((x,y,z),dtype=np.complex64)
    if return_sigma:
        sigma=np.full(shape=(x,y,z), fill_value=-1, dtype=np.float32)
        for k,v,s in zip(indices, F.data(), F.sigmas()):
            a[tuple(k)]=v
            sigma[tuple(k)]=s
        return (a, sigma)
    else:
        for k,v in zip(indices, F.data()):
            a[tuple(k)]=v
        return a

def cost_function_total(rho_H_grid, k_scales, f_heavy, measured_phis, measured_Is, measured_sigIs,
                        IsigI_cutoff=None):
    """Computes the scalar data-fidelity cost from a TOTAL real-space density grid."""
    F_H_grid = jnp.fft.fftn(rho_H_grid)
    F_tot_q_pred = jnp.stack([f_heavy + phi * F_H_grid for phi in measured_phis])
    I_calc = jnp.abs(F_tot_q_pred)**2
    k_scales_broadcast = jnp.expand_dims(k_scales, axis=tuple(range(1, F_H_grid.ndim + 1)))
    I_pred = jnp.exp(k_scales_broadcast) * I_calc
    if IsigI_cutoff is not None:
        f = measured_Is > IsigI_cutoff * measured_sigIs
    else:
        f = jnp.ones_like(measured_Is, dtype=bool)

    residuals = jnp.where(f & (measured_sigIs > 0), (I_pred - measured_Is)**2, 0)
    weights = jnp.where(f & (measured_sigIs > 0), 1 / (measured_sigIs**2 + 1e-9), 0)
    return jnp.sum(weights*residuals)/jnp.sum(weights)

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

def phase_retrieval_adam_direct(measured_data, phis, f_heavy_arr,
                                grid=None, k_scales=None, num_iterations=500, tol=1e-6,
                                learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                lambda_tv=0.01, oversampling_factor=1.0,
                                IsigI_cutoff=None):
    """
    Performs phase retrieval using the Adam optimizer.
    'measured_data' is now a list of miller.array objects.
    """
    f_heavy = jnp.array(to_numpy(f_heavy_arr))
    indices = f_heavy_arr.indices()

    if grid is None:
        grid_shape = f_heavy.shape
        grid_shape = tuple([int(g*oversampling_factor) for g in grid_shape])
    else:
        grid_shape = grid.shape

    # The first argument 'measured_data' is now the list of miller_arrays
    nps = [to_numpy(a, return_sigma=True) for a in measured_data]
    measured_Is = [jnp.abs(a[0]) for a in nps]
    measured_sigIs = [a[1] for a in nps]

    l_pad = tuple((g_new - g) // 2 + (g_new - g) % 2 for g_new, g in zip(grid_shape, f_heavy.shape))
    r_pad = tuple((g_new - g) // 2 for g_new, g in zip(grid_shape, f_heavy.shape))
    pad = tuple((l,r) for l, r in zip(l_pad, r_pad))

    f_heavy = jnp.fft.ifftshift(jnp.pad(jnp.fft.fftshift(f_heavy), pad))
    measured_Is = jnp.stack([jnp.fft.ifftshift(jnp.pad(jnp.fft.fftshift(I), pad)) for I in measured_Is])
    measured_sigIs = jnp.stack([jnp.fft.ifftshift(jnp.pad(jnp.fft.fftshift(sigI), pad, constant_values=-1)) for sigI in measured_sigIs])

    grad_fun = jax.jit(jax.grad(cost_function_total, argnums=(0,1)))

    def adam_body_fun(carry, _):
        rho_H_grid, k_scales, m_F, v_F, m_k, v_k, t = carry
        rho_H_prev_grid = rho_H_grid
        grad_rho_H_grid, grad_k_scales = grad_fun(rho_H_grid, k_scales, f_heavy, phis, measured_Is, measured_sigIs,
                                                  IsigI_cutoff=IsigI_cutoff)
        m_F_next = beta1 * m_F + (1 - beta1) * grad_rho_H_grid
        v_F_next = beta2 * v_F + (1 - beta2) * jnp.square(grad_rho_H_grid)
        t_next = t + 1
        m_F_hat = m_F_next / (1 - beta1**t_next)
        v_F_hat = v_F_next / (1 - beta2**t_next)
        update_step = learning_rate * m_F_hat / (jnp.sqrt(v_F_hat) + epsilon)
        rho_H_intermediate = rho_H_grid - update_step
        rho_H_intermediate = rho_H_intermediate.real
        rho_H_denoised = _tv_prox_jax(rho_H_intermediate, learning_rate * lambda_tv)
        rho_H_grid_next = jnp.array(rho_H_denoised, dtype=jnp.complex64)

        # --- Adam update for k_scales ---
        m_k_next = beta1 * m_k + (1 - beta1) * grad_k_scales
        v_k_next = beta2 * v_k + (1 - beta2) * jnp.square(grad_k_scales)
        m_k_hat = m_k_next / (1 - beta1**t_next)
        v_k_hat = v_k_next / (1 - beta2**t_next)
        update_step_k = learning_rate * m_k_hat / (jnp.sqrt(v_k_hat) + epsilon)
        k_scales_next = k_scales - update_step_k

        error = jnp.linalg.norm(rho_H_grid_next - rho_H_prev_grid) / (jnp.linalg.norm(rho_H_prev_grid) + 1e-9)
        return (rho_H_grid_next, k_scales_next, m_F_next, v_F_next, m_k_next, v_k_next, t_next), error

    if grid is None:
        rho_H_grid_init = jnp.zeros(shape=grid_shape, dtype=jnp.complex64)
        k_scales_init = jnp.zeros(shape=measured_Is.shape[0])
    else:
        rho_H_grid_init = jnp.array(grid)
        k_scales_init = jnp.array(k_scales)

    t_init = 0
    m_F_init, v_F_init = jnp.zeros_like(rho_H_grid_init), jnp.zeros_like(rho_H_grid_init)
    m_k_init, v_k_init = jnp.zeros_like(k_scales_init), jnp.zeros_like(k_scales_init)
    initial_carry = (rho_H_grid_init, k_scales_init, m_F_init, v_F_init, m_k_init, v_k_init, t_init)

    final_carry, _ = jax.lax.scan(adam_body_fun, initial_carry, None, length=num_iterations)
    final_rho_H_grid = final_carry[0]
    final_k_scales = final_carry[1]

    final_F_H_grid_jax = jnp.fft.fftn(final_rho_H_grid)
    final_F_H_grid_np = np.array(final_F_H_grid_jax, dtype=np.complex128)
    Fh_1d = np.array([final_F_H_grid_np[tuple(k)] for k in indices])
    return miller.array(f_heavy_arr.set(), data=flex.complex_double(Fh_1d)), final_rho_H_grid, final_k_scales

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

import numpy as np
import jax.numpy as jnp
import pandas as pd
import h5py
import scipy.linalg
from scipy.spatial.transform import Rotation
from iotbx.reflection_file_reader import any_reflection_file
from mmtbx.model import manager as model_manager
from cctbx.array_family import flex
import iotbx.pdb
from cctbx import miller, crystal
from . import active_loop_lib as al

class CoverageCalculator:
    """
    Calculates HKL coverage for a given orientation based on an instrument mask.
    This logic is adapted from precompute_angles.py.
    """
    def __init__(self, instrument_mask_file: str, reflection_file: str):
        print("Initializing CoverageCalculator...")
        # 1. Load instrument mapping
        with h5py.File(instrument_mask_file, 'r') as f:
            data = f['MDHistoWorkspace/data']
            Qx, Qy, Qz = [data[f'D{i}'][()] for i in range(3)]
            coverage = data['signal'][()].T
        
        Qx, Qy, Qz = [0.5*(comp[1:]+comp[:-1]) for comp in [Qx, Qy, Qz]]
        Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing='ij')

        mask = coverage > 0
        self.Q_lab = np.vstack([Qx[mask], Qy[mask], Qz[mask]]) # Shape (3, N)
        print(f"Loaded {self.Q_lab.shape[1]} covered points from instrument mask.")

        # 2. Load crystal info and calculate B matrix
        hkl_file = any_reflection_file(reflection_file)
        miller_array = hkl_file.as_miller_arrays()[0]
        cryst_symm = miller_array.crystal_symmetry()
        
        self.hkls_all_set = set(miller_array.indices())
        print(f"Loaded {len(self.hkls_all_set)} unique HKLs from reflection file.")

        uc = cryst_symm.unit_cell()
        astar, bstar, cstar, alphastar, betastar, gammastar = uc.reciprocal_parameters()
        Gstar = np.array([
            [astar**2, astar*bstar*np.cos(np.deg2rad(gammastar)), astar*cstar*np.cos(np.deg2rad(betastar))],
            [astar*bstar*np.cos(np.deg2rad(gammastar)), bstar**2, bstar*cstar*np.cos(np.deg2rad(alphastar))],
            [astar*cstar*np.cos(np.deg2rad(betastar)), bstar*cstar*np.cos(np.deg2rad(alphastar)), cstar**2]
        ])
        self.B = scipy.linalg.cholesky(Gstar)

    def get_covered_hkls(self, R: np.ndarray, U: np.ndarray) -> set:
        """
        Calculates the set of covered HKLs for a given R and U matrix.
        """
        RUB_inv = np.linalg.inv(R @ U @ self.B)
        
        # Map instrument coverage to hkl space
        # Q_lab is (3, N), RUB_inv is (3, 3)
        hkl_cov_raw = np.einsum('ij,jk->ik', RUB_inv / (2 * np.pi), self.Q_lab).round(0).astype(int).T
        
        current_hkl_set = set(map(tuple, np.unique(hkl_cov_raw, axis=0)))
        
        # Intersect with the full list of possible hkls
        return current_hkl_set.intersection(self.hkls_all_set)


class ActiveLearningLoop:
    """Manages the state and logic for a single active learning experiment."""

    def __init__(self, instrument_mask_file: str, reflection_file: str, pdb_file: str, 
                 polarization_files_csv: str, mtz_array_label: str, h_only_file: str = None):

        # --- CCTBX Initialization (same as before) ---
        self.miller_set = any_reflection_file(reflection_file).as_miller_arrays(merge_equivalents=False)[0]
        pdb_inp = iotbx.pdb.input(file_name=pdb_file)
        model = model_manager(model_input=pdb_inp)
        model.get_hierarchy().remove_hd()
        xrs = model.get_xray_structure()
        xrs.switch_to_neutron_scattering_dictionary()
        f_heavy_calc = xrs.structure_factors(d_min=self.miller_set.d_min(), algorithm='direct').f_calc()
        f_heavy_calc, self.miller_set = f_heavy_calc.common_sets(other=self.miller_set)

        p1_map = self.miller_set.expand_to_p1()
        self.f_heavy_map = f_heavy_calc.expand_to_p1()
        map_shape = p1_map.fft_map(resolution_factor=1./3).real_map_unpadded().all()
        self.n_voxels = np.prod(map_shape)
        self.p1_indices = list(p1_map.indices())

        self.f_h_only_ref = None
        f_h_only_ref = None
        if h_only_file is not None:
            f_h_only_ref_native = any_reflection_file(h_only_file).as_miller_arrays()[0]
            f_h_only_ref = f_h_only_ref_native.expand_to_p1()
            self.f_h_only_ref, self.f_heavy_map = f_h_only_ref.common_sets(other=self.f_heavy_map)

        # --- New Coverage and Data Pool Logic ---
        self.coverage_calculator = CoverageCalculator(instrument_mask_file, reflection_file)
        self.pool_data = al.load_full_dataset(polarization_files_csv, mtz_array_label)
        self.unmeasured_data = self.pool_data.copy()
        self.measured_data = pd.DataFrame()
        self.polarization_states = self.pool_data['phi_pol'].unique()

        # --- State Initialization ---
        self.U = np.eye(3) # Assume default orientation matrix for now
        self.next_state = None
        self.current_uncertainty = float('inf')
        self.F_H = None
        self.H_grid = None
        self.is_first_step = True
        self._calculate_next_step()

    def _generate_random_rotation(self) -> np.ndarray:
        """Generates a single random 3x3 rotation matrix."""
        return Rotation.random().as_matrix()

    def add_measurements(self, measurements: list):
        """Integrates new measurements and triggers the next calculation."""
        if self.next_state is None:
            raise ValueError("No next state was calculated. Cannot add measurements.")

        R = self.next_state['rotation_matrix']
        phi_pol = self.next_state['phi_pol']

        # Find the data corresponding to this state
        covered_hkls = self.coverage_calculator.get_covered_hkls(R, self.U)
        
        batch_mask = (self.unmeasured_data['hkl'].isin(covered_hkls)) & \
                     (self.unmeasured_data['phi_pol'] == phi_pol)

        batch_to_add = self.unmeasured_data[batch_mask].copy()

        # In a real system, you'd merge 'measurements' here.
        # For simulation, we just move the "perfect" data.
        # We also store the rotation matrix used for this measurement.
        batch_to_add['rotation_matrix'] = [R.tolist()] * len(batch_to_add)

        self.measured_data = pd.concat([self.measured_data, batch_to_add])
        self.unmeasured_data = self.unmeasured_data.drop(batch_to_add.index)

        self.is_first_step = False
        self._calculate_next_step()

    def _calculate_next_step(self):
        """Runs one cycle of reconstruction and scoring."""
        if self.is_first_step:
            # For the very first step, just pick a random orientation
            print("First step: Picking a random initial orientation.")
            R_cand = self._generate_random_rotation()
            phi_pol = self.polarization_states[0]
            self.next_state = {
                "rotation_matrix": R_cand,
                "phi_pol": phi_pol,
                "U": self.U
            }
            self.current_uncertainty = float('inf')
            return

        # 1. Reconstruct Map
        # Create miller_arrays from the measured_data DataFrame

        # Ensure the 'hkl' column exists before grouping
        if 'hkl' not in self.measured_data.columns and not self.measured_data.empty:
            self.measured_data['hkl'] = self.measured_data[['h','k','l']].apply(tuple, axis=1)
        miller_arrays = []
        indices = self.miller_set.indices()
        for phi in self.measured_data['phi_pol'].unique():
            I_hkl = self.measured_data.groupby(['hkl','phi_pol'])['I'].mean()
            I = [ I_hkl.get((tuple(k), phi), 0) for k in indices ]
            sigI_hkl = self.measured_data.groupby(['hkl','phi_pol'])['sigI'].mean()
            sigI = [ sigI_hkl.get((tuple(k), phi), -1) for k in indices ]
            m = miller.array(self.miller_set, data=flex.double(I), sigmas=flex.double(sigI))
            miller_arrays.append(m)

        self.F_H, self.H_grid = al.phase_retrieval_adam_direct(
            miller_arrays, self.polarization_states, self.f_heavy_map, grid=self.H_grid,
        )
        
        # 2. Calculate Current Uncertainty
        self.current_uncertainty = al.calculate_trace_of_covariance_direct_blocked(
            self.measured_data, self.F_H, self.f_heavy_map, self.p1_indices, self.n_voxels
        )
        
        # 3. Score Candidates
        self._score_candidate_orientations()

    def _score_candidate_orientations(self, n_candidates=20, epsilon=1.0):
        """Scores candidate states by efficiently calculating the change in the trace of the covariance matrix."""
        
        # Get baseline uncertainty contributions
        base_groups = al.prepare_friedel_groups(self.measured_data, self.F_H, self.f_heavy_map, self.p1_indices)
        base_trace_contributions = {}
        total_trace_contrib_base = 0.0
        M_base = 0
        for key, group_data in base_groups.items():
            const_block = jnp.array(group_data['consts'])
            weight_block = jnp.array(group_data['weights'])
            hkls_in_block = jnp.array(group_data['hkls'])
            M_base += len(const_block)
            trace_contrib = al._calculate_block_trace_jax(const_block, weight_block, self.n_voxels, hkls_in_block, epsilon)
            base_trace_contributions[key] = trace_contrib
            total_trace_contrib_base += trace_contrib
        
        trace_base = ((self.n_voxels - M_base) / epsilon) + ((1.0 / epsilon) * total_trace_contrib_base)
        
        best_state = None
        min_predicted_trace = trace_base

        for _ in range(n_candidates):
            R_cand = self._generate_random_rotation()
            covered_hkls = self.coverage_calculator.get_covered_hkls(R_cand, self.U)
            
            for phi_pol in self.polarization_states:
                batch_mask = (self.unmeasured_data['hkl'].isin(covered_hkls)) & \
                             (self.unmeasured_data['phi_pol'] == phi_pol)
                batch_to_add = self.unmeasured_data[batch_mask]

                if batch_to_add.empty:
                    continue

                # Calculate the trace contribution from this new batch
                candidate_groups = al.prepare_friedel_groups(batch_to_add, self.F_H, self.f_heavy_map, self.p1_indices)
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
                    new_trace_contrib = al._calculate_block_trace_jax(const_block_new, weight_block_new, self.n_voxels, hkls_in_block_new, epsilon)
                    predicted_trace_contrib_total += (new_trace_contrib - old_trace_contrib)
                
                predicted_trace = ((self.n_voxels - M_candidate) / epsilon) + ((1.0 / epsilon) * predicted_trace_contrib_total)

                if predicted_trace < min_predicted_trace:
                    min_predicted_trace = predicted_trace
                    best_state = {
                        "rotation_matrix": R_cand,
                        "phi_pol": phi_pol,
                        "U": self.U
                    }

        if best_state is None and not self.unmeasured_data.empty:
            # Fallback: just pick a random valid state from unmeasured data
            print("No optimal state found, picking random valid state.")
            random_sample = self.unmeasured_data.sample(n=1).iloc[0]
            R_cand = self._generate_random_rotation() # This isn't perfect, but it's a fallback
            phi_pol = random_sample['phi_pol']
            best_state = {
                "rotation_matrix": R_cand,
                "phi_pol": phi_pol,
                "U": self.U
            }
            
        self.next_state = best_state

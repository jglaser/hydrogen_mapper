import numpy as np
import pandas as pd
from . import active_loop_gemmi_lib as al

class ActiveLearningLoop:
    """Manages the state and logic for a single active learning experiment."""

    def __init__(self, events_file: str, reflection_file: str, pdb_file: str, polarization_files_csv: str, mtz_array_label: str):
        self.miller_set, self.f_heavy_map, self.p1_indices, self.n_voxels, self.f_h_only_ref = al.prepare_phasing_data(
            reflection_file, pdb_file, None
        )
        self.pool_data = al.load_full_dataset(polarization_files_csv, mtz_array_label)
        self.unmeasured_data = self.pool_data.copy()
        self.measured_data = pd.DataFrame()

        coverage_data = np.load(events_file)
        hkl_map_coverage = coverage_data['hkl_map']
        event_log = coverage_data['event_log']
        gonio_angles_to_search = np.unique(event_log[:, 0])
        
        hkls_heavy = np.array(self.f_heavy_map.indices)
        f_heavy_abs = np.abs(self.f_heavy_map.data)
        hkl_to_fheavy_map = {tuple(hkl): amp for hkl, amp in zip(hkls_heavy, f_heavy_abs)}
        
        optimal_gonio_angle = al.find_optimal_initial_goniometer_angle(gonio_angles_to_search, hkl_map_coverage, event_log, hkl_to_fheavy_map)
        
        self.next_state = {
            "goniometer_angle": optimal_gonio_angle,
            "phi_pol": self.pool_data['phi_pol'].unique()[0]
        }
        self.current_uncertainty = float('inf')
        self.F_H = None


    def add_measurements(self, measurements: list):
        """Integrates new measurements and triggers the next calculation."""
        batch_to_add = self.unmeasured_data[
            (self.unmeasured_data['goniometer_angle'] == self.next_state['goniometer_angle']) &
            (self.unmeasured_data['phi_pol'] == self.next_state['phi_pol'])
        ]
        self.measured_data = pd.concat([self.measured_data, batch_to_add])
        self.unmeasured_data = self.unmeasured_data.drop(batch_to_add.index)

        self._calculate_next_step()

    def _calculate_next_step(self):
        """Runs one cycle of reconstruction and scoring."""
        self.F_H = al.phase_retrieval_adam_direct(
            self.measured_data, self.pool_data['phi_pol'].unique(), self.f_heavy_map
        )
        self.current_uncertainty = al.calculate_trace_of_covariance_direct_blocked(
            self.measured_data, self.F_H, self.f_heavy_map, self.n_voxels
        )
        best_state = al.score_candidate_states_optimized(
            self.unmeasured_data, self.measured_data, self.F_H, self.f_heavy_map, self.n_voxels
        )
        if best_state is not None:
            self.next_state = {
                "goniometer_angle": best_state['goniometer_angle'],
                "phi_pol": best_state['phi_pol']
            }

import numpy as np
import pandas as pd
from iotbx.reflection_file_reader import any_reflection_file
from mmtbx.model import manager as model_manager
import iotbx.pdb
from cctbx import miller, crystal
from . import active_loop_lib as al

class ActiveLearningLoop:
    """Manages the state and logic for a single active learning experiment."""

    def __init__(self, events_file: str, reflection_file: str, pdb_file: str, polarization_files_csv: str, mtz_array_label: str, h_only_file: str = None):
        # --- Initialization from cctbx-based script ---
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
        self.grid = None
        map_shape = p1_map.fft_map(resolution_factor=1./3).real_map_unpadded().all()
        self.n_voxels = np.prod(map_shape)
        self.p1_indices = list(p1_map.indices())

        self.f_h_only_ref = None
        if h_only_file:
            f_h_only_ref_native = any_reflection_file(h_only_file).as_miller_arrays()[0]
            self.f_h_only_ref = f_h_only_ref_native.expand_to_p1()
            self.f_h_only_ref, self.f_heavy_map = self.f_h_only_ref.common_sets(other=self.f_heavy_map)

        self.pool_data = al.load_full_dataset(polarization_files_csv, mtz_array_label)
        self.unmeasured_data = self.pool_data.copy()
        self.measured_data = pd.DataFrame()

        coverage_data = np.load(events_file)
        hkl_map_coverage = coverage_data['hkl_map']
        event_log = coverage_data['event_log']
        gonio_angles_to_search = np.unique(event_log[:, 0])

        hkls_heavy = np.array(f_heavy_calc.indices())
        f_heavy_abs = np.array(f_heavy_calc.amplitudes().data())
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
        self.F_H, self.grid = al.phase_retrieval_adam_direct(
            self.measured_data, self.pool_data['phi_pol'].unique(), self.f_heavy_map, self.grid,
        )
        self.current_uncertainty = al.calculate_trace_of_covariance_direct_blocked(
            self.measured_data, self.F_H, self.f_heavy_map, self.p1_indices, self.n_voxels
        )
        best_state = al.score_candidate_states_optimized(
            self.unmeasured_data, self.measured_data, self.F_H, self.f_heavy_map, self.p1_indices, self.n_voxels
        )
        if best_state is not None:
            self.next_state = {
                "goniometer_angle": best_state['goniometer_angle'],
                "phi_pol": best_state['phi_pol']
            }

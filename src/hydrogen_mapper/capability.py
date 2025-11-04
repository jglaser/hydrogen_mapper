import uuid
from typing import Dict
from intersect_sdk import (
    IntersectBaseCapabilityImplementation,
    IntersectEventDefinition,
    intersect_message,
    intersect_status,
    intersect_event,
)
from .models import (
    StartLearningRequest, AddMeasurementRequest, ExperimentState, NextMeasurementEvent
)
from .active_learning_loop import ActiveLearningLoop

class ActiveLearningCapability(IntersectBaseCapabilityImplementation):
    intersect_sdk_capability_name = "ActiveLearning"

    def __init__(self):
        super().__init__()
        self.sessions: Dict[uuid.UUID, ActiveLearningLoop] = {}

    @intersect_message()
    def start_experiment(self, request: StartLearningRequest) -> uuid.UUID:
        """Initializes an active learning session and returns its unique ID."""
        session_id = uuid.uuid4()
        loop = ActiveLearningLoop(
            instrument_mask_file=request.instrument_mask_file, # CHANGED
            reflection_file=request.reflection_file,
            pdb_file=request.pdb_file,
            polarization_files_csv=request.polarization_files_csv,
            mtz_array_label=request.mtz_array_label
        )

        self.sessions[session_id] = loop
        return session_id

    def _get_base_experiment_state(self, loop: ActiveLearningLoop, session_id: uuid.UUID) -> ExperimentState:
        """Helper to populate a base ExperimentState object."""
        next_phi = loop.next_state.get('phi_pol') if loop.next_state else None
        
        return ExperimentState(
            session_id=session_id,
            status="Ready for measurement",
            total_measurements=len(loop.measured_data),
            next_goniometer_angle=None, # Stub field
            next_polarization_state=next_phi,
            uncertainty=loop.current_uncertainty
        )

    @intersect_message()
    def get_next_measurement(self, session_id: uuid.UUID) -> ExperimentState:
        """
        [DEPRECATED] This function is a stub. 
        Please use get_next_rotation_matrix.
        """
        loop = self.sessions.get(session_id)
        if not loop:
            raise ValueError("Session ID not found")
        
        # Get base state, but null-out the deprecated field
        state = self._get_base_experiment_state(loop, session_id)
        state.status = "Ready for measurement. [NOTE: Use get_next_rotation_matrix]"
        state.next_goniometer_angle = None 
        return state

    @intersect_message()
    def get_next_rotation_matrix(self, session_id: uuid.UUID) -> ExperimentState:
        """Gets the current state and next suggested measurement (rotation matrix)."""
        loop = self.sessions.get(session_id)
        if not loop:
            raise ValueError("Session ID not found")

        state = self._get_base_experiment_state(loop, session_id)

        # Add the new rotation matrix field
        if loop.next_state and 'rotation_matrix' in loop.next_state:
             # Convert numpy array to list for JSON serialization
            state.next_rotation_matrix = loop.next_state.get('rotation_matrix').tolist()

        return state

    @intersect_event(events={'next_measurement_ready': IntersectEventDefinition(event_type=NextMeasurementEvent)})
    @intersect_message()
    def add_measurements(self, request: AddMeasurementRequest) -> ExperimentState:
        """Adds measurements to a session and triggers the next learning step."""
        loop = self.sessions.get(request.session_id)
        if not loop:
            raise ValueError("Session ID not found")

        loop.add_measurements(request.measurements)

        if loop.next_state:
            self.intersect_sdk_emit_event(
                'next_measurement_ready',
                NextMeasurementEvent(
                    session_id=request.session_id,
                    goniometer_angle=None, # Stub field
                    rotation_matrix=loop.next_state["rotation_matrix"].tolist(),
                    polarization_state=loop.next_state["phi_pol"],
                    estimated_uncertainty=loop.current_uncertainty
                )
            )

        # Return the new state using the new accessor
        return self.get_next_rotation_matrix(request.session_id)

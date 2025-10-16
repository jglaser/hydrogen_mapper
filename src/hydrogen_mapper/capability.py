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
            events_file=request.events_file,
            reflection_file=request.reflection_file,
            pdb_file=request.pdb_file,
            polarization_files_csv='polarization_files.csv',  # Assuming a default name
            mtz_array_label='I'  # Assuming a default label
        )
        self.sessions[session_id] = loop
        return session_id

    @intersect_message()
    def get_next_measurement(self, session_id: uuid.UUID) -> ExperimentState:
        """Gets the current state and next suggested measurement for a session."""
        loop = self.sessions.get(session_id)
        if not loop:
            raise ValueError("Session ID not found")
        return ExperimentState(
            session_id=session_id,
            status="Ready for measurement",
            total_measurements=len(loop.measured_data),
            next_goniometer_angle=loop.next_state["goniometer_angle"],
            next_polarization_state=loop.next_state["phi_pol"],
            uncertainty=loop.current_uncertainty
        )

    @intersect_event(events={'next_measurement_ready': IntersectEventDefinition(event_type=NextMeasurementEvent)})
    @intersect_message()
    def add_measurements(self, request: AddMeasurementRequest) -> ExperimentState:
        """Adds measurements to a session and triggers the next learning step."""
        loop = self.sessions.get(request.session_id)
        if not loop:
            raise ValueError("Session ID not found")

        loop.add_measurements(request.measurements)

        self.intersect_sdk_emit_event(
            'next_measurement_ready',
            NextMeasurementEvent(
                session_id=request.session_id,
                goniometer_angle=loop.next_state["goniometer_angle"],
                polarization_state=loop.next_state["phi_pol"],
                estimated_uncertainty=loop.current_uncertainty
            )
        )

        return self.get_next_measurement(request.session_id)

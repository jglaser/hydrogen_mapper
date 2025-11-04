from pydantic import BaseModel
from typing import List, Optional
import uuid

class StartLearningRequest(BaseModel):
    """Input for starting a new active learning session."""
    instrument_mask_file: str
    reflection_file: str
    pdb_file: str
    polarization_files_csv: str
    mtz_array_label: str

class Measurement(BaseModel):
    """A single measurement of intensity and sigma."""
    intensity: float
    sigma: float

class AddMeasurementRequest(BaseModel):
    """Input for adding new measurements to a session."""
    session_id: uuid.UUID
    measurements: List[Measurement]

class ExperimentState(BaseModel):
    """The current state of an active learning session."""
    session_id: uuid.UUID
    status: str
    total_measurements: int
    next_goniometer_angle: Optional[float] = None # Stub field
    next_polarization_state: Optional[float] = None
    uncertainty: Optional[float] = None
    next_rotation_matrix: Optional[List[List[float]]] = None # <-- ADDED

class NextMeasurementEvent(BaseModel):
    """Event emitted when a new measurement is suggested."""
    session_id: uuid.UUID
    goniometer_angle: Optional[float] = None # <-- MADE OPTIONAL
    rotation_matrix: Optional[List[List[float]]] = None # <-- ADDED
    polarization_state: float
    estimated_uncertainty: float

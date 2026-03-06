from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from utils import AntennaArrayType, AntennaType, RadiationPattern, PolarizationType


class GeoPosition(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")
    alt: float = Field(..., description="Altitude in meters")

    def to_tuple(self):
        return (self.lat, self.lon, self.alt)

    @classmethod
    def from_tuple(cls, pos: tuple):
        return cls(lat=pos[0], lon=pos[1], alt=pos[2])


class Orientation(BaseModel):
    x: float
    y: float
    z: float

    def to_tuple(self):
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, pos: tuple):
        return cls(x=pos[0], y=pos[1], z=pos[2])


class AntennaArrayConfig(BaseModel):
    antenna_type: str = Field(..., description="Type of antenna: 'tx' or 'rx'")
    num_rows: int = Field(1, ge=1, description="Number of antenna rows")
    num_cols: int = Field(1, ge=1, description="Number of antenna columns")
    vertical_spacing: float = Field(
        0.5, gt=0, description="Vertical spacing between elements (in wavelengths)"
    )
    horizontal_spacing: float = Field(
        0.5, gt=0, description="Horizontal spacing between elements (in wavelengths)"
    )
    pattern: str = Field(
        "tr38901", description="Radiation pattern: 'iso', 'dipole', or 'tr38901'"
    )
    polarization: str = Field(
        "V", description="Polarization type: 'V', 'H', or 'cross'"
    )

    def to_class(self):
        return AntennaArrayType(self.antenna_type, 
                                self.num_rows, 
                                self.num_cols, 
                                self.vertical_spacing, 
                                self.horizontal_spacing, 
                                self.pattern, 
                                self.polarization)

    @classmethod
    def from_class(self, aa: AntennaArrayType):
        return self(aa.antenna_type, aa.planar_array.num_rows, aa.num_cols,
                    aa.horizontal_spacing, aa.vertical_spacing,
                    aa.radiation_pattern, aa.polarization_type)


class AntennaArrayResponse(BaseModel):
    """
    Maybe add more state to this later,
    but this is all the state that Sionna keeps
    """
    antenna_type: str
    num_antennas: int


class TransmitterCreate(BaseModel):
    """
    Creates a new transmitter in the scene
    """
    name: str = Field(..., description="Unique identifier for the device")
    position: GeoPosition
    orientation: Optional[Orientation] = None


class DeviceUpdate(BaseModel):
    position: GeoPosition
    orientation: Optional[Orientation] = None


class DeviceResponse(BaseModel):
    name: str
    position: GeoPosition
    orientation: Optional[Orientation] = None


class PathComputationRequest(BaseModel):
    max_depth: int = Field(
        3, ge=1, le=10, description="Maximum number of reflections/diffractions"
    )
    num_samples: int = Field(
        1e5, ge=1, le=1e8, description="Number of rays used in path computation"
    )


class PathComputationResponse(BaseModel):
    """Paths are only quantified via the channel impulse response, not this response"""
    
    path_count: int
    max_depth: int
    message: str = "Paths computed successfully"


class CirGains(BaseModel):
    """Complex channel gains with multiple representations"""

    real: List = Field(description="Real part of complex gains")
    imag: List = Field(description="Imaginary part of complex gains")
    magnitude: List = Field(description="Magnitude of complex gains")
    phase: List = Field(description="Phase of complex gains (radians)")


class CirShape(BaseModel):
    """Shape information for CIR arrays"""

    num_rx: int = Field(description="Number of receivers")
    num_rx_ant: int = Field(description="Number of receiver antennas")
    num_tx: int = Field(description="Number of transmitters")
    num_tx_ant: int = Field(description="Number of transmitter antennas")
    num_paths: int = Field(description="Number of propagation paths")
    num_time_steps: int = Field(description="Number of time steps")


class CirResponse(BaseModel):
    delays: List = Field(
        default_factory=list,
        description="Path delays in seconds [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]",
    )
    gains: CirGains = Field(description="Complex path gains")
    shape: CirShape = Field(description="Dimensions of the CIR arrays")
    message: str = "CIR retrieved successfully"


class SceneInfoResponse(BaseModel):
    object_count: int
    objects: List[str]
    transmitter_count: int
    receiver_count: int
    coordinate_reference: GeoPosition = Field(
        description="Lat/lon/alt origin used for local Sionna coordinate conversion"
    )


class SceneCreateRequest(BaseModel):
    scene_path: Optional[str] = Field(
        None, description="Optional custom scene path. Defaults to bundled Sionna scene."
    )


class SceneCreateResponse(BaseModel):
    scene_id: str
    message: str = "Scene created successfully"


class MessageResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: str


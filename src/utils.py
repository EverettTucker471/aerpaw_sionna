from enum import Enum
from typing import Optional, Final, Tuple, Dict
from sionna.rt import PlanarArray
from pyproj import Transformer
from pyproj.enums import TransformDirection

# Position of LW1 in lat/lon/alt
# TODO: This should be different, based on the lake wheeler scene
ORIGIN_LAT_LON: Final[Dict[str, float]] = {"lat": 35.72750947, "lon": -78.69595819, "alt": 82.973}


class AntennaType(Enum):
    """
    Type of Antenna (transmitter and receiver)
        Used in setting arrays, transmitter/receiver characteristics
    """

    Transmitter = "tx"
    Receiver = "rx"

    @classmethod
    def to_enum(cls, s: str):
        if s == "tx":
            return AntennaType.Transmitter
        elif s == "rx":
            return AntennaType.Receiver
        else:
            raise Exception(f"Invalid input for Antenna {s}, must be 'tx' or 'rx")


class RadiationPattern(Enum):
    """radiation patterns available in sionna"""

    ISO = "iso"
    DIPOLE = "dipole"
    DIRECTIONAL = "tr38901"


class PolarizationType(Enum):
    """Type of Polarization available"""

    VERTICAL = "V"
    HORIZONTAL = "H"
    SLANT = "VH"
    CROSS = "cross"


class AntennaArrayType():
    def __init__(self, 
                 antenna_type: AntennaType, 
                 num_rows: Optional[int] = None, 
                 num_cols: Optional[int] = None, 
                 h_space: Optional[float] = None, 
                 v_space: Optional[float] = None, 
                 pattern: Optional[RadiationPattern] = None, 
                 polarization: Optional[PolarizationType] = None,
                 planar_array: Optional[PlanarArray] = None):
        self.antenna_type = antenna_type
        if planar_array is None:
            self.planar_array = PlanarArray(num_rows=num_rows, num_cols=num_cols,
                                            horizontal_spacing=h_space,
                                            vertical_spacing=v_space,
                                            pattern=pattern.value,
                                            polarization=polarization.value)
        else:
            self.planar_array = planar_array
        

    def to_sionna(self):
        return self.planar_array


    @classmethod
    def from_sionna(cls, antenna_type: str, planar_array: PlanarArray):
        return AntennaArrayType(antenna_type=AntennaType.to_enum(s=antenna_type), 
                                planar_array=planar_array)


class CoordinateConverter:
    """WGS84 converter between geodetic (lat/lon/alt) and local ENU coordinates."""

    def __init__(self, reference_origin: Optional[Dict[str, float]] = None):
        if not reference_origin:
            reference_origin = ORIGIN_LAT_LON
        self.origin = reference_origin

        pipeline = (
            f"+proj=pipeline "
            f"+step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m " # Step 1: Degrees to Radians
            f"+step +proj=cart +ellps=WGS84 "                                   # Step 2: Geographic to Geocentric
            f"+step +proj=topocentric +ellps=WGS84 "                            # Step 3: Geocentric to Topocentric ENU
            f"+lon_0={self.origin['lon']} +lat_0={self.origin['lat']} +h_0={self.origin['alt']}"
        )

        self.transformer = Transformer.from_pipeline(pipeline)


    def update_reference_origin(self, origin: Dict[str, float]) -> Dict[str, float]:
        self.origin = origin
        pipeline = (
            f"+proj=pipeline "
            f"+step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m " # Step 1: Degrees to Radians
            f"+step +proj=cart +ellps=WGS84 "                                   # Step 2: Geographic to Geocentric
            f"+step +proj=topocentric +ellps=WGS84 "                            # Step 3: Geocentric to Topocentric ENU
            f"+lon_0={self.origin['lon']} +lat_0={self.origin['lat']} +h_0={self.origin['alt']}"
        )

        self.transformer = Transformer.from_pipeline(pipeline)
        return self.origin


    def get_origin(self) -> Dict[str, float]:
        return self.origin


    def lat_lon_alt_to_local(
        self, lat: float, lon: float, alt: float
    ) -> Tuple[float, float, float]:
        """Convert geodetic coordinate to local ENU tuple (x=east, y=north, z=up)."""
        east, north, up = self.transformer.transform(lon, lat, alt, direction=TransformDirection.FORWARD)
        return (east, north, up)


    def local_to_lat_lon_alt(
        self, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        lon, lat, alt = self.transformer.transform(x, y, z, direction=TransformDirection.INVERSE)
        return (lat, lon, alt)
    

if __name__ == '__main__':
    cc = CoordinateConverter()
    ref = (35.6, -78, 100)
    local_coord = cc.lat_lon_alt_to_local(ref[0], ref[1], ref[2])
    output = cc.local_to_lat_lon_alt(local_coord[0], local_coord[1], local_coord[2])
    print(local_coord)
    print(output)

from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status

import main
from schemas import (
    CirGains,
    CirResponse,
    CirShape,
    TransmitterCreate,
    TransmitterUpdate,
    ReceiverCreate,
    ReceiverUpdate,
    DeviceResponse,
    GeoPosition,
    MessageResponse,
    Vector3D,
    PathComputationRequest,
    PathComputationResponse,
    SceneCreateRequest,
    SceneCreateResponse,
    SceneInfoResponse,
    StatusResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Sionna simulation...")
    try:
        await main.initialize()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        raise
    yield
    print("Shutting down...")
    await main.shutdown()


app = FastAPI(
    title="Sionna RT API",
    description="API for ray tracing simulation using Sionna",
    version="1.0.0",
    lifespan=lifespan,
)


def _raise_scene_not_found(scene_id: str):
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"Scene '{scene_id}' not found"
    )


@app.get("/", response_model=StatusResponse, tags=["Health"])
def root():
    return StatusResponse(status="running")


@app.post(
    "/scenes",
    response_model=SceneCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Scene"],
)
async def create_scene(payload: Optional[SceneCreateRequest] = None):
    try:
        scene_id = await main.create_scene(payload.scene_path if payload else None,
                                           payload.temperature if payload else None,
                                           payload.bandwidth if payload else None,
                                           payload.tx_array.to_type() if payload else None,
                                           payload.rx_array.to_type() if payload else None)
        return SceneCreateResponse(scene_id=scene_id)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scene: {str(e)}",
        )


@app.put("/scenes/{scene_id}/update_origin", response_model=GeoPosition, tags=["Scene"])
async def update_origin(scene_id: str, new_origin: GeoPosition):
    try:
        result = await main.update_origin(scene_id, GeoPosition.to_tuple(new_origin))
        return GeoPosition.from_tuple(result)
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scene info: {str(e)}",
        )


@app.get("/scenes/{scene_id}", response_model=SceneInfoResponse, tags=["Scene"])
async def get_scene(scene_id: str):
    try:
        return await main.get_scene_info(scene_id)
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scene info: {str(e)}",
        )


@app.post("/scenes/{scene_id}/reset", response_model=MessageResponse, tags=["Scene"])
async def reset_scene(scene_id: str):
    try:
        await main.reset_scene(scene_id)
        return MessageResponse(message="Scene reset successfully")
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset scene: {str(e)}",
        )


@app.post(
    "/scenes/{scene_id}/transmitters",
    response_model=DeviceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Transmitters"],
)
async def add_tx(scene_id: str, device: TransmitterCreate):
    """Add a new transmitter to the scene."""
    try:
        result = await main.add_transmitter(
            scene_id,
            device.name,
            device.position.to_tuple(),
            device.signal_power,
            device.velocity.to_tuple(),
            device.orientation.to_tuple() if device.orientation else None,
        )
        return DeviceResponse(
            name=result["name"],
            type="tx",
            position=GeoPosition.from_tuple(result["position"]),
            velocity=Vector3D.from_tuple(pos=result["velocity"]),
            signal_power=result["signal_power"],
            orientation=Vector3D.from_tuple(result["orientation"]),
        )
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transmitter data: {str(e)}",
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add transmitter: {str(e)}",
        )


@app.get("/scenes/{scene_id}/transmitters", response_model=List[str], tags=["Transmitters"])
async def list_tx(scene_id: str):
    """List all transmitters in the scene."""
    try:
        return await main.get_transmitters(scene_id)
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve transmitters: {str(e)}",
        )


@app.put(
    "/scenes/{scene_id}/transmitters/{name}",
    response_model=DeviceResponse,
    tags=["Transmitters"],
)
async def update_tx(scene_id: str, name: str, data: TransmitterUpdate):
    """Update transmitter position."""
    try:
        result = await main.update_transmitter(
            scene_id,
            name, 
            data.position.to_tuple() if data.position else None,
            data.signal_power,
            data.velocity.to_tuple() if data.velocity else None,
            data.orientation.to_tuple() if data.orientation else None,
        )
        return DeviceResponse(
            name=result["name"], 
            type="tx",
            position=GeoPosition.from_tuple(result["position"]) if result["position"] else None,
            velocity=Vector3D.from_tuple(result["velocity"]) if result["velocity"] else None,
            signal_power=result["signal_power"],
            orientation=Vector3D.from_tuple(result["orientation"]) if result["orientation"] else None,
        )
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transmitter '{name}' not found",
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update transmitter: {str(e)}",
        )


@app.post(
    "/scenes/{scene_id}/receivers",
    response_model=DeviceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Receivers"],
)
async def add_rx(scene_id: str, device: ReceiverCreate):
    """Add a new receiver to the scene."""
    try:
        result = await main.add_receiver(
            scene_id,
            device.name,
            device.position.to_tuple(),
            device.velocity.to_tuple(),
            device.orientation.to_tuple() if device.orientation else None,
        )
        return DeviceResponse(
            name=result["name"],
            type="rx",
            position=GeoPosition.from_tuple(result["position"]),
            signal_power=None,
            velocity=Vector3D.from_tuple(result["velocity"]),
            orientation=Vector3D.from_tuple(result["orientation"]),
        )
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid receiver data: {str(e)}",
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add receiver: {str(e)}",
        )


@app.get("/scenes/{scene_id}/receivers", response_model=List[str], tags=["Receivers"])
async def list_rx(scene_id: str):
    """List all receivers in the scene."""
    try:
        return await main.get_receivers(scene_id)
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve receivers: {str(e)}",
        )


@app.put(
    "/scenes/{scene_id}/receivers/{name}",
    response_model=DeviceResponse,
    tags=["Receivers"],
)
async def update_rx(scene_id: str, name: str, data: ReceiverUpdate):
    """Update receiver position."""
    try:
        result = await main.update_receiver(
            scene_id,
            name, 
            data.position.to_tuple() if data.position else None,
            data.velocity.to_tuple() if data.velocity else None,
            data.orientation.to_tuple() if data.orientation else None,
        )
        return DeviceResponse(
            name=result["name"], 
            type="rx",
            position=GeoPosition.from_tuple(result["position"]),
            signal_power=None,
            velocity=Vector3D.from_tuple(result["velocity"]),
            orientation=Vector3D.from_tuple(result["orientation"]),
        )
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Receiver '{name}' not found"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update receiver: {str(e)}",
        )


@app.post(
    "/scenes/{scene_id}/simulation/paths",
    response_model=PathComputationResponse,
    tags=["Simulation"],
)
async def compute_paths(scene_id: str, params: PathComputationRequest):
    try:
        result = await main.compute_paths(scene_id, params.max_depth, params.num_samples)
        return PathComputationResponse(
            path_count=result["path_count"], 
            max_depth=result["max_depth"],
            num_samples=result["num_samples"],
        )
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid path computation parameters: {str(e)}",
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute paths: {str(e)}",
        )


@app.get("/scenes/{scene_id}/simulation/cir", response_model=CirResponse)
async def get_cir(scene_id: str):
    """Retrieve the Channel Impulse Response (CIR)."""
    try:
        result = await main.get_cir(scene_id)
        return CirResponse(
            delays=result["delays"],
            gains=CirGains(**result["gains"]),
            shape=CirShape(**result["shape"]),
        )
    except main.SceneNotFoundError:
        _raise_scene_not_found(scene_id)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve CIR: {str(e)}",
        )

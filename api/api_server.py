# api/api_server.py
from fastapi import FastAPI, HTTPException, Query, Request, Path, Body
from pydantic import BaseModel, Extra

from batch import BatchPipeline
from batch import PipelineManager
from typing import List, Literal, Optional, Dict, Any
import pandas as pd
import json
from config import ConfigProvider, config_manager
from config import ConfigManager
from validation.gx_init import GXInitializer
from profiling import DataProfilerUpdated, ProfilerConfig



app = FastAPI(title="Data Ingestion API")

class LoosePayload(BaseModel):
    """
    A catch-all payload model that allows any extra fields.
    """
    class Config:
        extra = Extra.allow

@app.post("/ingest/{topic}")
async def ingest_data_item(
    payload: LoosePayload,
    request: Request,
    topic: str = Path(..., description="MQTT topic, e.g. /air-quality")
):
    """
    Receives POSTs to /ingest/{topic}, where:
      - `topic` is taken from the URL path (supports slashes via `{topic:path}`).
      - `payload` is any JSON object, with fields dynamically allowed.
    Dispatches into the same BatchPipeline(s) as MQTT.
    """
    manager = request.app.state.manager
    ok = manager.dispatch(topic, payload.dict())
    if not ok:
        raise HTTPException(status_code=404, detail=f"No pipeline for topic '{topic}'")
    return {"status": "queued", "topic": topic}




@app.post("/ingest/{topic}/sync")
async def ingest_data_batch(
    request: Request,
    topic: str = Path(..., description="MQTT topic, e.g. /air-quality"),
    config_id: str = Query(None, description="Optional config ID to use for processing"),
    payloads: List[LoosePayload] = Body(
        ..., 
        description="Array of records to process"
    ),
):

    config_name = f"{config_id}_{topic}"
    pipeline = BatchPipeline(topic, config_name, batch_size=1) # batch size of 1 for synchronous processing irrelevant

    # Convert your Pydantic models into plain dicts
    raw_dicts = [p.dict() for p in payloads]
    df = pd.DataFrame(raw_dicts)

    

    # Run your synchronous processing
    cleaned_df = pipeline.process_sync(df)
    cleaned = json.loads(cleaned_df.to_json(orient="records"))

    return {
        "status": "processed",
        "topic": topic,
        "count": len(cleaned),
        "cleaned": cleaned,
    }



@app.get("/configs/{cfg_type}", 
         summary="List configurations",
        description="Retrieve configurations filtered by kind. Use pagination via limit/offset.")
async def get_configs(
    cfg_type: Optional[Literal["mqtt", "validation"]] = Path(..., description="Filter to a specific config type")
):
    """
    Return the currently saved configurations from disk.

    - type=mqtt        -> all JSON files in 'config/' (mapping: filename -> content)
    - type=validation  -> all JSON files in 'config/validations/'
    - (no type)        -> both, as { "mqtt": {...}, "validation": {...} }
    """
    try:
        provider = ConfigProvider()

        if cfg_type == "mqtt":
            mqtt_cfg = provider.mqtt() 
            return {"mqtt": mqtt_cfg}

        if cfg_type == "validation":
            validation_cfg = provider.validation() 
            return {"validation": validation_cfg}

        # default: both
        mqtt_cfg = provider.mqtt()
        validation_cfg = provider.validation()
        return {"mqtt": mqtt_cfg, "validation": validation_cfg}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={str(e)})
    except json.JSONDecodeError as e:
        # surface which file was malformed if available
        raise HTTPException(status_code=422, detail=f"Invalid JSON in config files: {e}")
    
@app.post("/configs/{cfg_type}")
@app.post("/configs/{cfg_type}/{cfg_id}")
async def ingest_config(
    request: Request,
    cfg_type: Literal["mqtt", "validation"],
    payload: Dict[str, Any] = Body(...),
    cfg_id: str | None = None,   # <-- just a plain optional argument
):

    config_manager = ConfigManager(base_path="config")
    
    try:
        target = config_manager.write_atomic(cfg_type, cfg_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
    

    # reload gx and pipelines
    try:
        provider = ConfigProvider()

        if cfg_type == "mqtt":
            request.app.state.manager.reload_from_provider(provider)
        elif cfg_type == "validation":
            gx_initializer: GXInitializer =  request.app.state.gx
            gx_initializer.reload_gx()
        else:
            raise HTTPException(status_code=400, detail="Invalid configuration type. Use 'mqtt' or 'validation'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload configurations: {e}")
    

    return {
        "status": "Configuration saved",
        "type": cfg_type
    }


@app.delete("/configs/{cfg_type}/{cfg_id}",
            summary="Delete a configuration",
            description="Delete a specific configuration by type and ID.")
async def delete_config(
    request: Request,
    cfg_type: Literal["mqtt", "validation"] = Path(..., description="Type of configuration to delete"),
    cfg_id: Optional[str] = Path(..., description="ID of the configuration to delete (file stem for validation)")
):
    # only allow deleting a single configuration state file
    if cfg_type not in ["mqtt", "validation"]:
        raise HTTPException(status_code=400, detail="Invalid configuration type. Use 'mqtt' or 'validation'.")
    if cfg_type == "validation" and not cfg_id:
        raise HTTPException(status_code=400, detail="Validation config requires a file-stem in the path.")
    if cfg_type == "mqtt":
        # keep this explicit for now; you can enable later if desired
        raise HTTPException(status_code=400, detail="MQTT config deletion is disabled via this endpoint.")

    manager = ConfigManager(base_path="config")
    try:
        removed_paths = manager.delete("validation", cfg_id, missing_ok=False, pipelines = request.app.state.manager.pipelines)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete config: {e}")

    # reload GX after modification to validation states
    try:
        gx = request.app.state.gx
        gx.reload_gx()
    except AttributeError:
        raise HTTPException(status_code=500, detail="GX initializer missing on app.state (app.state.gx).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload GX after deletion: {e}")

    return {
        "status": "deleted",
        "type": cfg_type,
        "gx_reloaded": True,
    }



@app.post("/profiling/profile/{topic}")
async def profile_pipeline(
    request: Request,
    ts_col: Optional[str] = Query(None, description="Optional timestamp column name for profiling"),
    id_cols: Optional[List[str]] = Query(None, description="Optional list of ID column names for profiling"),
    payloads: List[LoosePayload] = Body(
        ..., 
        description="Array of records to process for profiling"
    ),
):
    """
    Endpoint to run a batch of data through a profiling pipeline and return rule suggestions and possible anomalies. 
    """

    profiler_cfg = ProfilerConfig(
        id_cols=id_cols or [],
        ts_col=ts_col,       
        drop_cols=[],
        contamination=0.1,
    )
    raw_records = [p.dict() for p in payloads]
    profiler = DataProfilerUpdated.from_records(records=raw_records, config=profiler_cfg)
    profile = profiler.run()
    

    

    
    return profile
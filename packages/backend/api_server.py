# backend/api_server.py
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.state import STORE, SessionState

from ripple_core.ripple_core.load import load_movie_database, load_electrodes, load_region_data, load_all_channels
from ripple_core.ripple_core.analyze import detect_ripples, normalize_ripples, reject_ripples, find_avg

app = FastAPI(
    title="Ripple Analysis API",
    description="Backend API for Ripple Analysis GUI",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Ripple Analysis API",
        "docs": "/docs",
        "frontend": "Please access the frontend at http://localhost:3000"
    }

# ---------- Requests ----------
class LoadReq(BaseModel):
    mode: str
    session: int
    trial: int
    channel: int | None = None
    region: str | None = None
    fs: int = 1000

class DetectReq(BaseModel):
    job_id: str
    rp_low: int = 100
    rp_high: int = 140
    order: int = 550
    window_ms: float = 20.0
    z_low: float = 2.5
    z_outlier: float = 9.0
    min_dur_ms: float = 30.0
    merge_dur_ms: float = 10.0
    epoch_ms: int = 200

class NormReq(BaseModel):
    job_id: str
    fmin: int = 2
    fmax: int = 200
    win_length: float = 0.060
    step: float = 0.001
    nw: float = 1.2
    tapers: int = 2
    tfspec_pad: int = 20

class RejectReq(BaseModel):
    job_id: str
    strict_threshold: float = 3.0

# ---------- Endpoints ----------
@app.post("/load")
def load(req: LoadReq):
    job_id = uuid.uuid4().hex
    st = SessionState(fs=req.fs)

    mode = req.mode
    if mode == "Select Channel(s)":
        if req.channel is None: raise HTTPException(400, "channel required")
        st.lfp = np.asarray(load_electrodes(req.session, req.trial, req.channel))
    elif mode == "Select Region(s)":
        if req.region is None: raise HTTPException(400, "region required")
        st.lfp, _ = load_region_data(req.session, req.trial, req.region)
        st.lfp = np.asarray(st.lfp)
    elif mode == "All Channels":
        st.lfp = np.asarray(load_all_channels(req.session, req.trial))
    else:
        raise HTTPException(400, "bad mode")

    STORE[job_id] = st
    return {"ok": True, "job_id": job_id, "shape": list(st.lfp.shape)}

@app.post("/detect")
def detect(req: DetectReq):
    st = STORE.get(req.job_id)
    if st is None or st.lfp is None: raise HTTPException(400, "load first")

    st.det_res = detect_ripples(
        st.lfp,
        fs=st.fs,
        rp_band=(req.rp_low, req.rp_high),
        order=req.order,
        window_ms=req.window_ms,
        z_low=req.z_low,
        z_outlier=req.z_outlier,
        min_dur_ms=req.min_dur_ms,
        merge_dur_ms=req.merge_dur_ms,
        epoch_ms=req.epoch_ms,
    )
    return {"ok": True, "n_events": int(len(st.det_res.peak_idx))}

@app.post("/normalize")
def normalize(req: NormReq):
    st = STORE.get(req.job_id)
    if st is None or st.det_res is None: raise HTTPException(400, "detect first")

    st.norm_res = normalize_ripples(
        st.lfp,
        fs=st.fs,
        raw_windowed_lfp=st.det_res.raw_windowed_lfp,
        real_duration=st.det_res.real_duration,
        fmin=req.fmin,
        fmax=req.fmax,
        win_length=req.win_length,
        step=req.step,
        nw=req.nw,
        tapers=req.tapers,
        tfspec_pad=req.tfspec_pad,
    )
    return {"ok": True}

@app.post("/reject")
def reject(req: RejectReq):
    st = STORE.get(req.job_id)
    if st is None or st.norm_res is None: raise HTTPException(400, "normalize first")

    st.rej_res = reject_ripples(
        freq_spec_actual=st.norm_res.freq_spec_actual,
        spec_f=st.norm_res.spec_f,
        mu=st.det_res.mu,
        sd=st.det_res.sd,
        strict_threshold=req.strict_threshold,
        env_rip=st.det_res.env_rip,
        peak_idx=st.det_res.peak_idx,
    )
    return {
        "ok": True,
        "passed": int(len(st.rej_res.pass_idx)),
        "total": int(st.norm_res.freq_spec_actual.shape[0]),
    }

@app.get("/events")
def events(job_id: str):
    st = STORE.get(job_id)
    if st is None or st.rej_res is None or st.det_res is None: raise HTTPException(400, "reject first")
    N = int(len(st.det_res.peak_idx))
    items = []
    for k in range(N):
        items.append({
            "k": k,
            "accepted": bool(~st.rej_res.markers[k]),
            "peak_idx": int(st.det_res.peak_idx[k]),
            "real_start": int(st.det_res.real_duration[k,0]),
            "real_end": int(st.det_res.real_duration[k,1]),
            "reason": st.rej_res.reasons[k] if k < len(st.rej_res.reasons) else None
        })
    return {"ok": True, "items": items}

@app.get("/event/{k}")
def event(job_id: str, k: int):
    st = STORE.get(job_id)
    if st is None or st.norm_res is None: raise HTTPException(400, "normalize first")
    return {
        "ok": True,
        "k": k,
        "raw_windowed": st.det_res.raw_windowed_lfp[k].tolist(),
        "bp_windowed": st.det_res.bp_windowed_lfp[k].tolist(),
        "spec_f": st.norm_res.spec_f.tolist(),
        "spec_t": st.norm_res.spec_t.tolist(),
        "freq_spec_windowed": st.norm_res.freq_spec_windowed[k].tolist(),
        "freq_spec_actual": st.norm_res.freq_spec_actual[k].tolist(),
        "normalized_spec": st.norm_res.normalized_ripple_windowed[:,:,k].tolist(),
    }

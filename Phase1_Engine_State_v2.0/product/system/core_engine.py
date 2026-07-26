#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ModelFusion runtime v2.5 (LIVE MODE + Priority Fix)
from __future__ import annotations
import time, json, csv, os
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterator

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ------------------ Config ------------------
INPUT_DIR   = Path("../io/Input/")
OUTPUT_DIR  = Path("../io/Output/")
TEST_DIR    = Path("../io/Test/")
DEBUG_LOG = OUTPUT_DIR / "debug_live.log"
STOP_FLAG = OUTPUT_DIR / "core_stop.flag"

def debug_log(msg):
    with DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f} | {msg}\n")

TEST_X_PATH   = TEST_DIR / "engine_total_X.npy"
TEST_Y_PATH   = TEST_DIR / "engine_total_benchmark_y.npy"
TEST_EVAL_CSV = TEST_DIR / "test_eval.csv"

M0_PATH = Path("../models/Model0/model0_cnn.keras")
M1_PATH = Path("../models/Model1/model1_cnn.keras")
M2_PATH = Path("../models/Model2/model2_randomforest.pkl")
M3_PATH = Path("../models/Model3/model3_randomforest.pkl")
M4_PATH = Path("../models/Model4/model4_cnn.keras")

POLL_INTERVAL_SEC = 0.5
HISTORY_SECONDS   = 4
RPM_INDEX         = 2

# Handled by UI Bootstrap
TEST_MODE = False  

# ------------------ Utils ------------------
def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

def find_latest_file(folder: Path) -> Optional[Path]:
    cands = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".npy",".csv",".txt"}]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None

def _is_finite_numeric_array(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.number) and np.isfinite(a).all()

def _to_float_or_nan(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _csv_val(v):
    if isinstance(v, (float, int, np.floating, np.integer)):
        return float(v)
    try: return float(v)
    except Exception: return str(v)

def _extract_bad_tokens_from_line(line: str) -> list[str]:
    bad = []
    parts = [s for s in line.replace(",", " ").split() if s]
    for tok in parts:
        try: _ = float(tok)
        except Exception: bad.append(tok)
    return bad

def parse_sample_from_file(path: Path) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    try:
        if path.suffix.lower()==".npy":
            arr = np.load(path, allow_pickle=True)
            arr = np.asarray(arr).reshape(-1)
            if arr.size < 4: return None, None
            try:
                last4 = arr[-4:].astype(np.float64)
            except Exception:
                return None, {"type":"Error Value","message":"Non-numeric values in .npy payload"}
            if not _is_finite_numeric_array(last4):
                return None, {"type":"Error Value","message":"Non-finite values detected"}
            return last4, None

        last = None
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip(): last = line.rstrip("\n")
        if last is None: return None, None

        bad = _extract_bad_tokens_from_line(last)
        if bad: return None, {"type":"Error Value","message":"Non-numeric tokens found"}

        parts = [s for s in last.replace(",", " ").split() if s]
        vals = np.array([float(x) for x in parts], dtype=np.float64)
        if vals.size < 4: return None, None
        sample = vals[:4]
        if not _is_finite_numeric_array(sample):
            return None, {"type":"Error Value","message":"Non-finite values detected"}
        return sample, None
    except Exception as e:
        return None, {"type":"Error Value", "message":f"Parse failure: {e}"}

def write_output(payload: Dict[str, Any]) -> Path:
    fname = "latest_test.json" if TEST_MODE else "latest.json"
    out_path = OUTPUT_DIR / fname
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path

# ------------------ Keras custom ------------------
def deltas_fn(t):
    d = t[:, 1:, :] - t[:, :-1, :]
    zero = tf.zeros_like(d[:, :1, :])
    return tf.concat([zero, d], axis=1)

# ------------------ Frozen helper ------------------
def _pair_is_frozen(prev_vals: Optional[np.ndarray], curr_vals: Optional[np.ndarray]) -> bool:
    if prev_vals is None or curr_vals is None: return False
    epsilon = 0.00000001
    # Temp, Pressure, Vib
    for j in (0, 1, 3):
        if np.isfinite(prev_vals[j]) and np.isfinite(curr_vals[j]):
            if abs(curr_vals[j] - prev_vals[j]) <= epsilon: return True
    # RPM (if non-zero)
    j = RPM_INDEX
    if np.isfinite(prev_vals[j]) and np.isfinite(curr_vals[j]):
        if (prev_vals[j] != 0.0) and (curr_vals[j] != 0.0):
            if abs(curr_vals[j] - prev_vals[j]) <= epsilon: return True
    return False

# ------------------ Runtime ------------------
class ModelFusion:
    def __init__(self):
        ensure_dirs()
        self.history = deque(maxlen=HISTORY_SECONDS)
        
        # Load models
        self.m0 = load_model(M0_PATH, custom_objects={"deltas_fn": deltas_fn})
        self.m1 = load_model(M1_PATH, custom_objects={"deltas_fn": deltas_fn})
        
        m2_loaded = joblib.load(M2_PATH)
        self.m2 = m2_loaded.get("model") if isinstance(m2_loaded, dict) else m2_loaded
        
        m3_loaded = joblib.load(M3_PATH)
        self.m3 = m3_loaded.get("model") if isinstance(m3_loaded, dict) else m3_loaded
        
        self.m4 = load_model(M4_PATH)

        self._row_reasons_tail = deque(maxlen=3)
        self._pair_frozen_tail = deque(maxlen=3)
        self.warmup_counter = 0

    @staticmethod
    def _row_reason(vals: np.ndarray) -> str:
        t, p, r, v = vals
        def in_range(x,a,b): return (x >= a) and (x <= b)
        if in_range(t, -90, -30) or in_range(t, 165, 300): return "Uncalibrated"
        if in_range(p, -6, 0.01) or in_range(p, 1.5, 6):   return "Uncalibrated"
        if in_range(r, -2000, -0.1) or in_range(r, 12000, 100000): return "Uncalibrated"
        if in_range(v, -20, -0.1) or in_range(v, 2, 20):   return "Uncalibrated"
        return ""
    def step_with_sample(self, sample: np.ndarray) -> Dict[str, Any]:
        s_obj = np.asarray(sample, dtype=object).reshape(-1)[:4]
        s_num = np.array([_to_float_or_nan(x) for x in s_obj], dtype=np.float64)
        debug_log(
         f"STEP t={time.time():.6f} "
         f"RPM={s_num[RPM_INDEX]:.3f} "
         f"Sample={s_num.tolist()}"
        )

        if not np.isfinite(s_num).all():
            self._row_reasons_tail.append("Error Value")
            return {
                "timestamp": time.time(),
                "route": "Rule(Error Value)->Unknown",
                "final": "Unknown [Error Value]"
            }

        # Append new sample
        self.history.append(s_num)
        debug_log(f"HISTORY: {[h.tolist() for h in self.history]}")

        # Detect engine reset (RPM >0 → 0)
        if len(self.history) >= 2:
            if self.history[-2][RPM_INDEX] > 0 and s_num[RPM_INDEX] == 0:
                print("🔁 ENGINE RESET DETECTED — clearing context")
                self.history.clear()
                self._row_reasons_tail.clear()
                self._pair_frozen_tail.clear()
                self.warmup_counter = 0
                self.history.append(s_num)

        window_ready = len(self.history) >= HISTORY_SECONDS

        # Forced warmup (first 4 samples ALWAYS cold)
        if self.warmup_counter < HISTORY_SECONDS:
            self.warmup_counter += 1
            return {
                "route": "ForcedWarmup",
                "final": "Engine Off (cold)",
                "timestamp": time.time()
            }

        prev_vals = self.history[-2] if len(self.history) >= 2 else None
        pair_frozen = _pair_is_frozen(prev_vals, s_num)
        row_reason = self._row_reason(s_num)

        # Priority Checks
        if pair_frozen or any(self._pair_frozen_tail):
         res = {"route":"Rule(Frozen Sensor)->Unknown", "final":"Unknown [Frozen Sensor]"}
        elif row_reason == "Uncalibrated" or "Uncalibrated" in self._row_reasons_tail:
          res = {"route":"Row/Window(Uncalibrated)->Unknown", "final":"Unknown [Uncalibrated]"}
        elif "Error Value" in self._row_reasons_tail:
         res = {"route":"Window(Error Value)->Unknown", "final":"Unknown [Error Value]"}

        else:
            win = np.stack(list(self.history), axis=0).reshape(1, HISTORY_SECONDS, 4).astype(np.float32)
            m0_y = int(np.argmax(self.m0.predict(win, verbose=0)))

            if m0_y == 1:
                res = {"route":"M0->Unknown[Uncalibrated]","final":"Unknown [Uncalibrated]"}
            else:
                m1_y = int(np.argmax(self.m1.predict(win, verbose=0)))

                if m1_y == 0:
                    res = {"route":"M0->M1->EngineStart","final":"Engine Start"}
                elif m1_y == 1:
                    y2 = int(self.m2.predict(s_num.reshape(1,-1).astype(np.float32))[0])
                    res = {"route":"M0->M1->M2","final": {0:"Engine Off (cold)", 1:"Engine Off (cooling)"}.get(y2, "Engine Off")}
                else:
                    y3 = int(self.m3.predict(s_num.reshape(1,-1).astype(np.float32))[0])

                    two = np.stack([self.history[-2], self.history[-1]], axis=0).astype(np.float32)
                    delta = (two[1, RPM_INDEX] - two[0, RPM_INDEX])
                    x4 = np.concatenate([two, np.array([[delta],[delta]])], axis=1).reshape(1,2,5)
                    y4 = int(np.argmax(self.m4.predict(x4, verbose=0)))

                    l3 = {0:"NormalLoad", 1:"HighLoad", 2:"CriticalLoad"}.get(y3, "Load")
                    l4 = {0:"(idle)", 1:"(accelerating)", 2:"(decelerating)"}.get(y4, "Behavior")
                    res = {"route":"M0->M1->M3+M4", "final": f"{l3} {l4}"}

        self._pair_frozen_tail.append(pair_frozen)
        self._row_reasons_tail.append(row_reason)
        res["timestamp"] = time.time()
        return res

# ------------------ Test mode ------------------
def run_test():
    ensure_dirs()
    if not TEST_X_PATH.exists():
        return

    X = np.load(TEST_X_PATH, allow_pickle=True)
    y_true_data = np.load(TEST_Y_PATH, allow_pickle=True) if TEST_Y_PATH.exists() else None

    # -------------------------------
    # Flatten X properly
    # -------------------------------
    samples = []
    if X.dtype == object:
        for seq in X:
            for row in seq:
                samples.append(row)
    else:
        samples = X.reshape(-1, 4)

    # -------------------------------
    # Flatten y_true to match samples
    # -------------------------------
    y_true_flat = []
    if y_true_data is not None:
        if y_true_data.dtype == object:
            for seq in y_true_data:
                for label in seq:
                    y_true_flat.append(label)
        else:
            y_true_flat = list(y_true_data.reshape(-1))

    # Safety check (very important)
    if y_true_data is not None:
        if len(samples) != len(y_true_flat):
            print("❌ LENGTH MISMATCH")
            print("Samples:", len(samples))
            print("y_true :", len(y_true_flat))
            return

    mf = ModelFusion()

    with TEST_EVAL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Temperature","Pressure","RPM","Vibration","y_true","y_pred","route"])

        for idx, sample in enumerate(samples):
            out = mf.step_with_sample(sample)

            yt = str(y_true_flat[idx]) if y_true_data is not None else ""

            writer.writerow([
                sample[0],
                sample[1],
                sample[2],
                sample[3],
                yt,
                out.get("final"),
                out.get("route")
            ])

            if idx % 500 == 0:
                f.flush()

# ------------------ Live loop ------------------
def run_loop():
    ensure_dirs()

    # ------------------------------
    # SIGNAL READY TO SIM
    # ------------------------------
    ready_flag = OUTPUT_DIR / "core_ready.flag"

    if ready_flag.exists():
        ready_flag.unlink()

    # Load models ONCE
    mf = ModelFusion()

    with ready_flag.open("w", encoding="utf-8") as f:
        f.write("READY")
        f.flush()
        os.fsync(f.fileno())

    debug_log("CORE READY SIGNAL CREATED")

    last_sig = None

    # ------------------------------
    # LIVE STREAM LOOP
    # ------------------------------
    while True:

        # STOP FLAG CHECK (FIRST line inside loop)
        if STOP_FLAG.exists():
            debug_log("STOP FLAG DETECTED — shutting down core loop")
            break

        latest = INPUT_DIR / "current.npy"

        if not latest.exists():
            time.sleep(POLL_INTERVAL_SEC)
            continue

        mtime = latest.stat().st_mtime

        debug_log(f"FILE MTIME: {mtime}")
        debug_log(f"CORE LOOP TIME: {time.time():.6f}")

        if mtime != last_sig:

            sample, err = parse_sample_from_file(latest)

            if err or sample is None:

                out = {
                    "final": "Unknown [Error Value]",
                    "route": "Parse Error",
                    "timestamp": time.time()
                }

            else:

                try:
                    out = mf.step_with_sample(sample)

                except Exception as e:

                    print("MODEL RUNTIME ERROR:", e)

                    out = {
                        "final": "Unknown [Runtime Error]",
                        "route": "Runtime Error",
                        "timestamp": time.time()
                    }

            write_output(out)

            last_sig = mtime

        time.sleep(POLL_INTERVAL_SEC)

    debug_log("CORE LOOP EXITED CLEANLY")


if __name__ == "__main__":
    mode_env = os.environ.get("ENGINE_TEST_MODE", "0")

    debug_log(f"ENGINE_TEST_MODE ENV = {mode_env}")
    debug_log(f"TEST_MODE VAR = {TEST_MODE}")
  
    if mode_env == "1":
     debug_log("STARTING TEST MODE")
     run_test()
    else:
     debug_log("STARTING LIVE MODE")
     run_loop()


# In[ ]:





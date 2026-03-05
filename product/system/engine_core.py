#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ModelFusion runtime v2.5 (LIVE MODE + Priority Fix)
from __future__ import annotations
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ------------------ Config ------------------

M0_PATH = Path("../models/Model0/model0_cnn.keras")
M1_PATH = Path("../models/Model1/model1_cnn.keras")
M2_PATH = Path("../models/Model2/model2_randomforest.pkl")
M3_PATH = Path("../models/Model3/model3_randomforest.pkl")
M4_PATH = Path("../models/Model4/model4_cnn.keras")

HISTORY_SECONDS   = 4
RPM_INDEX         = 2  

# ------------------ Utils ------------------

def _is_finite_numeric_array(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.number) and np.isfinite(a).all()

def _to_float_or_nan(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan
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

        if not np.isfinite(s_num).all():
         row_reason = "Error Value"
        else:
         row_reason = self._row_reason(s_num)
            

        # Append new sample
        self.history.append(s_num)

        # Detect engine reset (RPM >0 → 0)
        if len(self.history) >= 2:
         if self.history[-2][RPM_INDEX] > 0 and s_num[RPM_INDEX] == 0:

          self.history.clear()
          self._row_reasons_tail.clear()
          self._pair_frozen_tail.clear()
          self.warmup_counter = 0

          # Keep current OFF sample as first element
          self.history.append(s_num)

        window_ready = len(self.history) >= HISTORY_SECONDS

        # Forced warmup (first 4 samples ALWAYS cold)
        if self.warmup_counter < HISTORY_SECONDS:
            self.warmup_counter += 1
            return {
                "route": "ForcedWarmup",
                "final": "Engine Off (cold)",
                "window": [list(row) for row in self.history],
                "timestamp": time.time()
            }

        prev_vals = self.history[-2] if len(self.history) >= 2 else None
        pair_frozen = _pair_is_frozen(prev_vals, s_num)

        # ---- Build causal 4-step window (prev 3 + current) ----
        window_row_reasons = list(self._row_reasons_tail) + [row_reason]
        window_pair_flags  = list(self._pair_frozen_tail) + [pair_frozen]

        has_error_window  = any(r == "Error Value"  for r in window_row_reasons)
        has_uncal_window  = any(r == "Uncalibrated" for r in window_row_reasons)
        has_frozen_window = any(window_pair_flags)

        # ---- Priority: Error > Frozen > Uncalibrated > Model ----
        if has_error_window:
            res = {
                "route": "Window(Error Value)->Unknown",
                "final": "Unknown [Error Value]"
            }

        elif has_frozen_window:
            res = {
                "route": "Rule(Frozen Sensor)->Unknown",
                "final": "Unknown [Frozen Sensor]"
            }

        elif has_uncal_window:
            res = {
                "route": "Row/Window(Uncalibrated)->Unknown",
                "final": "Unknown [Uncalibrated]"
            }
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
                    res = {"route":"M0->M1->M2","final": {0:"Engine Off (cold)", 1:"Engine Off (warm)"}.get(y2, "Engine Off")}
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
        res["window"] = [list(row) for row in self.history]
        res["timestamp"] = time.time()
        return res
class CoreEngine:
    def __init__(self):
        self.model = ModelFusion()
        self.latest_state = None
        self.total_processed = 0

    def process_sample(self, sample):
        result = self.model.step_with_sample(sample)
        self.latest_state = result
        self.total_processed += 1
        return result

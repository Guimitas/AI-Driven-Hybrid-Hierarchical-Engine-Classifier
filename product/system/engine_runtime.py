import os
import time
import json
import csv
from pathlib import Path

from engine_core import CoreEngine
from engine_sim import EngineSim
import traceback
import sys
LOG_PATH = Path(__file__).parent / "runtime_debug.log"

def log(msg):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
# ---------------- Mode ----------------

TEST_MODE = os.environ.get("ENGINE_TEST_MODE") == "1"


# ---------------- Paths ----------------

OUTPUT_DIR = Path("../io/Output")
TEST_DIR   = Path("../io/Test")

DATA_PATH  = TEST_DIR / "engine_total_X.npy"
LIVE_JSON  = OUTPUT_DIR / "latest.json"
TEST_CSV   = TEST_DIR / "test_eval.csv"


# ---------------- Writers ----------------

def write_live(payload):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LIVE_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------- Modes ----------------

def run_live(engine, sim):
    for sample in sim.stream():
        result = engine.process_sample(sample)
        write_live(result)
        time.sleep(0.75)  # Adjust to desired speed


def run_test():
    import numpy as np

    log("===== RUN_TEST DEBUG START =====")
    log(f"DATA_PATH: {DATA_PATH.resolve()}")
    log(f"Y_PATH: {(TEST_DIR / 'engine_total_benchmark_y.npy').resolve()}")

    X = np.load(DATA_PATH, allow_pickle=True)
    Y = np.load(TEST_DIR / "engine_total_benchmark_y.npy", allow_pickle=True)

    log(f"len(X): {len(X)}")
    log(f"len(Y): {len(Y)}")
    log(f"X dtype: {X.dtype}")
    log(f"Y dtype: {Y.dtype}")

    total_rows = 0
    for i in range(len(X)):
        try:
            total_rows += len(X[i])
        except Exception:
            log(f"Sequence without len at index: {i}")

    log(f"Total rows in X: {total_rows}")

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    engine = CoreEngine()

    processed_rows = 0
    sequence_index = 0

    with TEST_CSV.open("w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Temperature",
            "Pressure",
            "RPM",
            "Vibration",
            "Predicted Label",
            "True Label",
            "Route",
        ])

        for seq, label_seq in zip(X, Y):

            log(f"Entering sequence {sequence_index}, length={len(seq)}")

            for sample, true_label in zip(seq, label_seq):

                processed_rows += 1
                result = engine.process_sample(sample)

                temp, pressure, rpm, vib = sample

                writer.writerow([
                    temp,
                    pressure,
                    rpm,
                    vib,
                    result.get("final"),
                    true_label,
                    result.get("route"),
                ])

            sequence_index += 1

    log(f"Processed rows: {processed_rows}")
    log("===== RUN_TEST DEBUG END =====")
# ---------------- Entry ----------------
def main():
    log("ENGINE RUNTIME STARTED")

    if TEST_MODE:
        log("RUNNING TEST MODE")
        run_test()
    else:
        log("RUNNING LIVE MODE")
        engine = CoreEngine()
        sim = EngineSim(DATA_PATH)
        run_live(engine, sim)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("CRASH DETECTED")
        log(traceback.format_exc())
        raise
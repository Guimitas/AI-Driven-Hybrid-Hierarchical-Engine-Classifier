# AI-Driven Hybrid-Hierarchical Engine Classifier

A hierarchical machine learning system that classifies **15 engine states** — Engine Off (cold / warm), Engine Start, three load levels (Normal / High / Critical) each with accelerating, idle, and decelerating behavior, plus three sensor-fault states (Uncalibrated, Frozen Sensor, Error Value) — from four time-series inputs: Temperature, Pressure, RPM, and Vibration. Built as a hierarchy combining rule-based detectors, Random Forest, and CNN models, reaching 99.8% accuracy on synthetic data.

**Why this project exists:** I'm a mechanical engineering student with no formal AI education — my background came from online courses (IBM's "Python for Data Science, AI & Development" and DeepLearning.AI's "AI For Everyone"). This was a self-directed learning project: build a complete engine-state classification system from scratch — defining the states, generating and processing the data, designing the architecture, training the models, and testing the result — to find out first-hand what standard ML models can and can't do. Over two months of building, breaking, and rebuilding, the biggest lesson was how limited these models are without guidance — patterns I could spot by eye were invisible to a single generic model. Reaching 99.8% accuracy required restructuring the whole system around the physics of the problem.

---

## 🎬 Quick Overview

*Short on time? This short video explains, in simple terms, how the final system works and how it reached 99.8% accuracy:*

[▶ **Watch the overview video**](https://youtu.be/_TW-erKZu38) — 🔊 Audio narration · 💬 English subtitles available

---

## 🔴 Demo 1 — LIVE Mode (Real-Time Classification)

The system receives sensor data in real time and classifies the engine state as it changes. The UI shows the current state, the decision path through the model hierarchy (e.g., M0 → M1 → M2), and the live sensor context window.

[▶ **Watch the LIVE demo**](https://youtu.be/8lmq0rs_uAA) — 🔇 No Audio


<img width="1036" alt="LIVE mode" src="https://github.com/user-attachments/assets/3a24e6fb-226c-4250-ba00-ff17eed83e13" />



---

## 🧪 Demo 2 — TEST Mode (Full Dataset Evaluation)

The system runs through the complete labeled dataset — 164,920 rows — and compares its predictions against the ground truth, reaching 99.8% accuracy.

[▶ **Watch the TEST demo**](https://youtu.be/ZeJaeyeq9EE) — 🔇 No Audio

<img width="1036" alt="TEST mode — prediction table" src="https://github.com/user-attachments/assets/fbf46dc2-f5da-4319-83d0-7883a4c6adf4" />


> **Why not 100%?** Nearly all misclassifications occur at the boundary between adjacent states (e.g., Normal vs High Load, or idle vs decelerating, where the defining values differ by less than one unit). Windows sitting exactly on a class boundary are inherently ambiguous — this is a property of how the states are defined, not a model failure. Notably, the system makes no serious errors: it never confuses NormalLoad with CriticalLoad, and never raises a false fault alert.

---

## 🧠 System Architecture

<img width="1040" alt="System architecture" src="https://github.com/user-attachments/assets/f61fd043-e016-48b0-8932-2d7b1ba3b890" />

Instead of one generic model handling every state, the system splits the problem into a hierarchy where each stage handles the task it's suited for:

- **Rule-based detectors (hard-coded):** frozen sensor and error-value detection — deterministic faults don't need ML.
- **M0:** uncalibrated detection
- **M1:** engine start detection
- **M2:** Engine Off — cold vs warm
- **M3 + M4:** load level (Normal / High / Critical) + behavior (idle / accelerating / decelerating)

Models are a mix of rule-based logic, Random Forest, and CNN — each chosen for the pattern type it handles best.

---

## 📄 Full Technical Report

The complete report — state definitions, synthetic data generation, architecture decisions, errors made along the way, and lessons learned — is available here:

[**📄 Download the Full Technical Report (PDF)**](TECHNICAL_REPORT.pdf)

---

## ⚠️ Limitations

Trained and evaluated on synthetic Python-generated data (with added noise). Real-sensor validation is the natural next step.

---

## 📁 Repository Structure

```
├── Phase1_Engine_State_v2.0/
│   ├── data/          # Empty folders — populated by running the generators
│   ├── models/        # The five trained models, ready to run without retraining
│   ├── notebooks/     # Data generation, model training, and evaluation
│   └── product/       # Final runtime system + control-panel UI
├── README.md
├── TECHNICAL_REPORT.pdf
└── requirements.txt
```

## 🛠️ Run It Yourself

**You will need:** Python 3.10+ · Jupyter · the packages below:

    pip install -r requirements.txt

*(tensorflow · scikit-learn · numpy · pandas · ipywidgets · joblib · matplotlib)*

All data is synthetic and regenerated locally — fixed seeds, identical results, nothing to download.

---

### 1 · Generate the test dataset

📂 `notebooks/DataGeneration/ModelFusion.SensorSimulation/` — run the 9 notebooks **in this order** (later ones build on top of earlier data):

| # | Notebook | What it does |
|---|----------|--------------|
| 1–5 | `EngineOff.SIM` → `EngineNormalLoad.SIM` → `EngineHighLoad.SIM` → `EngineCriticalLoad.SIM` → `EngineStart.SIM` | Generates each engine state |
| 6 | `EngineOCC.Train.SIM` | Creates the uncalibrated fault data |
| 7 | `EngineOCC.SIM` | Adds frozen-sensor + error-value faults on top |
| 8 | `SensorSimulator.DataCore.SIM` | Arranges everything into a realistic engine sequence (off → start → load …) |
| 9 | `SensorSimulator.Benchmark.SIM` | Labels every row → final test set (`engine_total_X.npy` + `Y`) |

> ✅ Each notebook prints **DONE** when finished. To verify, check `data/simulation/` — you'll find the `X`, the `Y`, and a `.csv` you can open to inspect the data manually.

---

### 2 · Run the system

Open **`product/ui/ControlPanel.ipynb`** and run it. A panel appears with two toggles:

| Mode | What it does |
|------|--------------|
| 🧪 **TEST** | Runs the full 164,920-row benchmark against the ground-truth labels |
| 🔴 **LIVE** | Streams simulated sensor data and classifies it in real time |

Takes ~10 s to start. No required order — toggle **ON / OFF / TEST / LIVE** freely, it's fool-proof.

---

### 3 · (Optional) Retrain the models

Pre-trained models are included — skip this unless you want to reproduce training.

**a. Regenerate the training data** — in `notebooks/DataGeneration/`, run the model folders in order:

`Model02.EngineOff` → `Model03.NormalLoad` → `Model03.HighLoad` → `Model03.CriticalLoad` → `Model04.Behaviour` → `Model01.EngineStart` → `Model00.RouterOCC` → `Model00,01.Router` → `Model00,01.Router.V2.0`

**b. Train** — in `notebooks/ModelTraining/`, run folders `Model0` → `Model4`. Some finish in seconds; the CNNs can take 20+ minutes.

> ⚠️ CNN training is stochastic — retraining may land slightly above or below the reported accuracies. A stronger machine makes this faster.
---

## 🔎 A note on AI use

As you'd expect from an AI-focused project, this work made heavy use of AI tools —
and being clear about which is part of the point, since knowing where these tools
help and where they don't is the subject of the project itself.

- **The video:** ElevenLabs for the voiceover; Claude for the intro/outro and step
  images, and for structuring the report. *(The technical diagrams are my own.)*
- **The code:** Gemini and ChatGPT were used heavily for the Python implementation.

**What's mine is everything else** — and it's the part that matters here. The
architecture, the state definitions, the hierarchy design, the diagnosis of why
things failed, and the decision to rebuild the entire system from scratch after
the first version collapsed. That was two months of trial and error and the real
substance of the project, documented in full in the technical report.

The goal was never to demonstrate coding ability — I've taken Python courses at
university and high school, plus IBM's "Python for Data Science" on Coursera (in my
portfolio). The goal was to understand **the other side of the coin**: what machine
learning can and cannot do, seen from a mechanical engineer's perspective.

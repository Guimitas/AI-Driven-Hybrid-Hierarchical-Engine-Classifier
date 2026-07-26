# AI-Driven Hybrid-Hierarchical Engine Classifier

A hierarchical machine learning system that classifies **15 engine states** — Engine Off (cold / warm), Engine Start, three load levels (Normal / High / Critical) each with accelerating, idle, and decelerating behavior, plus three sensor-fault states (Uncalibrated, Frozen Sensor, Error Value) — from four time-series inputs: Temperature, Pressure, RPM, and Vibration. Built as a hierarchy combining rule-based detectors, Random Forest, and CNN models, reaching 99.8% accuracy on synthetic data.

**Why this project exists:** I'm a mechanical engineering student with no formal AI education — my background came from online courses (IBM's "Python for Data Science, AI & Development" and DeepLearning.AI's "AI For Everyone"). This was a self-directed learning project: build a complete engine-state classification system from scratch — defining the states, generating and processing the data, designing the architecture, training the models, and testing the result — to find out first-hand what standard ML models can and can't do. Over two months of building, breaking, and rebuilding, the biggest lesson was how limited these models are without guidance — patterns I could spot by eye were invisible to a single generic model. Reaching 99.8% accuracy required restructuring the whole system around the physics of the problem.

---

## 🎬 Quick Overview

*Short on time? This short video explains, in simple terms, how the final system works and how it reached 99.8% accuracy:*

[▶ **Watch the overview video**](https://youtu.be/_TW-erKZu38)

---

## 🔴 Demo 1 — LIVE Mode (Real-Time Classification)

The system receives sensor data in real time and classifies the engine state as it changes. The UI shows the current state, the decision path through the model hierarchy (e.g., M0 → M1 → M2), and the live sensor context window.

[▶ **Watch the LIVE demo**](https://youtu.be/8lmq0rs_uAA)


<img width="1036" alt="LIVE mode" src="https://github.com/user-attachments/assets/3a24e6fb-226c-4250-ba00-ff17eed83e13" />



---

## 🧪 Demo 2 — TEST Mode (Full Dataset Evaluation)

The system runs through the complete labeled dataset — 164,920 rows — and compares its predictions against the ground truth, reaching 99.8% accuracy.

[▶ **Watch the TEST demo**](https://youtu.be/ZeJaeyeq9EE)

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

├── Phase1_Engine_State_v2.0/
│   ├── notebooks/     # Data generation, model training, and evaluation
│   ├── models/        # The five trained models — ready to run without retraining
│   ├── product/       # Final runtime system + control-panel UI
│   └── data/          # Empty folders; populated by running the generators
├── TECHNICAL_REPORT.pdf
├── requirements.txt
└── README.md

---

## 🚀 Running the Project

The dataset is synthetic and regenerable — you generate it locally rather than
downloading it. Fixed random seeds mean you get identical data every time, so
nothing large needs to live in the repo.

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Generate the data:** run the notebooks in `notebooks/` (data-generation
   section) to populate the empty `data/` folders.
3. **Train the models** *(optional — trained models are already in `models/`)*:
   run the training notebooks in order, M0 → M1 → M2 → M3 → M4.
4. **Run the system:** open the control panel in `product/` and toggle System ON.
   - **LIVE** mode streams simulated sensor data in real time.
   - **TEST** mode runs the full 164,920-row benchmark.

*Requirements: Python 3.10+, with the packages in `requirements.txt`.*

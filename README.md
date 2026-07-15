AI-Driven Hybrid-Hierarchical Engine Classifier

A hierarchical machine learning system that classifies engine operating states (off/cold, cooling, start, load levels, acceleration states) and sensor faults from four time-series inputs — Temperature, Pressure, RPM, and Vibration — combining rule-based detectors, Random Forest, and CNN models, reaching 99%+ precision on synthetic data.

Why this project exists: I'm a mechanical engineering student with no formal AI education — my background came from online courses (IBM's "Python for Data Science, AI & Development" and DeepLearning.AI's "AI For Everyone"). This was a self-directed learning project: build a complete engine-state classification system from scratch and find out first-hand what standard ML models can and can't do. Over two months of building, breaking, and rebuilding, the biggest lesson was how limited these models are without guidance — patterns I could spot by eye were invisible to a single generic model. Reaching ~99% precision required restructuring the whole system around the physics of the problem.


🎬 3-Minute Overview

Short on time? This 3-minute video explains how the final system works and how it reached 99% precision:

▶ Watch the overview video


🔴 Demo 1 — LIVE Mode (Real-Time Classification)

The system receives sensor data in real time and classifies the engine state as it changes. The UI shows the current state, the decision path through the model hierarchy (e.g., M0 → M1 → M2), and the live sensor context window.

▶ Watch the LIVE demo

Mostrar Imagem


🧪 Demo 2 — TEST Mode (Full Dataset Evaluation)

The system runs through the complete labeled dataset and compares its predictions against the ground truth, reaching ~99% precision.

▶ Watch the TEST demo

Mostrar Imagem


Why not 100%? Nearly all misclassifications occur at the boundary between adjacent states (e.g., Normal vs High Load, where the defining values differ by less than 1 unit). Windows sitting exactly on a class boundary are inherently ambiguous — this is a property of how the states are defined, not a model failure. Away from boundaries, classification is essentially perfect.




🧠 System Architecture

Mostrar Imagem

Instead of one generic model handling every state, the system splits the problem into a hierarchy where each stage handles the task it's suited for:


Rule-based detectors (hard-coded): frozen sensor and error-value detection — deterministic faults don't need ML.
M0: uncalibrated detection
M1: engine start detection
M2: Engine Off — cold vs cooling
M3 + M4: load level (Normal / High / Critical) + behavior (decelerating / steady / accelerating)


Models are a mix of rule-based logic, Random Forest, and CNN — each chosen for the pattern type it handles best.


📄 Full Technical Report

The complete report — state definitions, synthetic data generation, architecture decisions, errors made along the way, and lessons learned — is available here:

📄 Download the Full Technical Report (PDF)


⚠️ Limitations

Trained and evaluated on synthetic Python-generated data (with added noise). Real-sensor validation is the natural next step.


📁 Repository Structure

├── data/          # Synthetic data generation scripts
├── models/        # Trained models (rule-based, Random Forest, CNN)
├── ui/            # Real-time classification interface
├── docs/          # Full technical report (PDF)
└── README.md

(adjust to match your actual folder names)

# Simulation data

Realistic engine sequences — states in plausible order, with sensor faults
injected — generated independently of the training data.

Combined, these become the end-to-end benchmark in product/io/Test/, so the
system is never measured on data it was trained on.

Rebuild with notebooks/DataGeneration/ModelFusion.SensorSimulation/.

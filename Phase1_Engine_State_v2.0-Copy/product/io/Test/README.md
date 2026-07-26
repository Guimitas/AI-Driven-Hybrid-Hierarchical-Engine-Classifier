# Runtime test data

What the running system reads in TEST and LIVE mode.

| File | Role |
|---|---|
| engine_total_X.npy | sensor stream fed to the pipeline |
| engine_total_benchmark_y.npy | ground-truth labels |
| engine_total_benchmark.csv | readable mirror of both |
| test_eval.csv | written by the system after a TEST run |

Built from data/simulation/, never from training data — this is what makes
the end-to-end accuracy an honest measure of the whole pipeline rather than
of any single model.

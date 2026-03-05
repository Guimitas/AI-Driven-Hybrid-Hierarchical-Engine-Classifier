from pathlib import Path
import numpy as np


class EngineSim:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.samples = self._load_and_flatten()

    def _load_and_flatten(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        X = np.load(self.dataset_path, allow_pickle=True)

        samples = []

        if X.dtype == object:
            for seq in X:
                seq = np.asarray(seq)
                if seq.ndim == 2:
                    for row in seq:
                        samples.append(row[:4])
        elif X.ndim == 3:
            for i in range(X.shape[0]):
                for t in range(X.shape[1]):
                    samples.append(X[i, t, :4])
        elif X.ndim == 2:
            for row in X:
                samples.append(row[:4])
        else:
            raise ValueError("Unsupported dataset shape")

        return samples

    def stream(self):
        for sample in self.samples:
            yield sample
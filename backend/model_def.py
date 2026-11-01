"""Architecture of the pickled FDM fault classifier.

``models/model.pkl`` is a plain ``pickle.dump`` of a whole ``nn.Module`` whose
class lived in ``__main__`` at training time, so unpickling needs a class of
that name to be importable from ``__main__``. Defining it here and aliasing it
onto ``__main__`` (see ``register_for_unpickling``) lets the checkpoint load
without touching the file.
"""

import sys

import torch.nn as nn

# Derived from the checkpoint: 4 conv blocks each halving the spatial size, so
# the 16384-unit flatten (256 * 8 * 8) implies a 128x128 input.
INPUT_SIZE = 128
NUM_CLASSES = 5


class Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def register_for_unpickling() -> None:
    """Make ``__main__.Model`` resolve to this class."""
    main = sys.modules.get("__main__")
    if main is not None and not hasattr(main, "Model"):
        main.Model = Model

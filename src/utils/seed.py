"""Global seeding utility. Call set_seed() once at the start of every entry point
so that dataset shuffling, model init, and dropout are reproducible (Section 24)."""
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic-ish; full determinism costs speed, so we don't force it.
        torch.backends.cudnn.benchmark = True
    except ImportError:
        # torch may not be installed yet on a pure data-prep environment.
        pass

import random


def set_seed(seed: int = 42) -> None:
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


import numpy as np
try:
    from pygfm.public.utils import early_stopping as pygfm_early_stopping
    _HAS_PYGFM_EARLY_STOP = True
except ImportError:
    _HAS_PYGFM_EARLY_STOP = False


class EarlyStopping:
    """
    Early stopping wrapper. Uses pygfm's ``early_stopping`` helper internally
    when available, otherwise falls back to own implementation.

    Tracks the best validation score and stops when no improvement is seen
    for ``patience`` consecutive checks.
    """
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_val = -np.inf
        self.best_dict = None
        self.early_stop = False

    def __call__(self, result):
        if result['val'] > self.best_val:
            self.best_val = result['val']
            self.best_dict = result
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

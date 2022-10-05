import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_norm_cls(norm_type):
    if norm_type == 'minmax':
        return MinMax
    if norm_type == 'standardization':
        return Standardization
    
    raise ValueError('norm type should be "minmax" or "standardization"')

class NormBase():
    def __init__(self) -> None:
        self.norm_info = None

    def _get_axis(self, mask_axis, dim):
        if mask_axis is None:
            return tuple(range(dim))
        if isinstance(mask_axis, int):
            mask_axis = (mask_axis, )
        axis = [i for i in range(dim) if not (i in mask_axis)]
        return tuple(axis)

    def fit(self, x):
        x = np.array(x)
        return x

    def transform(self,x):
        x = np.array(x)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class MinMax(NormBase):
    def __init__(self, mask_axis=None) -> None:
        # axis: axis the is mask off
        super().__init__()
        self.mask_axis = mask_axis

    def fit(self, x: np.ndarray):
        x = np.array(x)
        axis = self._get_axis(self.mask_axis, np.ndim(x))
        self.min = np.min(x, axis=axis, keepdims=True)
        self.max = np.max(x, axis=axis, keepdims=True)
    
    def transform(self, x: np.ndarray):
        x = np.array(x)
        x_norm = x - self.min
        np.divide(x-self.min, self.max-self.min, x_norm, where=self.max-self.min!=0)
        return x_norm


class Standardization(NormBase):
    def __init__(self, mask_axis=None) -> None:
        super().__init__()
        self.mask_axis = mask_axis
    
    def fit(self, x):
        x = np.array(x)
        axis = self._get_axis(self.mask_axis, np.ndim(x))
        self.mean = np.mean(x, axis=axis, keepdims=True)
        self.std = np.std(x, axis=axis, keepdims=True)

    def transform(self, x):
        x = np.array(x)
        x_norm = x - self.mean
        np.divide(x-self.mean, self.std, x_norm, where=self.std!=0)
        return x_norm
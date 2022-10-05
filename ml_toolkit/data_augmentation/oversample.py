from imblearn.over_sampling import SMOTE, RandomOverSampler
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ['DataAug_SMOTE', 'DataAug_RANDOM']

class Base():
    def __init__(self, random_state) -> None:
        if random_state == None:
            random_state = np.random.RandomState()
        self.random_state = random_state

    def get_sampler(self):
        return None

    def resample(self, x, y):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        self.sampler = self.get_sampler()
        x_res, y_res = self.sampler.fit_resample(x, y)
        x_res = x_res.reshape(-1,*shape[1:])
        return x_res, y_res

class DataAug_SMOTE(Base):
    def __init__(self, random_state=None) -> None:
        super().__init__(random_state)

    def get_sampler(self):
        return SMOTE(random_state=self.random_state)

class DataAug_RANDOM(Base):
    def __init__(self, random_state) -> None:
        super().__init__(random_state)

    def get_sampler(self):
        return RandomOverSampler(random_state=self.random_state)
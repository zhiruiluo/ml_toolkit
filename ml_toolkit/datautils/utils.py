import numpy as np
from typing import Union, List
import logging
logger = logging.getLogger(__name__)

def find_class(labels: Union[np.ndarray, List]):
    classes = sorted(np.unique(labels))
    class_to_index = {classname: i for i,
                            classname in enumerate(classes)}
    logger.info(f'class_to_index {class_to_index}')
    nclass = len(classes)
    index = np.vectorize(class_to_index.__getitem__)(labels)
    if len(index.shape) == 2:
        index = index.reshape(-1)
    logger.info(f'Label counts: {list(enumerate(np.bincount(index)))}')
    return index, nclass, class_to_index
import torch 
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

def get_num_gpus():
    num_gpus = torch.cuda.device_count()
    logger.info(f'[num_gpus] {num_gpus}')
    return num_gpus

def cpu_count():
    if os.environ.get('SLURM_CPUS_ON_NODE'):
        cpus_reserved = int(os.environ['SLURM_CPUS_ON_NODE'])
    else:
        cpus_reserved = 8
    return cpus_reserved
    # return multiprocessing.cpu_count()

def cuda_available():
    if get_num_gpus() == 0:
        return False
    return True

def set_cpu_affinity(num_cpus):
    logger.info('[cpu affinity] %s',os.sched_getaffinity(0))
    cpu_count = os.cpu_count()
    choices = np.random.choice(list(range(1,cpu_count)),replace=False, size=num_cpus).tolist()
    if choices == []:
        logger.info('[cpu affinity] use default setting %s', os.sched_getaffinity(0))
        return
    os.sched_setaffinity(0, choices)
    logger.info('[cpu affinity] set new affinity %s',os.sched_getaffinity(0))
import random
import time

def get_seed(determ=False):
    if determ:
        return 34
    random.seed(time.time())
    seed = random.randint(0,100000)
    return seed


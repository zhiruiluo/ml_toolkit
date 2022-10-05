import logging
import unittest
from datetime import datetime

from ml_toolkit.database.database_control1 import DatabaseManager
from ml_toolkit.database.database_model1 import *

logger = logging.getLogger(__name__)

class Test(unittest.TestCase):
    def test_case(self):
        assert True==True


def test2():
    dbm = DatabaseManager('test_new.db','./db')
    model = dict(name='CNN_BiLSTM', params=dict(bn=True, nclass=2, text='abc'))
    dataset = dict(name='ECO', params=dict(case='1', extraction=True))
    exper = dict(exp=1,splits='3:1:1',nrepeat='10',nfold='1',epochs=10)

    for i in range(1,4):
        perfs = {
            'train':dict(phase='train', accuracy=0.9, f1macro=0.9, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y')),
            'val':dict(phase='val', accuracy=0.8, f1macro=0.9, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y')),
            'test':dict(phase='test', accuracy=0.7, f1macro=0.9, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y'))
        }
        result = dict(fold=1, repeat=i)
        dbm.save_results(dataset,model,exper,result, perfs)
    for i in range(1,4):
        perfs = {
            'train':dict(phase='train', accuracy=0.3, f1macro=0.1, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y')),
            'val':dict(phase='val', accuracy=0.4, f1macro=0.1, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y')),
            'test':dict(phase='test', accuracy=0.1, f1macro=0.1, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y'))
        }
        result = dict(fold=2, repeat=i)
        dbm.save_results(dataset,model,exper,result, perfs)

    
    logger.info(dbm.get_perfs(dataset,model,exper,result))
    dbm.show_all(Results)
    dbm.delete_results_by_keys(dataset,model,exper,result)
    logger.info('deleted')
    dbm.show_all(Results)

def test_add_gridtable():
    db = DatabaseManager('exp_eco_grid.db','./db')
    
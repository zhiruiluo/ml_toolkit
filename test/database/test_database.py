import logging
import unittest
from datetime import datetime

from ml_toolkit.database.database_control import DatabaseManager
from ml_toolkit.database.database_model import *

logger = logging.getLogger(__name__)

class Test(unittest.TestCase):
    def test_case(self):
        assert True==True


def test2():
    dbm = DatabaseManager('test_new.db')
    model = dict(name='CNN_BiLSTM', params=dict(bn=True, nclass=2, text='abc'))
    dataset = dict(name='ECO', params=dict(case='1', extraction=True))
    exper = dict(exp=1,splits='3:1:1',nrepeat='10',nfold='1')

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

    dbm.show_all(Performances)
    dbm.show_all(Results)
    dbm.show_all(Experiments)
    dbm.get_average_repeat()
    dbm.show_all(AverageRepeat)

    dbm.delete_results(1, {'fold':1, 'repeat':2})
    dbm.show_all(Performances)
    dbm.show_all(Results)
    dbm.show_all(Experiments)
    

def test_grid_search():
    dbm = DatabaseManager('test_grid_search.db')
    exper = dict(exp=1,splits='3:1:1',nrepeat='1',nfold='1')
    dataset = dict(name='ECO', params=dict(case='1'))

    for k1,v1 in {'n_estimators': [100, 200]}.items():
        for k2, v2 in {'criterion': ['gini', 'entropy']}.items():
            model = dict(name='RF', params={k1:v1,k2:v2})
            perfs = {
            'train':dict(phase='train', accuracy=0.9, f1macro=0.9, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y')),
            'val':dict(phase='val', accuracy=0.8, f1macro=0.9, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y')),
            'test':dict(phase='test', accuracy=0.7, f1macro=0.9, 
            timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y'))
            }
            result = dict(fold=1, repeat=1)
            dbm.save_results(dataset,model,exper,result, perfs)  

    
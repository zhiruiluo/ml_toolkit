import logging
from ml_toolkit.gridsearch.gridsearch import GridSearch
from ml_toolkit.utils.param_production import from_options
from ml_toolkit.database.database_control import DatabaseManager
from tqdm import tqdm
from pyinstrument import Profiler
import os

logger = logging.getLogger(__name__)

test_db = 'test_gridsearch'

def python_options_2():
    eco_options = dict(
        dataset_setting=[1],
        study_case=['case1','case2','case3','case4'],
        norm_type='minmax',
        imb_sam=1,
        t_en=1,
        w_en=1,
        hol_en=0,
        fix_en=0,
        prob_en=0,
        plugs_en=0,
        plugs_cum=0,
        sm_cum=0
    )
    datasets = {'dataset':[{'ECO': eco_options}]}
    rf = {
        'RF': {
            'n_estimators': [100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 100, 200],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['auto, sqrt', 'log2']
        }
    }
    adaboost = {
        'AdaBoost': {
            'base_estimator': [None],
            'n_estimators': [50],
            'learning_rate': [1.0],
            'algorithm': ['SAMME', 'SAMME.R'],
            'random_state': [None]
        }
    }
    svm = {
        'SVM': {
            'C': [1.0, 0.5],
            'kernel': ['linear', {'poly': {'degree': [3]}}, {'rbf': {'gamma': ['scale', 'auto']}}, 'sigmoid'],
        }
    }
    knn = {
        'kNN': {
            'n_neighbors': [5],
            'radius': [1.0], 
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [30],
            'p': [2,1],
        }
    }
    models = {'model':[rf, adaboost, svm, knn]}
    trainer_options = dict(
        epochs=50,
        patience=50,
        lr=1e-3,
        weight_decay=5e-4,
        determ='',
        batch_size=32
    )
    results = dict(
        fold=1,
        repeat=[1,2,3,4,5]
    )

    experiment_options = dict(
        exp=3,
        splits='3:1:1',
        nfold=1,
        nrepeat=2,
        epochs=10,
    )

    system_options = dict(expname='exp_eco')

    params = {
        'trainer':trainer_options,
        'system': system_options,
        'model': models,
        'dataset': datasets,
        'exp': experiment_options,
        'result': results,
    }
    return params

def sequential_simulate_perfs():
    db = DatabaseManager(f'{test_db}.db','./db')
    options = python_options_2()
    total = 0
    for p in from_options(options):
        d_dict = p['dataset'].copy()
        d_name = d_dict['dataset']
        d_dict.pop('dataset')
        dataset = dict(name=d_name, params=d_dict)

        m_dict = p['model'].copy()
        m_name = m_dict['model']
        m_dict.pop('model')
        model = dict(name=m_name, params=m_dict)
        exp = p['exp']
        result = p['result']

        perfs = {
            'train': dict(phase='train',accuracy = 0.9),
            'val': dict(phase='val', accuracy=0.8),
            'test': dict(phase='test',accuracy=0.7)
        }
        db.save_results(dataset, model, exp, result, perfs)
        total += 0
        # logger.debug(f'[simulate] {total}')


def insert(p):
    i, p = p
    db = DatabaseManager(f'{test_db}_mulp.db','./db')
    logger.info(i)
    d_dict = p['dataset'].copy()
    d_name = d_dict['dataset']
    d_dict.pop('dataset')
    dataset = dict(name=d_name, params=d_dict)

    m_dict = p['model'].copy()
    m_name = m_dict['model']
    m_dict.pop('model')
    model = dict(name=m_name, params=m_dict)
    exp = p['exp']
    result = p['result']

    perfs = {
        'train': dict(phase='train',accuracy = 0.9),
        'val': dict(phase='val', accuracy=0.8),
        'test': dict(phase='test',accuracy=0.7)
    }
    db.save_results(dataset, model, exp, result, perfs)

def simulate_multiproces_insert():
    if os.path.isfile(f'./db/{test_db}_mulp.db'):
        os.remove(f'./db/{test_db}_mulp.db')
    options = python_options_2()
    total = 0
    from multiprocessing import Pool
    db = DatabaseManager(f'{test_db}_mulp.db','./db')
    timeout=15
    with Pool(40) as p:
        p.map(insert, [p for p in enumerate(from_options(options))])

    logger.debug(f'[simulate] {total}')

# def test_gridsearch_read_yaml():
#     gs = GridSearch(test_db, python_options_2())
#     gs.parameters_to_yaml('param1.yaml')
#     gs.parameters_from_yaml('param1.yaml')

def test_multiprocess_insert():
    p = Profiler()
    with p:
        simulate_multiproces_insert()
    logger.info(f'\n{p.output_text()}')


# def test_grid_search():
#     if os.path.isfile(f'./db/{test_db}.db'):
#         os.remove(f'./db/{test_db}.db')
#     options = python_options_2()
#     p = Profiler()
#     with p:
#         sequential_simulate_perfs()
#     logger.info(f'\n{p.output_text()}')
#     logger.info(options)
#     gs = GridSearch(test_db, options)
#     p = Profiler()
#     with p:
#         gs.check_status()
#     logger.info(f'\n{p.output_text()}')

# def test_grid_get_all_keys():
#     options = python_options_2()
#     gs = GridSearch(test_db, options)
#     gs.save_results_csv('./results/', 'grid_search1.csv')

# def test_grid_evaluate():
#     options = python_options_2()
#     gs = GridSearch(test_db, options)
#     results = gs.get_evaluates_Repeat()
#     logger.info(results)
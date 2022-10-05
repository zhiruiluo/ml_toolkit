import sys
from pathlib import Path

import sqlalchemy

sys.path.append('.')
import logging
import os

import pandas as pd
from sqlalchemy import (and_,or_,create_engine,delete,update,select,bindparam)

from sqlalchemy.orm import Session, sessionmaker, aliased
from ml_toolkit.database.database_model import Results, Base

logger = logging.getLogger(__name__)

def lockretry(func):
    def wrapper(*args, **kwargs):
        retry_times = 0
        while True:
            if retry_times != 0:
                logger.info(f'[LockRetry] retry {func.__name__} {retry_times} times!')
            try:
                ret = func(*args, **kwargs)
            except sqlalchemy.exc.OperationalError as e:
                logger.info(f'[LockRetry] {e} retry {func.__name__}!')
                retry_times += 1
            else:
                break
        return ret
    return wrapper

class DatabaseManager():
    def __init__(self, db_name, db_dir=__file__):
        self.db_name = db_name
        self.db_dir = Path(os.path.realpath(os.path.dirname(db_dir)))
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir.joinpath(db_name)
        self.timeout = 15
        logger.info(f'[DatabaseManager] {self.db_path}')
        self._create_tables()

    @property
    def engine(self):
        if not hasattr(self,'_engine'):
            self._engine = create_engine(f'sqlite+pysqlite:///{self.db_path}', echo=False, future=True,connect_args={'timeout': self.timeout})
        return self._engine

    @property
    def session(self) -> Session:
        if not hasattr(self,'_session'):
            Session = sessionmaker(bind=self.engine, autoflush=False)
            self._session = Session
        return self._session

    def _create_tables(self):
        if not os.path.isfile(self.db_path):
            Base.metadata.create_all(self.engine)
    
    @lockretry
    def insert(self, x):
        with self.session.begin() as session:
            session.add(x)

    @lockretry
    def insert_rows(self, model, rows):
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(
                    model.__table__.insert(),
                    rows
                )

    @lockretry
    def execute_rows(self, statm, rows):
        with self.engine.begin() as conn:
            conn.execute(statm,rows)

    @lockretry
    def get_first(self, statm):
        with self.session() as session:
            obj = session.execute(statm).first()
        return obj

    @lockretry
    def get_all(self, statm):
        with self.session() as session:
            objs = session.execute(statm).all()
        return objs

    def get_or_create(self, model, statm):
        obj = self.get_first(statm)
        if obj is None:
            self.insert(model)
            obj = self.get_first(statm)
        return obj[0]

    def get_by_statm(self, statm):
        obj = self.get_first(statm)
        if obj is None:
            return None
        return obj[0]

    def get_statement_exact_match(self, model: Results):
        clauses = []
        conditions = [k for k in vars(model) if k != '_sa_instance_state']
        for k in conditions:
            clause = getattr(model.declarativeClass(),k) == getattr(model,k)
            clauses.append(clause)

        statm = select(model.declarativeClass()).where(and_(*clauses))
        return statm

    def delete_statement_exact_match(self, model: Results):
        clauses = []
        conditions = [k for k in vars(model) if k != '_sa_instance_state']
        for k in conditions:
            clause = getattr(model.declarativeClass(),k) == getattr(model,k)
            clauses.append(clause)

        statm = delete(model.declarativeClass()).where(and_(*clauses))
        return statm
        
    def get_statement_match_conditions(self, model: Results, conditions: list):
        clauses = []
        for k in conditions:
            clause = getattr(model.declarativeClass(),k) == getattr(model,k)
            clauses.append(clause)

        statm = select(model.declarativeClass()).where(and_(*clauses))
        return statm

    def param_from_dict(self, params_dict):
        if params_dict is None:
            params_keys = ''
            params_values = ''
        else:
            params_keys = sorted(list(params_dict.keys()))
            params_values = [str(params_dict[k]) for k in params_keys]
        p = dict(
            p_keys=str(params_keys),
            p_values=str(params_values)
        )
        return p
        
    def perfs_from_dict(self, perfs):
        perf_dict = {}
        for phase, metrics in perfs.items():
            for mname, metric in metrics.items():
                if mname == 'phase':
                    continue
                k = f'{phase}_{mname}'
                perf_dict[k] = metric
        return perf_dict
    
    def perfs_from_table(self, r_obj):
        perfs = {}
        for phase in ['train','val','test']:
            perfs[phase] = {}
            for k,v in vars(r_obj).items():
                if phase in k:
                    mname = k.split('_')[1]
                    perfs[phase][mname] = v
        return perfs

    def results_from_dict(self,dataset,model,exp,result,grid,perfs):
        key_pairs = {}
        if dataset:
            d_p=self.param_from_dict(dataset['params'])
            key_pairs.update(dict(dataset=dataset['name'],d_p_k=d_p['p_keys'],d_p_v=d_p['p_values']))
        if model:
            m_p=self.param_from_dict(model['params'])
            key_pairs.update(dict(model=model['name'],m_p_k=m_p['p_keys'],m_p_v=m_p['p_values']))
        if exp:
            key_pairs.update(exp)
        if result:
            key_pairs.update(result)
        if grid:
            key_pairs.update(grid)
        if perfs:
            key_pairs.update(self.perfs_from_dict(perfs))
        r_obj = Results(**key_pairs)
        return r_obj, list(key_pairs.keys())

    def insert_results_table(self,dataset,model,exp,result,grid,perfs):
        r_obj,_ = self.results_from_dict(dataset,model,exp,result,grid,perfs)
        self.insert(r_obj)

    def retrieve_results_table(self,dataset,model,exp,result,grid):
        r_obj, keys = self.results_from_dict(dataset,model,exp,result,grid,None)
        statm = self.get_statement_exact_match(r_obj)
        r_obj = self.get_by_statm(statm)
        return r_obj
    
    def get_perfs(self,dataset,model,exp,result,grid):
        r_obj=self.retrieve_results_table(dataset,model,exp,result,None)
        perfs = {}
        if r_obj is not None:
            perfs = self.perfs_from_table(r_obj)
        return perfs

    def save_results(self,dataset,model,exp,result,grid,perfs):
        self.insert_results_table(dataset,model,exp,result,grid,perfs)

    def delete_results_by_keys(self, dataset,model,exp,result):
        r_obj,_ = self.results_from_dict(dataset,model,exp,result,None)
        statm = self.delete_statement_exact_match(r_obj)
        self.session.execute(statm)
        self.session.commit()

    def update_grid(self, update_grid_list):
        if len(update_grid_list) == 0:
            return
        logger.info(f'[update_grid] updating rows {len(update_grid_list)}')
        statm = (
            update(Results).
            where(Results.id==bindparam('_id')).
            values(grid_id=bindparam('grid_id'))
        )
        
        self.execute_rows(statm,update_grid_list)

    def delete_grid(self, delete_grid_list):
        if len(delete_grid_list) == 0:
            return
        logger.info(f'[delete_grid] deleting rows {len(delete_grid_list)}')
        statm = (
            delete(Results).
            where(Results.id==bindparam('_id'))
        )
        self.execute_rows(statm,delete_grid_list)

    def show_all(self, table):
        statm = select(table)
        results = self.session.execute(statm).all()
        for r in results:
            logger.info(r)

        return results

    def to_pandas(self, table):
        with self.engine.connect() as conn:
            df = pd.read_sql(table.__table__.select(), conn)
        logger.info(df)
        return df

    def to_pandas_statm(self, statm):
        with self.engine.connect() as conn:
            df = pd.read_sql(statm, conn)
        return df

    def _split_params_to_df(self, p_k, p_v, needed):
        p_k = p_k.apply(lambda x: '[]' if x == '' else eval(x)).to_list()
        p_v = p_v.apply(lambda x: '[]' if x == '' else eval(x)).to_list()
        series = {}
        for k in needed:
            series[k] = []
        rows = 0
        for params_k, params_v in zip(p_k, p_v):
            rows += 1
            for k, v in zip(params_k, params_v):
                if k in needed:
                    series[k].append(v)

            for k in series:
                if len(series[k]) < rows:
                    series[k].append(None)
        df_param = pd.DataFrame.from_dict(series)
        return df_param

    def split_param_dataframe(self, df, d_need, m_need):
        df_d_params = self._split_params_to_df(df['d_p_k'], df['d_p_v'], d_need)
        df_m_params = self._split_params_to_df(df['m_p_k'], df['m_p_v'], m_need)
        
        df = df.drop(['d_p_k','d_p_v','m_p_k','m_p_v'], axis=1)
        df = pd.concat([df, df_d_params,df_m_params],axis=1)
        return df

    def get_all_results_as_dataframe(self):
        df = self.to_pandas_statm(self.get_all_results())
        df = df.set_index('id')
        return df

    def get_all_results(self):
        statm = select(Results)
        return statm


def test1():
    dbm = DatabaseManager('test_new.db')

    mymodel = dict(name='CNN_BiLSTM', params=dict(bn=True, nclass=2, text='abc'))
    model_obj, _ = dbm.retrieve_or_insert_model(mymodel, mymodel['params'])

    mydataset = dict(name='ECO', params=dict(case='1', extraction=True))
    dataset_obj, _ = dbm.retrieve_or_insert_dataset(mydataset, mydataset['params'])
    
    exper = dict(exp=1,splits='3:1:1',nrepeat='10',nfold='1')
    exper_obj, _, _ = dbm.retrieve_or_insert_experiments(exper, model_obj, dataset_obj)

    from datetime import datetime
    perf = dict(phase='train', accuracy='0.9', f1macro='0.91', 
        timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y'))
    train_perf_obj = dbm.retrieve_or_insert_performances(perf)
    
    perf = dict(phase='val', accuracy='0.9', f1macro='0.91', 
        timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y'))
    val_perf_obj = dbm.retrieve_or_insert_performances(perf)

    perf = dict(phase='test', accuracy='0.9', f1macro='0.91', 
        timestamp=datetime.strptime('01-02-2020', '%m-%d-%Y'))
    test_perf_obj = dbm.retrieve_or_insert_performances(perf)

    result = dict(fold=1, repeat=1)
    result_obj, _,_,_,_ = dbm.retrieve_or_insert_results(result, exper_obj, train_perf_obj, 
        val_perf_obj, test_perf_obj)
    
    # dbm.show_all(Models)
    dbm.show_model_join()
    # dbm.show_all(Datasets)
    dbm.show_all(Results)

def test2():
    dbm = DatabaseManager('test_new.db')
    model = dict(name='CNN_BiLSTM', params=dict(bn=True, nclass=2, text='abc'))
    dataset = dict(name='ECO', params=dict(case='1', extraction=True))
    exper = dict(exp=1,splits='3:1:1',nrepeat='10',nfold='1')

    from datetime import datetime
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

    dbm.show_all(Results)
    dbm.get_average_repeat()

    dbm.delete_results(1, {'fold':1, 'repeat':2})
    dbm.show_all(Results)
    
    
if __name__ == '__main__':
    from ..logger.get_configured_logger import get_console_logger
    my_logger = get_console_logger(True)
    # test1()
    test2()

import logging
import os
import time
from datetime import datetime

from ml_toolkit.database.database_control import DatabaseManager
from ml_toolkit.modscaner import ModuleScannerBase
from ml_toolkit.trainer.options import setup_arg_fake, Options

logger = logging.getLogger(__name__)

class HyperparameterManager():
    def __init__(self, dataselection: ModuleScannerBase, modelselection: ModuleScannerBase, args=None, db_name=None) -> None:
        self.ds = dataselection
        self.ms = modelselection
        if db_name is not None:
            self.db_name = db_name
            self.db = DatabaseManager(db_name)
        else:
            self.db = None
        if args is not None:
            self.update_values_from_args(args)

        self.options = Options()
        
    def _init_keys(self):
        self.dataset_param_keys, _ = self.ds.getParams(self.args.dataset, args=self.args)
        self.model_param_keys, _ = self.ms.getParams(self.args.model)
        if self.args.ensemble:
            self.model_param_keys += ['ensemble']
        self.exper_keys = ['exp','nfold','nrepeat', 'epochs','splits']
        self.result_keys = ['fold','repeat']
        self.grid_keys = ['grid_id']

        self.params_dict = {
            'dataset': self.dataset_param_keys, 
            'model': self.model_param_keys,
            'exp': self.exper_keys,
            'result': self.result_keys,
            'grid': self.grid_keys
        }

    def update_values_from_args(self, args):
        self.args = args
        self._init_keys()
        self._update_values()

    def update_values_from_namespace(self,namespace):
        self.args = self.options.setup_arg_fill_default(namespace)
        self._init_keys()
        self._update_values()

    def _update_values(self):
        keys = [*self.model_param_keys, *self.dataset_param_keys, *self.exper_keys, *self.result_keys]
        self._key_params = {k: vars(self.args)[k] for k in keys}

        keys = ['exp','model','dataset','fold','repeat']
        pairs = {k: self.key_params[k] for k in keys if k in self.key_params}
        time_stamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        pairs.update({'time':time_stamp, 'data':self.args.dataset, 'model':self.args.model})
        self._unique_time_str = "-".join([f"{k}={v}" for k,v in pairs.items()])

    @property
    def key_params(self):
        return self._key_params

    @property
    def unique_time_str(self):
        return self._unique_time_str
    
    @property
    def version_name(self):
        jobid = os.environ.get('SLURM_JOB_ID')
        if jobid is not None:
            version = f'{jobid}_fold_{self.args.fold}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
        else:
            version = f'fold_{self.args.fold}_{time.strftime("%h-%d-%Y-%H:%M:%S")}'
        return version

    def __str__(self):
        return "-".join([f"{k}={v}" for k,v in self.key_params.items()])

    def get_params(self, param_type):
        if param_type not in self.params_dict.keys():
            raise ValueError(f"param_type should in {list(self.params_dict.keys())}")
        if self.params_dict[param_type] is None:
            return None
        return {k: vars(self.args)[k] for k in self.params_dict[param_type]}

    def check_finished(self):
        if self.args.no_db:
            return False
        if self.db is None:
            raise ValueError('[Hyperparameter] DB is not initialized!')
        dataset, model, exp, result, grid = self.get_all_params()
        perfs = self.db.get_perfs(dataset,model,exp,result,grid)
        if perfs == {}:
            return False
        elif perfs != {}:
            all_perfs_exist = 0
            for k, v in perfs.items():
                if v:
                    all_perfs_exist += 1
            if all_perfs_exist == 3:
                return True
            elif all_perfs_exist >= 0:    
                return False
            else:
                return False
        return False

    def merge_test_results(self, results):
        timestamp = datetime.now()
        perfs = {}
        for phase in ['train','val','test']:
            perf = dict(
                phase = phase,
                accuracy = results.get(f'{phase}_acc', None),
                accmacro = results.get(f'{phase}_accmacro', None),
                f1macro = results.get(f'{phase}_f1macro', None),
                epoch = results.get(f'{phase}_epoch', None),
                confmx = str(results.get(f'{phase}_confmx', None)),
                timestamp = timestamp,
            )
            perfs[phase] = perf
            logger.info(perf)

        return perfs

    def save_results(self, result):
        if self.args.no_db:
            return
        if self.db is None:
            raise ValueError('[Hyperparameter] DB is not initialized!')
        perfs = self.merge_test_results(result)
        dataset, model, exp, result, grid = self.get_all_params()
        self.db.save_results(dataset, model, exp, result, grid, perfs)

    def get_all_params(self):
        dataset = dict(name=self.args.dataset, params=self.get_params('dataset'))
        model = dict(name=self.args.model, params=self.get_params('model'))
        exp = self.get_params('exp')
        result = self.get_params('result')
        grid = self.get_params('grid')
        return dataset, model, exp, result, grid

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

    def get_all_params_dict(self):
        dataset, model, exp, result, grid = self.get_all_params()
        my_params = {}
        my_params.update({"dataset":dataset['name']})
        my_params.update({"model":model['name']})
        d_p = self.param_from_dict(dataset['params'])
        m_p = self.param_from_dict(model['params'])
        my_params.update({'d_p_k':d_p['p_keys'],'d_p_v':d_p['p_values']})
        my_params.update({'m_p_k':m_p['p_keys'],'m_p_v':m_p['p_values']})
        my_params.update(exp)
        my_params.update(result)
        # my_params.update(grid)
        return my_params
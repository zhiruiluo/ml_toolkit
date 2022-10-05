import itertools
import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from ml_toolkit.database.database_control import DatabaseManager
from ml_toolkit.gridsearch.utils import TaskState, allbut
from ml_toolkit.trainer.parameter_manager import HyperparameterManager
from ml_toolkit.utils.param_helper import (get_max,
                                           get_max_by_single_model_dataset)
from ml_toolkit.utils.param_production import (flatten_params, from_options,
                                               parameters_from_yaml,
                                               parameters_to_yaml,
                                               params_to_namespace)

logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass(frozen=True)
class Params():
    dataset: str
    model: str
    d_p_k: str
    d_p_v: str
    m_p_k: str
    m_p_v: str
    exp: int
    splits: str
    nfold: int
    nrepeat: int
    epochs: int
    fold: int
    repeat: int


class GridSearch():
    def __init__(self, parameters: Union[dict, str]) -> None:
        if isinstance(parameters, str):
            self.parameters_from_yaml(parameters)
        elif isinstance(parameters, dict):
            self.params_dict = parameters
        self.db = DatabaseManager(self.params_dict['system']['expname']+'.db')
        self.hpm = HyperparameterManager()
        self.grid_id_to_parameters = {}
        self.task_status = {}
        self.task_count = {}
        self.task_total = 0
        self.prepare()

    def parameters_to_yaml(self, path):
        parameters_to_yaml(path, self.params_dict)
    
    def parameters_from_yaml(self, path):
        self.params_dict = parameters_from_yaml(path)
        self.params_dict.pop('grid_config')

    def get_params_list_for_unfinished(self):
        df = self.db.get_all_results_as_dataframe()
        df_dict = {}
        for result_id, row in df.iterrows():
            row_dict = row.to_dict()
            params = Params(**self.select_dict(row_dict))
            df_dict[params] = row['grid_id']
        
        unfinished_params_list = []
        for grid_id, p in enumerate(from_options(self.params_dict)):
            self.hpm.update_values_from_namespace(params_to_namespace(p))
            params = Params(**self.hpm.get_all_params_dict())
            if params not in df_dict:
                unfinished_params_list.append((grid_id, flatten_params(p)))
        
        return unfinished_params_list

    def get_params_list_for_all(self):
        params_list = []
        for grid_id, p in enumerate(from_options(self.params_dict)):
            params_list.append((grid_id, flatten_params(p)))
        return params_list

    def prepare(self):
        if self.params_dict is None:
            raise ValueError("[GridSearch] assign parameters did't setup!")
        
        self.task_status = {}
        self.grid_id_to_parameters = {}
        total = 0
        for grid_id, p in enumerate(from_options(self.params_dict)):
            self.grid_id_to_parameters[grid_id] = p
            self.task_status[grid_id] = TaskState.INITIAL_STATE
            total += 1
        self.task_total = total
        
    def get_dict_from_params(self, table):
        if table not in self.parameters.keys():
            raise ValueError(f"param_type should in {list(self.params_dict.keys())}")
        if self.params_dict[table] is None:
            return None
        return {k: vars(self.args)[k] for k in self.params_dict[table]}

    def get_phase_status(self, df):
        phase_finished = {'train': False, 'val':False, 'test':False}
        finished_count = 0
        for phase in ['train','val','test']:
            if not pd.isna(df[f'{phase}_accuracy']):
                phase_finished[phase] = True
                finished_count += 1
        if finished_count == 0:
            return TaskState.NOT_START
        elif finished_count <= 2:
            return TaskState.IN_PROGRESS
        elif finished_count == 3:
            return TaskState.FINISHED

    def select_dict(self, row_dict):
        keys = ['dataset','model','d_p_k','d_p_v','m_p_k','m_p_v','exp','nfold','nrepeat','epochs','splits',
        'fold','repeat']
        new_dict = {k:row_dict[k] for k in keys}
        return new_dict

    def _get_grid_id_to_parameters(self):
        grid_id_to_parameters = {}
        for grid_id, p in enumerate(from_options(self.params_dict)):
            self.hpm.update_values_from_namespace(params_to_namespace(p))
            params = Params(**self.hpm.get_all_params_dict())
            grid_id_to_parameters[grid_id] = params
        return grid_id_to_parameters

    def _get_params_to_result_ids(self):
        df = self.db.get_all_results_as_dataframe()
        logger.info(f'[update] df count {len(df)}')
        params_resultId_dict = {}
        for result_id, row in df.iterrows():
            row_dict = row.to_dict()
            params = Params(**self.select_dict(row_dict))
            params_resultId_dict[params] = result_id
        return params_resultId_dict, df

    def _get_mismatch_grid_list(self):
        grid_id_to_parameters = self._get_grid_id_to_parameters()
        params_resultId_dict, df = self._get_params_to_result_ids()

        mismatch_count = 0
        update_grid_list = []
        reached_results_set = set()

        for real_grid_id, params in grid_id_to_parameters.items():
            if params not in params_resultId_dict:
                continue
            result_id = params_resultId_dict[params]
            
            df_grid_id = df.loc[result_id, 'grid_id'].tolist()
            reached_results_set.add(result_id)
            if df_grid_id != real_grid_id:
                df.loc[result_id,'grid_id'] = real_grid_id
                mismatch_count += 1
                update_grid_list.append({'_id': result_id, 'grid_id': real_grid_id})

        return update_grid_list, reached_results_set

    def update_grid(self):
        update_grid_list, reached_results_set = self._get_mismatch_grid_list()
        logger.info(f'[update] {len(update_grid_list)}')
        
        self.db.update_grid(update_grid_list)

        df = self.db.get_all_results_as_dataframe()
        delete_grid_list = []
        unreached_results_set = set()
        for result_id, row in df.iterrows():
            if result_id not in reached_results_set:
                unreached_results_set.add(result_id)
                delete_grid_list.append({'_id':result_id})
        logger.info(f'[unreached] {len(unreached_results_set)} df len {df.shape[0]}')

        
        self.db.delete_grid(delete_grid_list)

    def refresh(self):
        finished = 0
        in_progress = 0
        df = self.db.get_all_results_as_dataframe()
        self.cached_record = df
        temp_status = set()
        for result_id, result in df.iterrows():
            grid_id = result['grid_id']
            assert grid_id not in temp_status
            temp_status.add(grid_id)
            if self.task_status[grid_id] == TaskState.FINISHED:
                finished += 1
                continue
            status = self.get_phase_status(result)
            
            self.task_status[grid_id] = status
            if status == TaskState.FINISHED:
                finished += 1
            elif status == TaskState.IN_PROGRESS:
                in_progress += 1
        
        notstart = self.task_total-finished-in_progress
        self.task_count = {'total': self.task_total, 'finished': finished, 'in_progress': in_progress, 'notstart': notstart}

    def check_status(self):
        self.refresh()
        logger.info('[check_status] %s', self.task_count)
        logger.info('[check_status] prograss {:.2f}%'.format((self.task_count["finished"]/self.task_total)*100))

    def build_df_all(self):
        params = []
        for i, p in enumerate(from_options(self.params_dict)):
            print(p)
            break

    def build_tree(self):
        if not hasattr(self, 'cached_record'):
            self.refresh()
        
        self.build_df_all()
        for k, v in self.params_dict.items():
            logger.info(f'[build_tree] {k} {v}')

    def _get_all_keyed_results(self) -> pd.DataFrame:
        # results = []
        results = self.db.get_all_results_as_dataframe()
        if results.empty:
            return results
        results = results.sort_values('grid_id')
        self.results = results
        # self.results = pd.concat(results, axis=0)
        return self.results

    def save_results_csv(self, dir='./results'):
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        results = self._get_all_keyed_results()
        fn=f"gridsearch_{self.params_dict['system']['expname']}.csv"
        results.to_csv(dir.joinpath(fn), index=False, header=True)
        logger.info(f'[Gridsearch] save {fn}')

    def save_repeat_mean_csv(self, dir='./results'):
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        results = self._get_repeat_mean()
        fn=f"gridsearch_repeat_mean_{self.params_dict['system']['expname']}.csv"
        results.to_csv(dir.joinpath(fn), index=False, header=True)
        logger.info(f'[Gridsearch] save {fn}')
        return results

    def get_pivot_df(self, dataset, d_max_keys, d_mean_keys, model, m_max_keys, m_mean_keys, metric, orientation):
        """ orientation: 0 for dataset as index and model as column, 1 vice versa """
        results = self._get_repeat_mean()
        if results.empty:
            return None
        df_max = get_max(results, dataset, d_max_keys, d_mean_keys, model, m_max_keys, m_mean_keys, metric)
        if df_max.empty:
            logger.info(f'[plot skipped] {dataset} {model}')
            return None
        dataset_concat, model_concat = df_max.columns[0:2]
        if orientation == 0:
            index = dataset_concat
            columns = model_concat
        else:
            index = model_concat
            columns = dataset_concat
        df_pivot = pd.pivot_table(df_max, metric, index=index, columns=columns)
        return df_pivot
    
    def get_max_results(self, dataset, d_max_keys, d_mean_keys, model, m_max_keys, m_mean_keys, metric):
        results = self._get_repeat_mean()
        if results.empty:
            return None
        df_max = get_max(results, dataset, d_max_keys, d_mean_keys, model, m_max_keys, m_mean_keys, metric)
        if df_max.empty:
            logger.info(f'[plot skipped] {dataset} {model}')
            return None
        return df_max

    def get_max_results_v2(self, datasets, d_max_keys, d_mean_keys, models, m_max_keys, m_mean_keys, major_metric, other_metrics, statistics):
        # results = self._get_repeat_mean()
        results = self._get_all_keyed_results()
        if datasets == '*':
            datasets = results['dataset'].unique()
        if models == '*':
            models = results['model'].unique()
        max_results = []
        for dataset, model in itertools.product(datasets, models):
            df_max = get_max_by_single_model_dataset(results, [dataset], d_max_keys, d_mean_keys, [model], 
                    m_max_keys, m_mean_keys, major_metric, other_metrics, True, statistics)
            if df_max.empty:
                logger.info(f'[plot skipped] {dataset} {model}')
                return None
            max_results.append(df_max)
        if len(max_results) != 0:
            max_results = pd.concat(max_results, axis=0).sort_values(by=['dataset',major_metric]).reset_index(drop=True)
            # max_results = max_results.astype({'count': int})
            return max_results
        else:
            return None
        

    def plot_table(self, df_data, orientation, fn):
        dataset_concat, model_concat, metric = df_data.columns
        if orientation == 0:
            index = dataset_concat
            columns = model_concat
        else:
            index = model_concat
            columns = dataset_concat
        df_pivot = pd.pivot_table(df_data, metric, index=index, columns=columns)
        
        ax = df_pivot.plot.barh(figsize=(10,10))
        min_v = df_data.min().values[-1]
        max_v = df_data.max().values[-1]
        ax.set_xlim(min_v-0.1,max_v+0.1)

        if orientation == 0:
            ax.set_title(f'{metric} plot on dataset vs model')
            ax.set_ylabel(metric)
        else:
            ax.set_title(f'{metric} plot on model vs dataset')
            ax.set_ylabel(metric)
        # plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout()
        plt.savefig(fn)

    def plot(self, datasets, d_max_keys, d_mean_keys, models, m_max_keys, m_mean_keys, metrics, orientation):
        logger.info(f'[plot] {datasets}, {models}')
        results = self._get_repeat_mean()
        for dataset in datasets:
            for model in models:
                for metric in metrics:
                    df_max = get_max(results, dataset, d_max_keys, d_mean_keys, model, m_max_keys, m_mean_keys, metric)
                    if df_max.empty:
                        logger.info(f'[plot skipped] {dataset} {model}')
                        continue
                    dataset_concat, model_concat = df_max.columns[0:2]
                    if orientation == 0:
                        index = dataset_concat
                        columns = model_concat
                    else:
                        index = model_concat
                        columns = dataset_concat
                    df_pivot = pd.pivot_table(df_max, metric, index=index, columns=columns)
                    
                    ax = df_pivot.plot.barh(figsize=(10,10))
                    min_v = df_max.min().values[-1]
                    max_v = df_max.max().values[-1]
                    ax.set_xlim(min_v-0.1,max_v+0.1)

                    if orientation == 0:
                        ax.set_title(f'{metric} plot on {dataset} vs {model}')
                        ax.set_ylabel(metric)
                    else:
                        ax.set_title(f'{metric} plot on {model} vs {dataset}')
                        ax.set_ylabel(metric)
                    # plt.legend(bbox_to_anchor=(1.0, 1.0))
                    plt.tight_layout()
                    plt.savefig(f"./results/gridsearch_{dataset}_{model}_{self.params_dict['system']['expname']}_ort-{orientation}_{metric}.png")

    def _get_repeat_mean(self):
        df_results = self._get_all_keyed_results()
        if df_results.empty:
            return df_results
        results_grp = df_results.groupby(by=allbut(df_results, ['repeat']))
        
        grp_mean =  results_grp.mean()
        grp_count = results_grp.count()['test_accuracy']
        grp_count.name = 'count'
        df_repeat_mean = pd.concat([grp_mean,grp_count],axis=1).reset_index().drop(['repeat','grid_id'],axis=1)
        return df_repeat_mean

    def get_evaluates_Repeat(self):
        results = self._get_all_keyed_results()
        grouped = results.groupby(['dataset','d_p_v','d_p_k','model','m_p_k','m_p_v','exp','nfold','nrepeat','epochs','splits','fold'])
        mean_repeat = grouped.mean().reset_index()
        std_repeat = grouped.std().reset_index()
        best_test_acc_row = mean_repeat[mean_repeat['test_accuracy'] == mean_repeat['test_accuracy'].max()].iloc[[0],:]
        best_acc_std = std_repeat[mean_repeat['test_accuracy'] == mean_repeat['test_accuracy'].max()].iloc[[0],:]['test_accuracy'].to_numpy().item()
        best_accuracy = best_test_acc_row['test_accuracy'].to_numpy().item()
        best_params = best_test_acc_row[['dataset','d_p_v','d_p_k','model','m_p_k','m_p_v', 'exp', 'nfold','nrepeat', 'epochs','splits','fold']].values.tolist()
        return {'best_acc': best_accuracy,'best_acc_std': best_acc_std,'best_params': best_params, 'best_row': best_test_acc_row}


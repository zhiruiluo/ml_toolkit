# import pandas as pd
# from multiprocess.pool import Pool
# from typing import Tuple,List
from enum import Enum
import logging 
logger = logging.getLogger(__name__)

class TaskState(Enum):
    INITIAL_STATE = 0
    NOT_START = 1
    IN_PROGRESS = 2
    FINISHED = 3

# def fast_to_dict_records(df: pd.DataFrame) -> List:
#     data = df.values.tolist()
#     columns = df.columns.tolist() 
#     return [
#         dict(zip(columns, datum))
#         for datum in data
#     ]

# def select_result_df(conditions):
#     df,dataset,model,exp,result,m_p,d_p,grid_id = conditions
#     df_result = df[
#         (df['exp']==exp['exp']) &
#         (df['model']==model['name']) &
#         (df['m_p_k']==m_p['p_keys']) &
#         (df['m_p_v']==m_p['p_values']) &
#         (df['dataset']==dataset['name']) &
#         (df['d_p_k']==d_p['p_keys']) &
#         (df['d_p_v']==d_p['p_values']) & 
#         (df['splits']==exp['splits']) &
#         (df['nrepeat']==exp['nrepeat']) &
#         (df['nfold']==exp['nfold']) &
#         (df['epochs']==exp['epochs']) &
#         (df['fold']==result['fold']) &
#         (df['repeat']==result['repeat'])
#     ]
#     return [grid_id, df_result]
    
# def select_result_by_id(conditions):
#     df, id, i = conditions
#     result = df[df['id'] == id]
#     return [i, result]

# def pool_select(conditions: List[Tuple], func) -> List:
#     with Pool(10) as p:
#         selected_result_list = p.map(func, conditions)

#     return selected_result_list

# def from_param( p):
#     d_dict = p['dataset'].copy()
#     d_name = d_dict['dataset']
#     d_dict.pop('dataset')
#     dataset = dict(name=d_name, params=d_dict)

#     m_dict = p['model'].copy()
#     m_name = m_dict['model']
#     m_dict.pop('model')
#     model = dict(name=m_name, params=m_dict)
#     exp = p['exp']
#     result = p['result']
#     return dataset, model, exp, result

# def perfs_from_df(df):
#     # result_dict = df.to_dict('records')[0]
#     result_dict = fast_to_dict_records(df)[0]
#     perfs = {}
#     for phase in ['train','val','test']:
#         perfs[phase] = {}
#         for k,v in result_dict.items():
#             if phase in k:
#                 mname = k.split('_')[1]
#                 perfs[phase][mname] = v
#     return perfs

def allbut(df, excluded_key):
    all_key = ['dataset','d_p_v','d_p_k','model','m_p_k','m_p_v','exp','nfold','nrepeat','epochs','splits','fold','repeat']
    return [k for k in all_key if k not in excluded_key]

# def split_params_to_df(p_k, p_v, needed):
#     p_k = p_k.apply(lambda x: '[]' if x == '' else eval(x)).to_list()
#     p_v = p_v.apply(lambda x: '[]' if x == '' else eval(x)).to_list()
#     series = {}
#     for k in needed:
#         series[k] = []
#     rows = 0
#     for params_k, params_v in zip(p_k, p_v):
#         rows += 1
#         for k, v in zip(params_k, params_v):
#             if k in needed:
#                 series[k].append(v)

#         for k in series:
#             if len(series[k]) < rows:
#                 series[k].append(None)
#     df_param = pd.DataFrame.from_dict(series)
#     return df_param

# def split_param_dataframe(df, d_need, m_need):
#     df_d_params = split_params_to_df(df['d_p_k'], df['d_p_v'], d_need)
#     df_m_params = split_params_to_df(df['m_p_k'], df['m_p_v'], m_need)
    
#     df = df.drop(['d_p_k','d_p_v','m_p_k','m_p_v'], axis=1)
#     df = pd.concat([df, df_d_params,df_m_params],axis=1)
#     return df
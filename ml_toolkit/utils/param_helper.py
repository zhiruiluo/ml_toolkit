from turtle import st
from numpy import isin
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def split_params_to_df(df_p_k, df_p_v, needed):
    p_k = df_p_k.apply(lambda x: '[]' if x == '' else eval(x)).to_list()
    p_v = df_p_v.apply(lambda x: '[]' if x == '' else eval(x)).to_list()
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
    df_param.index = df_p_k.index
    return df_param

def split_param_with_padding_none(df):
    d_keys = set()
    m_keys = set()
    
    for index, row in df[['d_p_k','m_p_k']].drop_duplicates().iterrows():
        d_key_list = eval(row['d_p_k'])
        m_key_list = eval(row['m_p_k'])
        d_keys.update(d_key_list)
        m_keys.update(m_key_list) 

    d_keys = sorted(list(d_keys))
    m_keys = sorted(list(m_keys))

    df_d_params = split_params_to_df(df['d_p_k'], df['d_p_v'], d_keys)
    df_m_params = split_params_to_df(df['m_p_k'], df['m_p_v'], m_keys)
    return df_d_params, df_m_params


def remove_keys(src_keys, removed_keys):
    return sorted(list(set(src_keys) - set(removed_keys)))

def union_keys(keys_1, keys_2):
    return sorted(list(set(keys_1).union(set(keys_2))))

def get_max(results, dataset, d_max_keys, d_mean_keys, model, m_max_keys, m_mean_keys, metric):
    if not isinstance(dataset, list):
        dataset = [dataset]
    if not isinstance(model, list):
        model = [model]
    # select given dataset and model
    df_r = results[(results['dataset'].isin(dataset)) & (results['model'].isin(model))].copy()
    if df_r.empty:
        logger.info(f'[get_max] results_empty')
        return df_r
    # split aggragated params into columns
    df_d_params, df_m_params = split_param_with_padding_none(df_r)
    # logger.info(f'df_d_params {df_d_params}')
    # logger.info(f'df_m_params {df_m_params}')
    df_r = df_r.drop(['d_p_k','d_p_v','m_p_k','m_p_v'], axis=1)
    df_r = pd.concat([df_r,df_d_params,df_m_params],axis=1)
    # drop non-needed keys
    if isinstance(d_max_keys, str):
        if d_max_keys == 'all':
            d_max_keys = df_d_params.columns.values.tolist()
            d_max_keys = remove_keys(d_max_keys, d_mean_keys)
    if isinstance(m_max_keys, str):
        if m_max_keys == 'all':
            m_max_keys = df_m_params.columns.values.tolist()
            m_max_keys = remove_keys(m_max_keys, m_mean_keys)
    need_keys = ['dataset', *d_max_keys, *d_mean_keys, 'model', *m_max_keys, *m_mean_keys]
    drop_keys = remove_keys(df_r.columns, union_keys(need_keys,[metric]))
    # logger.info(drop_keys)
    df_r = df_r.drop(drop_keys, axis=1)
    # averaging mean keys 
    mean_keys = sorted(list(set(df_r.columns) - set(d_mean_keys+m_mean_keys+[metric])))
    # logger.info(mean_keys)
    df_r = df_r.groupby(mean_keys)[[*mean_keys,metric]].mean().reset_index()
    # get the row with max metrics
    # max_keys = sorted(list(set(df_r.columns) - set(d_max_keys+m_max_keys)))
    max_keys = remove_keys(df_r.columns, union_keys(d_max_keys,m_max_keys))
    # logger.info(max_keys)
    max_idx = df_r.groupby(max_keys)[metric].transform(max) == df_r[metric]
    df_max = df_r[max_idx].reset_index(drop=True)
    # logger.info(df_max.columns)
    # aggragate dataset keys and model keys
    df_combine_d = df_max[['dataset', *d_max_keys]].agg('-'.join, axis=1)
    df_combine_m = df_max[['model', *m_max_keys]].agg('-'.join, axis=1)
    df_new_max = pd.concat([df_combine_d, df_combine_m, df_max[[metric]]],axis=1)
    df_new_max.columns = ['-'.join(['dataset',*d_max_keys]), '-'.join(['model',*m_max_keys]), metric]
    return df_new_max


def get_mean_and_statistics(df_r, group_keys, all_metrics, statistics):
    df = df_r.groupby(group_keys)[[*group_keys,*all_metrics]].agg(statistics)
    columns = []
    for col in df.columns.values:
        if col[1] == 'mean':
            columns += [col[0].strip()]
        else:
            columns += ['_'.join(col).strip()]
    df.columns = columns
    df = df.reset_index()
    return df, columns

def get_default_max_keys(params_keys, input_max_keys, input_mean_keys):
    if isinstance(input_max_keys,str) and input_max_keys in ['all','*']:
        max_keys = remove_keys(params_keys, input_mean_keys)
        return max_keys
    else:
        return input_max_keys

def split_params_into_columns(df_results):
    df_d_params, df_m_params = split_param_with_padding_none(df_results)
    df_r = df_results.drop(['d_p_k','d_p_v','m_p_k','m_p_v'], axis=1)
    df_r = pd.concat([df_r,df_d_params,df_m_params],axis=1)
    d_params_keys = df_d_params.columns.values.tolist()
    m_params_keys = df_m_params.columns.values.tolist()
    return df_r, d_params_keys, m_params_keys

def get_max_by_single_model_dataset(
        results, dataset, d_max_keys, d_mean_keys, 
        model, m_max_keys, m_mean_keys, major_metric, 
        other_metrics, short_col_name, statistics):
    df_r = results[(results['dataset'].isin(dataset)) & (results['model'].isin(model))].copy()
    if df_r.empty:
        logger.info(f'[get_max] results_empty')
        return df_r
    # split aggragated params into columns
    df_r, d_params_keys, m_params_key = split_params_into_columns(df_r)
    d_max_keys = get_default_max_keys(d_params_keys, d_max_keys, d_mean_keys)
    m_max_keys = get_default_max_keys(m_params_key, m_max_keys, m_mean_keys)
    # drop no needed keys
    need_keys = ['dataset', *d_max_keys, *d_mean_keys, 'model', *m_max_keys, *m_mean_keys]
    all_metrics = [major_metric] + other_metrics
    drop_keys = remove_keys(df_r.columns, union_keys(need_keys, all_metrics))
    df_r = df_r.drop(drop_keys, axis=1)
    # averaging mean keys 
    mean_keys = sorted(list(set(df_r.columns) - set(d_mean_keys+m_mean_keys+all_metrics)))
    df_r, all_metrics = get_mean_and_statistics(df_r,mean_keys,all_metrics, statistics)
    # get the row with max metrics
    max_keys = remove_keys(df_r.columns, union_keys(d_max_keys,m_max_keys))
    max_idx = df_r.groupby(max_keys)[major_metric].transform(max) == df_r[major_metric]
    df_max = df_r[max_idx].reset_index(drop=True)
    # aggragate dataset keys and model keys
    if not short_col_name:
        df_combine_d = df_max[['dataset', *d_max_keys]].agg('-'.join, axis=1)
        df_combine_m = df_max[['model', *m_max_keys]].agg('-'.join, axis=1)
        df_new_max = pd.concat([df_combine_d, df_combine_m, df_max[all_metrics]],axis=1)
        df_new_max.columns = ['-'.join(['dataset',*d_max_keys]), '-'.join(['model',*m_max_keys]), *all_metrics]
    else:
        df_combine_d = df_max[['dataset', *d_max_keys]].agg('-'.join, axis=1)
        df_combine_m = df_max[['model', *m_max_keys]].agg('-'.join, axis=1)
        df_new_max = pd.concat([df_combine_d, df_combine_m, df_max[all_metrics]],axis=1)
        df_new_max.columns = ['dataset', 'model', *all_metrics]
    return df_new_max

def get_keys(df):
    df_d_params, df_m_params = split_param_with_padding_none(df)
    keys_dict = {}
    keys_dict['dataset'] = df_d_params.columns.values.tolist()
    keys_dict['model'] = df_m_params.columns.values.tolist()
    return keys_dict
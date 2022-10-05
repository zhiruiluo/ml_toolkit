import pandas as pd

def pretty_print_confmx(confmx):
    return ' '.join([f'r{i:<3d}' for i in range(len(confmx))]) + f' sum' + '\n' + \
        '\n'.join([' '.join([f'{cell.item():4d}' for cell in row]) + f'{sum(row):4d}' for row in confmx])


def pretty_print_confmx_pandas(confmx):
    pd.set_option('display.max_columns', None)
    df_confmx = pd.DataFrame(confmx.numpy())
    df_confmx['sum'] = df_confmx.sum(axis=1)
    str_confmx = str(df_confmx)
    pd.reset_option('display.max_columns')
    return str_confmx
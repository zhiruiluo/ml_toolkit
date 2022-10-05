import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class IndexSpliter():
    def __init__(self, key_name, labels, splits: str, nrepeat, index_buffer_flag=True) -> None:
        self.key_name = key_name
        self.labels = labels
        self.splits = splits
        self.nrepeat = nrepeat
        self.index_buffer_flag = index_buffer_flag
        self._index_save_path = './spliting/'

    def _splits_(self, labels):
        train_index, val_index, test_index = None,None,None
        yield train_index, val_index, test_index

    def get_split(self):
        train_index, val_index, test_index = self._setup_split_index()
        logger.info(f'[spliter summary] (splits {self.splits}) train {len(train_index)} val {len(val_index)} test {len(test_index)}')
        return train_index, val_index, test_index

    def get_split_repeat(self, repeat):
        self.repeat = repeat
        train_index, val_index, test_index = self._setup_split_index()
        logger.info(f'[spliter summary] (splits {self.splits}) train {len(train_index)} val {len(val_index)} test {len(test_index)}')
        return train_index, val_index, test_index

    def split_and_save_index(self, path):
        train_idx, val_idx, test_idx = self._splits_(self.labels)
        train_idx = [(i, 0) for i in train_idx]
        test_idx = [(i, 2) for i in test_idx]

        df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
        df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
        if val_idx is not None:
            val_idx = [(i,1) for i in val_idx]
            df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
            df_tuple = (df_train, df_val, df_test)
        else:
            df_tuple = (df_train, df_test)
        
        df_split_index = pd.concat(df_tuple, axis=0, ignore_index=True)
        if self.index_buffer_flag:
            df_split_index.to_csv(path, index=False)
            logger.info(f'[spliter] index created and saved!')
        else:
            logger.info(f'[spliter] index created without saving to file!')
        
        return df_split_index

    def _setup_split_index(self):
        if not os.path.isdir(self._index_save_path):
            os.mkdir(self._index_save_path)

        if self.nrepeat <= 1:
            path = os.path.join(self._index_save_path, f'spliting_{self.key_name}_{self.splits}.csv')
        else:
            path = os.path.join(self._index_save_path, f'spliting_{self.key_name}_{self.splits}_{self.nrepeat}_{self.repeat}.csv')
        if not os.path.isfile(path) or not self.index_buffer_flag:
            df_split_index = self.split_and_save_index(path)
        else:
            df_split_index = pd.read_csv(path)
            if len(df_split_index) != len(self.labels):
                logger.info(f'[spliter] df_split_index and labels length not match {len(df_split_index)} != {len(self.labels)}')
                df_split_index = self.split_and_save_index(path)
            else:
                logger.info(f'[spliter] Read index from file, length of index {len(df_split_index)}')

        return self._df_to_index(df_split_index)
        
    def _df_to_index(self, df_split_index):
        train_index = df_split_index[df_split_index['train_type']==0]['index'].to_numpy()
        val_index = df_split_index[df_split_index['train_type']==1]['index'].to_numpy()
        test_index = df_split_index[df_split_index['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index
        
        return train_index, val_index, test_index
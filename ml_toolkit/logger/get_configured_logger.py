import sys

sys.path.append('.')
import logging
import logging.config
import os
import time
from pathlib import Path

import yaml


class myFileHandler(logging.FileHandler):
    def __init__(self, root, exp_name='', subdirectory='', mode='a', fn='', encoding='utf-8', delay=False):
        path = os.path.join(root, exp_name, subdirectory)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if fn == '':
            fn = f"{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}.log"
        
        filename = os.path.join(path, fn)
        super().__init__(filename, mode, encoding, delay)

class errorFileHandler(logging.FileHandler):
    def __init__(self, root, exp_name='', subdirectory='', mode='a', fn='', encoding='utf-8', delay=False):
        path = os.path.join(root, exp_name, subdirectory)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if fn == '':
            fn = f"{time.strftime('%Y-%m-%d-%H:%M:%S')}_{os.getpid()}.err"
        
        filename = os.path.join(path, fn)
        super().__init__(filename, mode, encoding, delay)

class StdToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, std, fileno, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        self.std = std
        self.fileno = fileno

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line
        self.std.write(self.linebuf.rstrip())

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''

def get_yaml_config(logger_config_path):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),logger_config_path)
    with open(config_path, 'r', encoding='utf-8') as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)
    return config_dict

def get_logger_seperate_config(debug=False, exp_name = '', subdirectory='', filename = '', disable_stream_output=False, log_std=False):
    logger_config_path = 'logger_config_seperate.yaml'
    config_dict = get_yaml_config(logger_config_path)
    
    for fh, suffix in zip(['file', 'errorlog'], ['.log','.err']):
        config_dict['handlers'][fh]['fn'] = filename + suffix
        config_dict['handlers'][fh]['exp_name'] = exp_name
        config_dict['handlers'][fh]['subdirectory'] = subdirectory
    
    if debug:
        for k in config_dict['loggers'].keys():
            config_dict['loggers'][k]['level'] = 'DEBUG'
        # config_dict['loggers']['ml_toolkit']['level'] = 'DEBUG'
        config_dict['root']['level'] = 'DEBUG'
        
    if disable_stream_output:
        for k in config_dict['loggers'].keys():
            config_dict['loggers'][k]['handlers'] = ['file', 'errorlog']
        # config_dict['loggers']['ml_toolkit']['handlers'] = ['file', 'errorlog']
        # config_dict['loggers']['src']['handlers'] = ['file', 'errorlog']
        config_dict['root']['handlers'] = ['file', 'errorlog']
    logging.config.dictConfig(config_dict)
    if log_std:
        stdout_log = logging.getLogger("STDOUT")
        sl = StdToLogger(stdout_log, sys.stdout, logging.INFO)
        sys.stdout = sl
        
        stderr_log = logging.getLogger("STDERR")
        sl = StdToLogger(stderr_log, sys.stderr, logging.ERROR)
        sys.stderr = sl

    return logging.getLogger()

def get_root_logger_default_config(debug=False, exp_name = '', subdirectory='', filename = '', log_std=False):
    logger_config_path = 'logger_config.yaml'
    config_dict = get_yaml_config(logger_config_path)

    config_dict['handlers']['file']['fn'] = filename
    config_dict['handlers']['file']['exp_name'] = exp_name
    config_dict['handlers']['file']['subdirectory'] = subdirectory
    
    if debug:
        config_dict['loggers']['ml_toolkit']['level'] = 'DEBUG'
        config_dict['root']['level'] = 'DEBUG'

    logging.config.dictConfig(config_dict)

    root_logger = logging.getLogger()
    
    if log_std:
        stdout_log = logging.getLogger("STDOUT")
        sl = StdToLogger(stdout_log, sys.stdout, 1, logging.INFO)
        sys.stdout = sl
        
        stderr_log = logging.getLogger("STDERR")
        sl = StdToLogger(stderr_log, sys.stderr, 2, logging.ERROR)
        sys.stderr = sl

    return root_logger
    
def get_console_logger(debug=True):
    logger_config_path = 'logger_config_console.yaml'
    config_dict = get_yaml_config(logger_config_path)
    if debug:
        config_dict['loggers']['__main__']['level'] = 'DEBUG'
        config_dict['loggers']['ml_toolkit']['level'] = 'DEBUG'
        config_dict['loggers']['src'] = {'level': 'DEBUG'}
        config_dict['root']['level'] = 'INFO'
    logging.config.dictConfig(config_dict)

    return logging.getLogger()

def get_logger_simple():
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    return logger

def get_file_logger_simple(filename, debug=False):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger

def get_rotate_logger(filename, dir='', debug=False):
    from logging.handlers import RotatingFileHandler
    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)
    fn = dir.joinpath(filename)
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    fh = RotatingFileHandler(fn)
    fh.doRollover
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger
    

if __name__ == '__main__':
    #set_logging_config()
    config_dict = get_yaml_config()
    logging.config.dictConfig(config_dict)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug('This message should appear on the console')
    logger.info('So should this')
    logger.warning('And this, too')

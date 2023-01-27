import copy
import importlib
import inspect
import os
from pathlib import Path
import enum
from typing import Callable
import logging

logger = logging.getLogger(__name__)


def import_class(name):
    components = name.split('.')
    mod = importlib.import_module('.'.join(components[:-1]))
    mod = getattr(mod, components[-1])
    return mod

def import_function_or_class(module_name,method_name):
    module = importlib.import_module(f'{module_name}')
    method = getattr(module, method_name, None)
    if not method or not isinstance(method, type):
        module = importlib.import_module(f'{module_name}.{method_name}')
        method = getattr(module, method_name, None)
        if not method or not isinstance(method, type):
            raise ValueError(f"module {module_name}.{method_name} has no attribute '{method_name}'")
    return method

def filter_dict(func, kwarg_dict, args):
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])
    if 'args' not in kwarg_dict:
        kwarg_dict.update(args)
    else:
        raise ValueError('args exists in kwarg_dict')
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return filtered_dict

def init_class_from_namespace(class_, namespace):
    common_kwargs = filter_dict(class_, copy.deepcopy(vars(namespace)), {'args': namespace})
    return class_(**common_kwargs)

def init_module(class_, args):
    sign = inspect.signature(class_).parameters.values()
    sign = set([val.name for val in sign])
    kwarg_dict = {'args': args}
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return class_(**filtered_dict)

def default_model_dict_init(root):
    model_dict = {}
    root = Path(root)
    modules = os.listdir(root)
    for mod in modules:
        if mod.endswith('__init__.py') or mod == '__pycache__':
            continue
        if mod.endswith('.py'):
            model_dict[mod[:-3]] = str(root.joinpath(mod[:-3])).replace('/','.') + '.' +  mod[:-3]
        else:
            model_dict[mod] = str(root.joinpath(mod[:-3])).replace('/','.') + '.' +  mod
    return model_dict


class ModuleScannerBase():
    def __init__(self, root, module_category, module_dict_init: Callable = None) -> None:
        self.root = Path(root)
        self.module_category = module_category
        self.module_dict_init = module_dict_init
        
    @property
    def module_list(self):
        if not hasattr(self, '_module_list'):
            self._module_list = scan_dir_pyfile(self.root)
        return self._module_list

    @property
    def module_dict(self):
        if not hasattr(self, '_module_dict'):
            if self.module_dict_init is None:
                self._module_dict = default_model_dict_init(self.root)
            else:
                self._module_dict = self.module_dict_init()
        return self._module_dict

    def default(self) -> str:
        return self.choices()[0]

    def choices(self):
        return sorted(self.module_dict.keys())

    def enum(self):
        choices = self.choices()
        cho = {}
        for i, c in enumerate(choices):
            cho[c] = i
        return enum.Enum(self.module_category, cho)
    
    def getClass(self, name):
        if name not in self.choices():
            raise ValueError(f'No module named {name}!')
        cls = import_class(self.module_dict[name])
        return cls

    def getObj(self, name, args):
        cls = self.getClass(name)
        return init_module(cls, args)
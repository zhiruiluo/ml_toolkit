import copy
import importlib
import inspect
import os
from pathlib import Path
import enum

def import_function_or_class(module_name,method_name):
    module = importlib.import_module(f'{module_name}')
    method = getattr(module, method_name, None)
    if not method:
        module = importlib.import_module(f'{module_name}.{method_name}')
        method = getattr(module, method_name, None)
        if not method:
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
    # print(namespace)
    common_kwargs = filter_dict(class_, copy.deepcopy(vars(namespace)), {'args': namespace})
    return class_(**common_kwargs)

def init_module(class_, args):
    sign = inspect.signature(class_).parameters.values()
    sign = set([val.name for val in sign])
    kwarg_dict = {'args': args}
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return class_(**filtered_dict)

def scan_dir_pyfile(root):
    modules = os.listdir(Path(root))
    # fns = [ f[:-3] for f in modules if not f.endswith('__init__.py') and f.endswith('.py')]
    fns = []
    for mod in modules:
        if mod.endswith('__init__.py') or mod == '__pycache__':
            continue
        if mod.endswith('.py'):
            fns.append(mod[:-3])
        else:
            fns.append(mod)
    return sorted(fns)


class ModuleScannerBase():
    def __init__(self, root, module_category) -> None:
        self.root = Path(root)
        self.module_category = module_category
        self.module_list = scan_dir_pyfile(self.root)

    def default(self) -> str:
        return self.choices()[0]

    def choices(self):
        return self.module_list

    def enum(self):
        choices = self.choices()
        cho = {}
        for i, c in enumerate(choices):
            cho[c] = i
        return enum.Enum(self.module_category, cho)

    def load_class(self,name):
        path = str(self.root.joinpath(name)).replace('/','.')
        return import_function_or_class(path,name)

    def getClass(self, name):
        if name not in self.choices():
            raise ValueError(f'No module named {name}!')
        cls = self.load_class(name)
        return cls

    def getObj(self, name, args):
        cls = self.getClass(name)
        return init_module(cls, args)

    def getConfig(self, name, args):
        if name not in self.choices():
            raise ValueError(f'No module named {name}!')
        dc = copy.deepcopy(vars(args)[f'{self.module_category}Parm'])
        # path = str(self.root.joinpath(name)).replace('/','.')
        # config_cls = import_function_or_class(path,'Config')
        # print(vars(args)[f'{self.module_category}Parm'])
        # return init_class_from_namespace(config_cls, args)
        # exit()
        return dc
        
    def getConfigCls(self, name):
        if name not in self.choices():
            raise ValueError(f'No module named {name}!')
        path = str(self.root.joinpath(name)).replace('/','.')
        config_cls = import_function_or_class(path,'Config')
        return config_cls

    def getParams(self, name):
        if name not in self.choices():
            raise ValueError(f'No module named {name}!')
        path = str(self.root.joinpath(name)).replace('/','.')
        get_options = import_function_or_class(path,'get_options')
        options = get_options()
        param_keys = [o[0] for o in options]
        return param_keys, options
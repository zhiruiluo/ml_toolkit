from itertools import product
from ruamel.yaml import YAML
from argparse import Namespace
import os

def parameters_production(params):
    params_groups = []
    params_keys = []
    for k,v in params.items():
        if isinstance(v,list):
            params_groups.append(v)
        else:
            params_groups.append([v])
        params_keys.append(k)

    production_list = list(product(*params_groups))
    param_list = []
    for v in production_list:
        param_list.append({k:p for k, p in zip(params_keys, v)})
    return param_list
    
def parameters_generator(params):
    def unpack(data):
        for k, v in data.items():
            if isinstance(v, dict) and len(v) >0:
                k1 = list(v.keys())[0]
                yield {k: k1}
                if isinstance(v[k1], dict):
                    yield from unpack(v[k1])
                else:
                    yield {k: v}
                # yield from unpack(v)
            elif isinstance(v, list):
                yield {k:v}
            else:
                yield {k:v}

    def de_list(list_embedded_params):
        for p in parameters_production(list_embedded_params):
            l = {}
            for i in unpack(p):
                l.update(i)

            for n in parameters_production(l):
                list_flag = False
                dict_flag = False
                for k,v in n.items(): 
                    if isinstance(v, list):
                        list_flag = True
                        break
                    if isinstance(v, dict):
                        dict_flag = True
                        break
                if list_flag or dict_flag:
                    yield from de_list(n)
                else:
                    yield n


    yield from de_list(params)

def from_options(options):
    n = []
    param_k = []
    for k,v in options.items():
        if v == {}:
            continue
        nn = []
        param_k.append(k)
        for i in parameters_generator(v):
            nn.append(i)

        n.append(nn)
    for i in product(*n):
        yield {k: v for k,v in zip(param_k, i)}

def flatten_params(params):
    pairs = {}
    for k, v in params.items():
        pairs.update(v)
    return pairs

def argparser_from_options(options):
    for p in from_options(options):
        yield flatten_params(p)

def params_to_namespace(params):
    pairs = flatten_params(params)
    my_namespace = Namespace(**pairs)
    return my_namespace

def parameters_from_yaml(path):
    yaml = YAML()
    with open(path, 'r') as fp:
        params_dict = yaml.load(fp)
    return params_dict

def parameters_to_yaml(path, params_dict):
    yaml = YAML()
    yaml.default_flow_style = False
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path, 'w') as fp:
        yaml.dump(params_dict, fp)
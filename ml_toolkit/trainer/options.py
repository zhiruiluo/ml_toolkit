import argparse
import logging
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

from ml_toolkit.models.model_select import ModelSelection
from ml_toolkit.datasets.dataset_select import DatasetSelection

def add_ensemble_hyperparameter(parser):
    parser.add_argument('--ensemble', type=str, default=None, choices=['Voting','Bagging'])

def add_model_hyperparameter(parser, model):
    _, options = ModelSelection.getParams(model)
    for o in options:
        parser.add_argument('--'+o[0], type=o[1], default=o[2])
    
def add_dataset_parameter(parser, dataset):
    _, options = DatasetSelection.getParams(dataset)
    for o in options:
        parser.add_argument('--'+o[0], type=o[1], default=o[2])

def add_trainer_parameter(parser):
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=12,
                            help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--determ', action='store_true',
                        help='Deterministic flag')
    parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size')
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='0 <= label smoothing < 1, 0 no smoothing')

def add_experiment_parameter(parser):
    parser.add_argument('--model', type=str, default=ModelSelection.default_model(), choices=ModelSelection.choices())
    parser.add_argument('--dataset', type=str, default=DatasetSelection.default(), help='dataset choice', 
                        choices=DatasetSelection.choices())
    parser.add_argument('--exp', type=int, default=1,
                            help='Unique experiment number')
    parser.add_argument('--nfold', type=int, default=1)
    parser.add_argument('--nrepeat', type=int, default=1)

def add_results_parameter(parser):
    parser.add_argument('--fold', type=int, default=1,
                            help='The fold number for current training')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--splits', type=str, default='3:1:1')

def add_grid_parameter(parser):
    parser.add_argument('--grid_id', type=int, default=0)

def add_system_parameter(parser):
    parser.add_argument('--expname', type=str, default='testexp')
    parser.add_argument('--no_cuda', action='store_true',
                    help='Disables CUDA training.')
    parser.add_argument('--debug', action='store_true',
                            help='Debug flag')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_db', action='store_true')

def pre_build_arg():
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--dataset', type=str, default=DatasetSelection.default())
    model_parser.add_argument('--model', type=str, default=ModelSelection.default_model())
    dataset = model_parser.parse_known_args()[0].dataset
    model = model_parser.parse_known_args()[0].model
    return dataset, model

def build_arg(dataset, model):
    parser = argparse.ArgumentParser()
    add_system_parameter(parser)
    add_dataset_parameter(parser,dataset)
    add_model_hyperparameter(parser, model)
    add_ensemble_hyperparameter(parser)
    add_experiment_parameter(parser)
    add_results_parameter(parser)
    add_trainer_parameter(parser)
    add_grid_parameter(parser)
    return parser

def setup_arg():
    dataset, model = pre_build_arg()
    parser = build_arg(dataset, model)
    return parser.parse_args()

def setup_arg_fake(dataset, model, namespace):
    parser = build_arg(dataset, model)
    exist_dict = vars(namespace)
    args = []
    for k, v in exist_dict.items():
        args.append(f'--{k}')
        args.append(str(v))
    args = ' '.join(args).split()
    return parser.parse_args(args,namespace=namespace)


class Options():
    def __init__(self) -> None:
        if not OmegaConf.has_resolver('Params'):
            OmegaConf.register_new_resolver('Params', self.ParamsResolver)
        self.arg_conf = self.load_config()

    def list_to_omega(self, options):
        omega_obj = {}
        for o in options:
            omega_obj[o[0]] =  {'type':o[1].__name__,'default': o[2]}
        return omega_obj

    def ParamsResolver(self, mode, name):
        if mode == 'default':
            if name == 'model':
                return ModelSelection.default_model()
            elif name == 'dataset':
                return DatasetSelection.default()
        elif mode == 'params':
            if name == 'model':
                return self.list_to_omega(ModelSelection.getParams(self.model)[1])
            elif name == 'dataset':
                return self.list_to_omega(DatasetSelection.getParams(self.dataset)[1])

    def load_config(self):
        conf = OmegaConf.load('./src/trainer/options.yaml')
        arg_conf = {}
        for k ,args in conf.items():
            for arg in args:
                for arg_name, arg_setting in arg.items():
                    arg_conf[arg_name] = arg_setting
        return arg_conf

    def setup_arg(self,dataset,model,namespace):
        return setup_arg_fake(dataset,model,namespace)

    def update_default(self, defaults, name_dict):
        for k, v in defaults.items():
            if k in name_dict:
                continue
            else:
                if v == None:
                    name_dict[k] = None
                elif 'action' not in v:
                    if v['type'] != 'str':
                        if v['default'] is not None:
                            value = eval(f"{v['type']}({v['default']})")
                        else:
                            value = None
                    else:
                        value = v['default']
                    name_dict[k] = value
                else:
                    if v['action'] == 'store_true':
                        value = False
                    else:
                        value = True
                    name_dict[k] = value

    def setup_arg_fill_default(self,namespace):
        name_dict = vars(namespace)
        self.update_default(self.arg_conf,name_dict)

        model_params = self.list_to_omega(ModelSelection.getParams(namespace.model)[1])
        dataset_params = self.list_to_omega(DatasetSelection.getParams(namespace.dataset)[1])
        
        self.update_default({**model_params,**dataset_params}, name_dict)
        return namespace
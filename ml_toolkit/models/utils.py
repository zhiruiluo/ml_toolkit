from argparse import Namespace
from ml_toolkit.trainer.parameter_manager import HyperparameterManager
from ml_toolkit.datasets.dataset_select import DatasetSelection
from ml_toolkit.models.model_select import ModelSelection

def prepare_dataset(dataset, **kwargs):
    hpm = HyperparameterManager()
    hpm.update_values_from_namespace(Namespace(dataset=dataset, **kwargs))
    DatasetSelection.getParams(dataset)
    my_dataset = DatasetSelection.getDataset(dataset,args=hpm.args)
    return my_dataset

def prepare_model_dataset(dataset, model, **kwargs):
    hpm = HyperparameterManager()
    hpm.update_values_from_namespace(Namespace(dataset=dataset, model=model, **kwargs))
    DatasetSelection.getParams(dataset)
    my_dataset = DatasetSelection.getDataset(dataset, args=hpm.args)
    my_model = ModelSelection.getModel(model, hpm=hpm, args=hpm.args)
    return my_dataset, my_model


from torch.profiler import profile, ProfilerActivity, record_function
class log_profiler(object):
    def __init__(self, logger = None) -> None:
        if logger is None:
            from ml_toolkit.logger.get_configured_logger import get_console_logger
            logger = get_console_logger()
        
        self.logger = logger

    def __enter__(self):
        self.prof = profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True)
        self.record_f = record_function("model_inference")
        self.prof.__enter__()
        self.record_f.__enter__()
        return self.logger


    def __exit__(self, type, value, traceback):
        self.record_f.__exit__(type, value, traceback)
        self.prof.__exit__(type, value, traceback)
        self.logger.info(self.prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
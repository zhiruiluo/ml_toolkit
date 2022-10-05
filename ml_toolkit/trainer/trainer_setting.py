# import hunter
# from hunter import trace, Q
# trace(
#     Q(module_in=['pytorch_lightning.trainer.trainer','pytorch_lightning.utilities.device_parser'], kind='line', action=hunter.CodePrinter())
# )        
from dataclasses import dataclass
import logging
import os
jobid = os.environ.get('SLURM_JOB_ID')
from pathlib import Path

import time
import random
import pytorch_lightning as pl
from ml_toolkit.utils.cuda_status import get_num_gpus
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from ray_lightning import RayStrategy
# from torch.profiler import ProfilerActivity, profile, record_function

# my_logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

logger = logging.getLogger(__name__)

# @dataclass
# class TrainerConfig():
#     expname: str
#     no_cuda: bool
#     accelerator: str
#     devices: int
#     epochs: int
#     precision: int
#     auto_bs: bool
#     profiler: str
    

def setup_logger(exp_name,dir,version=None):
    pl_logger = TensorBoardLogger(
        save_dir=dir,
        name=exp_name,
        version=version,
        )
    return pl_logger

def configure_callbacks(args):
        monitor = 'val_acc_epoch'
        # monitor = 'val_acc_epoch'
        earlystop = EarlyStopping(
            monitor=monitor,
            patience=args.patience,
            mode='max'
        )
        chc_dir = Path('model_checkpoint')
        if not chc_dir.exists():
            chc_dir.mkdir(parents=True, exist_ok=True)
            
        ckp_cb = ModelCheckpoint(
            dirpath=Path('model_checkpoint').joinpath(f'{args.dataset}_{args.model}'),
            filename=args.expname + '-{epoch:02d}-{val_acc_epoch:.3f}',
            monitor=monitor,
            save_top_k=1,
            mode='max'
            )
        
        pb_cb = TQDMProgressBar(refresh_rate=1)
        
        lr_cb = LearningRateMonitor(logging_interval='step')
        
        return [earlystop, ckp_cb, lr_cb, pb_cb]

def get_trainer(args, logdir='tensorboard_logs', version=None, precision=32, fast_dev_run=False, auto_bs=None):
    
    if jobid is not None:
        expname = f'{args.expname}_jobid_{jobid}_time_{time.strftime("%m%d-%H%M", time.localtime())}'
    else:
        expname = "{}_time_{}".format(
            args.expname,
            time.strftime("%m%d-%H%M", time.localtime())
        )
    logger.info('[get_trainer] 2')
    if args.no_cuda:
        args.accelerator = 'cpu'
    else:
        if get_num_gpus() == 0:
            args.accelerator = 'cpu'
        else:
            args.accelerator = 'gpu'
    logger.info('[get_trainer] 2')
    
    callbacks = [*configure_callbacks(args)]
    params = {
        'accelerator':args.accelerator,
        'fast_dev_run':fast_dev_run,
        'precision':precision,
        'max_epochs': args.epochs,
        'auto_scale_batch_size': False if auto_bs == '' else auto_bs,
        'logger': setup_logger(expname, logdir, version),
        'callbacks': callbacks,
    }
    
    if args.profiler:
        params['profiler'] = 'pytorch'
        
    logger.info('[get_trainer] %s', params)
    trainer = pl.Trainer(**params)
    return trainer

def get_trainer_ddp(args, logdir='tensorboard_logs', version=None, precision=32, fast_dev_run=False, auto_bs=None):
    if jobid is not None:
        expname = f'{args.expname}_jobid_{jobid}_time_{time.strftime("%m%d-%H%M", time.localtime())}'
    else:
        expname = "{}_time_{}".format(
            args.expname,
            time.strftime("%m%d-%H%M", time.localtime())
        )
    logger.info('[get_trainer] 2')
    # assert args.no_cuda == False
    # assert get_num_gpus() > 0
    args.accelerator = 'gpu'
    
    callbacks = [*configure_callbacks(args)]
    params1 = {
        # 'accelerator':args.accelerator,
        'fast_dev_run':fast_dev_run,
        'precision':precision,
        'max_epochs': args.epochs,
        'auto_scale_batch_size': False if auto_bs == '' else auto_bs,
        'logger': setup_logger(expname, logdir, version),
        'callbacks': callbacks,
        'strategy': RayStrategy(num_workers=1, num_cpus_per_worker=8, use_gpu=True, resources_per_worker={'GPU': 1})
    }

    
    if args.profiler:
        params1['profiler'] = 'pytorch'
    logger.info('[get_trainer_ddp] gpus %s  %s', get_num_gpus(), params1)
    trainer = pl.Trainer(**params1)
    return trainer

def training_flow(trainer: pl.Trainer, model, dataset):
    logger.info('[start training flow]')
    # tune_result = trainer.tune(model, datamodule=dataset)
    # new_batch_size = tuner.scale_batch_size(model)
    # logger.info('[New Batch Size] %s', tune_result)
    # model.hparams.batch_size = new_batch_size
    # cpus = random.choices(list(range(os.cpu_count())), k=8)
    # os.sched_setaffinity(0, cpus)
    trainer.fit(model, datamodule=dataset)
    fit_results = trainer.logged_metrics
    
    ckp_cb = trainer.checkpoint_callback
    earlystop_cb = trainer.early_stopping_callback

    logger.info('Interrupted %s, early stopped epoch %s', trainer.interrupted, earlystop_cb.stopped_epoch)
    # test model
    if os.path.isfile(ckp_cb.best_model_path) and not trainer.interrupted:
        test_results = trainer.test(ckpt_path=ckp_cb.best_model_path, datamodule=dataset)[0]
    # else:
    #     test_results = trainer.test(model, datamodule=dataset)[0]
    
    # delete check point
    if os.path.isfile(ckp_cb.best_model_path):
        os.remove(ckp_cb.best_model_path)
    
    ## convert test_result dictionary to dictionary
    if not trainer.interrupted:
        results = {**fit_results, **test_results}
        return results
    
    return None

from __future__ import annotations
from simple_slurm import Slurm
from pathlib import Path
from ml_toolkit.utils.param_production import parameters_from_yaml
import subprocess
from typing import Union, Dict
from ml_toolkit.gridsearch.gridsearch import GridSearch
import math

def default_options(job_name='test', ntasks=1, log_root='slurm/slurm_out/', partition='epscor'):
    log_path = Path(log_root).joinpath(job_name)
    log_path.mkdir(parents=True, exist_ok=True)
    options = {
        'job_name': job_name,
        'output': log_path.joinpath('%j.out'),
        'error':  log_path.joinpath('%j.err'),
        'ntasks': ntasks,
        'gpus_per_task': 1,
        'cpus_per_task': 8,
        'mem_per_cpu': '4G',
        'partition': partition,
        'exclude': 'discovery-g[1]',
        'time': '3-24:00:00',
    }
    return options, log_path

def job_array_options(array_number=0, job_name='test', log_root='slurm/slurm_out/', ntasks=1, partition='epscor', 
        gpus_per_task=1,cpus_per_task=8, mem_per_cpu='4G', exclude='discovery-g[1]', time='3-24:00:00'):
    log_path = Path(log_root).joinpath(job_name)
    log_path.mkdir(parents=True, exist_ok=True)
    options = {
        'job_name': job_name,
        'output': log_path.joinpath('%A-%a.out'),
        'error':  log_path.joinpath('%A-%a.err'),
        'array': range(array_number),
        'ntasks': ntasks,
        'gpus_per_task': gpus_per_task,
        'cpus_per_task': cpus_per_task,
        'mem_per_cpu': mem_per_cpu,
        'partition': partition,
        'exclude': exclude,
        'time': time,
    }
    return options, log_path

def parameters_to_str(params):
    return ' '.join([f'--{k}={v}' if v!='' else f'--{k}' for k,v in params.items() if v is not None])

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Launcher():
    PY_FILE = '$pyfile'
    def __init__(self, params, job_name, partition, gpu_require, python_file_path, user, conda_env, log_root='./slurm/slurm_out', verbose=False) -> None:
        # self.params = params
        self.job_name = job_name
        self.log_root = log_root
        self.user = user
        self.conda_env = conda_env
        self.partition = partition
        self.gpu_require = gpu_require
        self.python_file_path = python_file_path
        self.verbose = verbose
        self.prepare_flag = False
        self.gridsearch = GridSearch(params)

    @staticmethod
    def from_gridconfig(grid_file: Union[Dict, str], verbose=False) -> Launcher:
        if grid_file:
            if isinstance(grid_file, str):
                grid_config = parameters_from_yaml(grid_file)
            # print(grid_config)
            grd = grid_config.pop('grid_config', None)
            arguments = dict(
                params=grid_config,
                job_name=grd['job_name'],
                partition=grd['partition'],
                python_file_path=grd['python_file_path'],
                user=grd['user'],
                conda_env=grd['conda_env'],
                log_root=grd['log_root'],
                gpu_require=grd['gpu_require'],
                verbose=verbose
                )
            return Launcher(**arguments)        

    def get_ntasks_and_njobs(self, total_tasks, max_njobs=10):
        if total_tasks < max_njobs:
            ntasks = 1
            njobs = total_tasks
            return ntasks, njobs, 1
        tasks_per_job = math.ceil(total_tasks/max_njobs)
        if self.gpu_require:
            if tasks_per_job > 10:
                ntasks = 10 
            else:
                ntasks = tasks_per_job
        else:
            ntasks = tasks_per_job // 10
        ntasks = 1 if ntasks == 0 else ntasks
        return ntasks, max_njobs, tasks_per_job

    def prepare(self, param_for_all=False):
        if param_for_all:
            params_list = self.gridsearch.get_params_list_for_all()
        else:
            params_list = self.gridsearch.get_params_list_for_unfinished()
        total_tasks = len(params_list)
        if total_tasks == 0:
            print('[Launcher] all tasks finished')
            exit()

        ntasks, njobs, tasks_per_job = self.get_ntasks_and_njobs(total_tasks)
        print ("[Launcher] {:<10} {:<10} {:<14} {:<10}".format('njobs', 'ntasks', 'tasks_per_job', 'total_tasks'))
        print ("[Launcher] {:<10} {:<10} {:<14} {:<10}".format(njobs, ntasks, tasks_per_job, total_tasks))
        
        if self.gpu_require:
            slurm_options, log_path = job_array_options(njobs,job_name=self.job_name,log_root=self.log_root,partition=self.partition,
                gpus_per_task=1, cpus_per_task=8, mem_per_cpu='4G', ntasks=ntasks
            )
        else:
            slurm_options, log_path = job_array_options(njobs,job_name=self.job_name,log_root=self.log_root,partition=self.partition,
                gpus_per_task=0, cpus_per_task=4, mem_per_cpu='2G', ntasks=ntasks
            )
        
        self.python_cmd = 'python -u'
        self.srun_cmd = f'srun -n1 -N1 --exclusive -u'
        self.scrip_dir = Path(f'./slurm/{self.job_name}_launcher')
        self.scrip_dir.mkdir(parents=True, exist_ok=True)
        self.run_cmds = self._prepare_cmds_in_seperated_file(params_list, tasks_per_job)
    
        self.log_path = log_path
        self.slurm = Slurm(**slurm_options)
        
        self.scrip_path = self.scrip_dir.joinpath('SBASH_AUTO.sh')
        with open(self.scrip_path, mode='w') as fp:
            fp.write(str(self.slurm))
            fp.write(self.run_cmds)
        
        if self.verbose:
            print(self.slurm)
            print(self.run_cmds)
        self.prepare_flag = True

    def _prepare_cmds_in_seperated_file(self, params_list, tasks_per_job):
        run_cmds = []
        run_cmds += self._prologue_cmd()

        job_total = 0
        all_jobs_in_seperated_file = {}
        for job_id, job_chunk_params in enumerate(chunks(params_list, tasks_per_job)):
            job_name = f'job_{job_id}'
            job_cmds = []
            job_cmds += [f'pyfile={self.python_file_path}']
            for grid_id, pars in job_chunk_params:
                pars.update({'grid_id':grid_id})
                param_values = parameters_to_str(pars)
                single_cmd = self._task_cmd(param_values)
                job_cmds += [single_cmd]
            job_cmds += ['wait']
            job_cmds = '\n'.join(job_cmds)
            all_jobs_in_seperated_file[job_name] = job_cmds
            execute_job_bash = f'bash {self.scrip_dir.joinpath(job_name)}.sh'
            run_cmds += [self._if_statement(job_id, execute_job_bash)]
            job_total += 1

        for fn, cmds in all_jobs_in_seperated_file.items():
            path = self.scrip_dir.joinpath(fn+'.sh')
            with open(path, 'w') as f:
                f.write(cmds)

        run_cmds += self._epilogue_cmds()
        run_cmds += [f'# jobs {job_total} tasks_per_jobs {tasks_per_job} total_tasks {len(params_list)}']
        run_cmds = '\n'.join(run_cmds)
        return run_cmds

    def _prologue_cmd(self):
        cmds = [
            'module load anaconda3',
            f'conda activate {self.conda_env}',
            #f'mkdir -p {log_path}',
            f'pyfile={self.python_file_path}',
            #f'{self.srun_out_path}',
            #f'mkdir -p $logpath',
            #f'echo $logpath',
            ''
        ]
        return cmds

    def _epilogue_cmds(self):
        cmds = [
            # 'sleep 2',
            # f'squeue -u {self.user}'
            ''
        ]
        return cmds

    @staticmethod
    def _if_statement(taskid,cmds):
        return f'if [[({Slurm.SLURM_ARRAY_TASK_ID} -eq {taskid})]]; then\n' + \
                    f'{cmds}\n' +\
                    f'fi'

    def _task_cmd(self, param_values):
        return ' '.join([self.srun_cmd,self.python_cmd, Launcher.PY_FILE, param_values]) + ' &'
    
    def _jobarray_cmd(self, array_task_id, param_values):
        single_cmd = ' '.join([self.python_cmd,Launcher.PY_FILE, param_values])
        single_cmd = self._if_statement(array_task_id,single_cmd)
        return single_cmd
    
    def _execute_squeue_(self):
        bashCommands = ['sleep 2', f'squeue -u {self.user}']
        for bash_cmd in bashCommands:
            process = subprocess.Popen(bash_cmd, shell=True,stdout=subprocess.PIPE, text=True)
            output, error = process.communicate()
            print(output)
        
    def launch(self):
        assert self.prepare_flag == True
        bashCommand = f'sbatch {self.scrip_path}'
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, text=True)
        output, error = process.communicate()
        print('[script output]',output)
        
        self._execute_squeue_()
        

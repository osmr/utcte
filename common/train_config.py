import os
from sys import platform
import argparse
import json


class TrainConfig(object):

    def __init__(self):
        self.prm = None

    def load(self,
             model,
             optimizer,
             data_source,
             task_name,
             framework_name):

        args = self._parse_args(
            model.params,
            optimizer.params,
            data_source.params,
            model.param_bounds,
            optimizer.param_bounds,
            data_source.param_bounds,
            task_name,
            framework_name)

        args_dict = vars(args)

        if args.cfg_file and os.path.exists(args.cfg_file):
            with open(args.cfg_file, 'r') as f:
                cfg_args_dict = json.load(f)
            args_dict.update(cfg_args_dict)

        args_dict = dict((k, tuple(v) if isinstance(v, list) else v) for k, v in args_dict.items())

        model.update_params(args_dict)
        optimizer.update_params(args_dict)
        data_source.update_params(args_dict)

        model.update_param_bounds(args_dict)
        optimizer.update_param_bounds(args_dict)
        data_source.update_param_bounds(args_dict)

        self.prm = args_dict

    def _parse_args(self,
                    mdl_params,
                    opt_params,
                    dat_params,
                    mdl_param_bounds,
                    opt_param_bounds,
                    dat_param_bounds,
                    task_name,
                    framework_name):
        is_win_os = "win" in platform.lower()
        parser = argparse.ArgumentParser(
            description='Example of controlable training a neural net classifier for the {} task on the base of {} framework'.format(task_name, framework_name),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        file_group = parser.add_argument_group(
            'File and directory names')
        file_group.add_argument(
            '--cfg',
            dest='cfg_file',
            help='Path to JSON-config file, which has absolute priority',
            default='',
            type=str)
        file_group.add_argument(
            '--work-dir',
            dest='work_dir',
            help='Path to work directory for temporary and resulted files',
            default='../TEMP/',
            type=str)
        file_group.add_argument(
            '--project',
            dest='project_name',
            help='Name of the training project (to store files in a separate subdir in work dir)',
            default='{}_{}'.format(task_name, framework_name.lower()),
            type=str)
        file_group.add_argument(
            '--prefix',
            dest='model_filename_prefix',
            help='Prefix for checkpoint files',
            default=task_name,
            type=str)
        file_group.add_argument(
            '--synch-list',
            dest='synch_list',
            help='List of other files with hyper parameters',
            default=[],
            nargs='*',
            type=str)
        fixed_params_group = parser.add_argument_group(
            'Fixed parameters')
        fixed_params_group.add_argument(
            '--gpus',
            dest='gpus',
            help='List of GPU device numbers to train with, empty is CPU',
            default=([] if is_win_os else [0]),
            #default=[0],
            nargs='*',
            type=int)
        fixed_params_group.add_argument(
            '--max-epoch',
            dest='max_num_epoch',
            help='Maximum number of epochs',
            default=(100 if is_win_os else 12000),
            type=int)
        fixed_params_group.add_argument(
            '--min-epoch',
            dest='min_num_epoch',
            help='Minimum number of epochs',
            default=100,
            type=int)
        fixed_params_group.add_argument(
            '--bo-iters',
            dest='bo_num_iter',
            help="Hyperparameter optimization's number of iterations",
            default=200,
            type=int)
        fixed_params_group.add_argument(
            '--bo-kappa',
            dest='bo_kappa',
            help="Hyperparameter optimization's Kappa parameter",
            default=2.576,
            type=float)
        fixed_params_group.add_argument(
            '--bo-min-rand',
            dest='bo_min_rand_num',
            help="Minimum count of random points in the hyperparam optimizing",
            default=10,
            type=int)
        fixed_params_group.add_argument(
            '--sync-period',
            dest='sync_period',
            help="Period of synchronization from other hyper-parameter files",
            default=5,
            type=int)
        mdl_variable_params_group = parser.add_argument_group(
            "Model's variable parameters")
        for mdl_params_key, mdl_params_value in mdl_params.items():
            mdl_variable_params_group.add_argument(
                '--' + mdl_params_key,
                dest=mdl_params_key,
                help=" ",
                default=mdl_params_value,
                type=type(mdl_params_value))
        for mdl_param_bound_key, mdl_param_bound_value in mdl_param_bounds.items():
            mdl_variable_params_group.add_argument(
                '--' + mdl_param_bound_key,
                dest=mdl_param_bound_key,
                help=" ",
                default=mdl_param_bound_value,
                nargs=2,
                type=type(mdl_param_bound_value[0]))
        opt_variable_params_group = parser.add_argument_group(
            "Optimizer's variable parameters")
        for opt_params_key, opt_params_value in opt_params.items():
            opt_variable_params_group.add_argument(
                '--' + opt_params_key,
                dest=opt_params_key,
                help=" ",
                default=opt_params_value,
                type=type(opt_params_value))
        for opt_param_bound_key, opt_param_bound_value in opt_param_bounds.items():
            opt_variable_params_group.add_argument(
                '--' + opt_param_bound_key,
                dest=opt_param_bound_key,
                help=" ",
                default=opt_param_bound_value,
                nargs=2,
                type=type(opt_param_bound_value[0]))
        dat_variable_params_group = parser.add_argument_group(
            "Data source's variable parameters")
        for dat_params_key, dat_params_value in dat_params.items():
            dat_variable_params_group.add_argument(
                '--' + dat_params_key,
                dest=dat_params_key,
                help=" ",
                default=dat_params_value,
                type=type(dat_params_value))
        for dat_param_bound_key, dat_param_bound_value in dat_param_bounds.items():
            dat_variable_params_group.add_argument(
                '--' + dat_param_bound_key,
                dest=dat_param_bound_key,
                help=" ",
                default=dat_param_bound_value,
                nargs=2,
                type=type(dat_param_bound_value[0]))
        args = parser.parse_args()
        return args

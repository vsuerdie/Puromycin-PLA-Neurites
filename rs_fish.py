import os
import subprocess
import logging
from utils import set_logger, get_args


def norm_all_paths(path_macro, path_imagej, path_output_log): # path_rs_param_log
    path_macro = os.path.normpath(path_macro)
    # path_rs_param_log = os.path.normpath(path_rs_log)
    path_imagej = os.path.normpath(path_imagej)
    path_output_log = os.path.normpath(path_output_log)

    logging.info(f'Path ImageJ macro: {path_macro}')
    # logging.info(f'Path log file with input parameters: {path_rs_param_log}')
    logging.info(f'Path ImageJ: {path_imagej}')
    logging.info(f'Path output log: {path_output_log}')


def main():

    set_logger()
    logging.info('logger set')

    arg_names = ['path_rs_fish_macro', 'path_imagej', 'path_output_log']
    args = get_args(arg_names)
    path_macro = args['path_rs_fish_macro']
    path_imagej = args['path_imagej']
    path_output_log = args['path_output_log']
    norm_all_paths(path_macro, path_imagej, path_output_log)

    cmd = f'{path_imagej} --headless --run {path_macro} > {path_output_log}'
    os.system(cmd)
    logging.info(f'Command executed: {cmd}')


if __name__ == "__main__":
    main()
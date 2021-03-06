#!/usr/bin/env python3
import os


def traversal_logic(indices):
    results = []
    intercept = ","
    keys = indices.keys()

    for key in keys:
        print(key + intercept, end='')
    print(list(keys))
    recursive_in(list(keys))


def recursive_in(prefix_list, line=""):
    if len(prefix_list) == 1:
        print(line + prefix_list[0])
    else:
        print(prefix_list[0])
        recursive_in(prefix_list[1:], line + prefix_list[0])


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def get_log_dirs(prefix="."):
    result_dirs = []
    for root, dirs, files in os.walk(prefix, topdown=False):
        for dir in dirs:
            if "MB" in dir:
                result_dirs.append(os.path.join(root, dir))
    return result_dirs


def get_log_and_std_files(prefix=".", with_stat_csv=False):
    LOG_FILES = []
    stdout_files = []
    qps_files = []
    stat_csvs = []
    for root, dirs, files in os.walk(prefix, topdown=False):
        for filename in files:
            if "stdout" in filename:
                stdout_files.append(os.path.join(root, filename))
            if "LOG" in filename:
                LOG_FILES.append(os.path.join(root, filename))
            if "report.csv" in filename:
                qps_files.append(os.path.join(root, filename))
            if "stat_result.csv" in filename:
                stat_csvs.append(os.path.join(root, filename))
    if with_stat_csv:
        return stdout_files[0], LOG_FILES[0], qps_files[0], stat_csvs[0]
    else:
        return stdout_files[0], LOG_FILES[0], qps_files[0]

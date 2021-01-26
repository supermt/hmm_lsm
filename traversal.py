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


def get_log_dirs(prefix="."):
    result_dirs = []
    for root, dirs, files in os.walk(prefix, topdown=False):
        for dir in dirs:
            if "MB" in dir:
                result_dirs.append(os.path.join(root, dir))
    return result_dirs


def get_log_and_std_files(prefix="."):
    LOG_FILES = []
    stdout_files = []
    qps_files = []
    for root, dirs, files in os.walk(prefix, topdown=False):
        for filename in files:
            if "stdout" in filename:
                stdout_files.append(os.path.join(root, filename))
            if "LOG" in filename:
                LOG_FILES.append(os.path.join(root, filename))
            if "report.csv" in filename:
                qps_files.append(os.path.join(root, filename))
    return stdout_files[0], LOG_FILES[0],qps_files[0]

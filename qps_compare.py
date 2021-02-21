# here is an example from online-ml/river

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from log_class import log_recorder
from traversal import get_log_and_std_files, mkdir_p
from traversal import get_log_dirs


def load_log_and_qps(log_file, ground_truth_csv):
    # load the data
    return log_recorder(log_file, ground_truth_csv)


def data_cleaning_by_max_MBPS(bucket_df, MAX_READ=2000, MAX_WRITE=1500):
    read = bucket_df["read"]
    bad_read = read >= MAX_READ
    read[bad_read] = MAX_READ
    write = bucket_df["write"]
    bad_write = write >= MAX_WRITE
    write[bad_write] = MAX_WRITE
    return bucket_df


def read_report_csv_with_change_points(report_file):
    file_lines = open(report_file).readlines()
    result = []
    print(len(file_lines))
    for file_line in file_lines[1:]:
        records = file_line.replace("\n", "").split(",")
        if (len(records) > 2):
            normed = int(records[3]) / (64 * 1024 * 1024)
            records = records[0:2]
            records.append(normed)
        else:
            records.append(1)
        result.append(records)
    result = np.array(result)
    return pd.DataFrame(result, columns=["secs_elapsed", "qps", "change_points"])


if __name__ == '__main__':

    log_dir_prefix = "log_files/"
    dirs = get_log_dirs(log_dir_prefix)
    for log_dir in dirs:
        if "64MB" in log_dir:
            stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
            report_df = read_report_csv_with_change_points(report_csv)
            # report_df.plot(subplots=True)
            report_df.to_csv("tmp.csv")
            report_df = pd.read_csv("tmp.csv")
            print(report_df)
            report_df.plot(subplots=True)

            # plt.scatter(report_df["secs_elapsed"], report_df["qps"])
            output_path = "DOTA_Result_compare/%s/" % log_dir.replace(log_dir_prefix, "").replace("/", "_")
            mkdir_p(output_path)
            plt.savefig("{}/qps_origin.pdf".format(output_path), bbox_inches="tight")
            plt.savefig("{}/qps_origin.png".format(output_path), bbox_inches="tight")

            plt.close()

    log_dir_prefix = "DOTA_embedded/"
    dirs = get_log_dirs(log_dir_prefix)
    for log_dir in dirs:
        if "64MB" in log_dir:
            stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
            report_df = read_report_csv_with_change_points(report_csv)
            # report_df.plot(subplots=True)
            report_df.to_csv("tmp.csv")
            report_df = pd.read_csv("tmp.csv")
            print(report_df)
            report_df.plot(subplots=True)

            # plt.scatter(report_df["secs_elapsed"], report_df["qps"])
            output_path = "DOTA_Result_compare/%s/" % log_dir.replace(log_dir_prefix, "").replace("/", "_")
            mkdir_p(output_path)
            plt.savefig("{}/qps_with_DOTA.pdf".format(output_path), bbox_inches="tight")
            plt.savefig("{}/qps_with_DOTA.png".format(output_path), bbox_inches="tight")
            plt.close()

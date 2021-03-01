# here is an example from online-ml/river

import matplotlib.pyplot as plt

from feature_selection import read_report_csv_with_change_points
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


def plot_qps(dirs, log_prefix, output_prefix, plot_features, fig_name, condition=""):
    for log_dir in dirs:
        if condition in log_dir:
            print(log_dir)
            stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
            report_df = read_report_csv_with_change_points(report_csv)
            print(len(report_df))
            plt.subplot(211)
            plt.plot(report_df["secs_elapsed"], report_df["interval_qps"], color="r")
            plt.ylim(0, 600000)
            plt.subplot(212)
            plt.plot(report_df["secs_elapsed"], report_df["change_points"], color="g")
            plt.ylim(0, 16)
            # report_df[plot_features].plot(subplots=True)
            output_path = output_prefix + "/%s/" % log_dir.replace(log_dir_prefix, "").replace("/", "_")
            mkdir_p(output_path)
            plt.savefig("{}/{}.pdf".format(output_path, fig_name), bbox_inches="tight")
            plt.savefig("{}/{}.png".format(output_path, fig_name), bbox_inches="tight")
            plt.clf()


if __name__ == '__main__':
    log_dir_prefix = "log_files/"
    plot_features = ["interval_qps", "change_points"]
    plot_qps(get_log_dirs(log_dir_prefix), log_dir_prefix, "DOTA_Result_compare", plot_features, "QPS_origin", "64MB")
    log_dir_prefix = "DOTA_embedded/"
    plot_qps(get_log_dirs(log_dir_prefix), log_dir_prefix, "DOTA_Result_compare", plot_features, "QPS_change_point", "")

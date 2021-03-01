# here is an example from online-ml/river

import matplotlib.pyplot as plt

from feature_selection import read_report_csv_with_change_points, read_stat_csv
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


def plot_stat(dirs, log_prefix, output_prefix, fig_name, condition=""):
    for log_dir in dirs:
        if condition in log_dir:
            print(log_dir)
            stdout_file, LOG_file, report_csv, stat_csv = get_log_and_std_files(log_dir, with_stat_csv=True)

            report_df = read_report_csv_with_change_points(report_csv)
            stat_df = read_stat_csv(stat_csv)
            plt.subplot(411)
            plt.plot(report_df["secs_elapsed"], report_df["interval_qps"], color="r")
            plt.ylabel("qps")
            plt.ylim(0, 600000)

            plt.subplot(412)
            plt.plot(stat_df["secs_elapsed"], stat_df["cpu_utils"], color="b")
            plt.ylabel("cpu_utils")
            plt.plot()
            plt.ylim(0, 1200)

            plt.subplot(413)
            plt.plot(stat_df["secs_elapsed"], stat_df["disk_usage"], color="c")
            # plt.plot(stat_df["secs_elapsed"], [2e7 for x in stat_df["secs_elapsed"]], color="r")
            plt.ylabel("disk_utils")
            plt.hlines(1e7, 0, stat_df["secs_elapsed"].tolist()[-1], colors="r", linestyles="dashed")
            plt.hlines(2e7, 0, stat_df["secs_elapsed"].tolist()[-1], colors="g", linestyles="dashed")
            plt.hlines(3e7, 0, stat_df["secs_elapsed"].tolist()[-1], colors="b", linestyles="dashed")

            plt.plot()

            plt.subplot(414)
            plt.plot(report_df["secs_elapsed"], report_df["change_points"], color="g")
            plt.ylabel(r"SST Size")
            plt.ylim(0, 16)

            plt.tight_layout()
            # report_df[plot_features].plot(subplots=True)
            output_path = output_prefix + "/%s/" % log_dir.replace(log_prefix, "").replace("/", "_")
            mkdir_p(output_path)
            plt.savefig("{}/{}.pdf".format(output_path, fig_name), bbox_inches="tight")
            plt.savefig("{}/{}.png".format(output_path, fig_name), bbox_inches="tight")
            plt.clf()


if __name__ == '__main__':
    log_dir_prefix = "log_files/"
    plot_features = ["interval_qps", "change_points"]
    plot_stat(get_log_dirs(log_dir_prefix), log_dir_prefix, "CPU_compare", "QPS_origin", "12CPU")
    log_dir_prefix = "DOTA_embedded/"
    plot_stat(get_log_dirs(log_dir_prefix), log_dir_prefix, "CPU_compare", "QPS_change_point",
              "12CPU")

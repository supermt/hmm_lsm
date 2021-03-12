# here is an example from online-ml/river

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy

from feature_selection import generate_lsm_shape, combine_vector_with_qps, vectorize_by_compaction_output_level
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


def plot_lsm(lsm_shape, plot_level, fig, axes):
    axes[0, 1].set_title("Compaction Count")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(plot_level):
        axes[i, 0].plot(lsm_shape["time_micro"], lsm_shape["level" + str(i)], c=colors[i])
        axes[i, 0].set_ylabel("level" + str(i))

    return axes


def plot_compaction(compaction_df, plot_level, fig, axes):
    axes[0, 0].set_title("Level File Count")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(plot_level):
        axes[i, 1].plot(compaction_df["level" + str(i)], c=colors[i])
        # axes[i, 0].set_ylabel("level" + str(i))

    return axes


if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    log_dir_prefix = "fillrandom_pri_L1_Deep_L0/"
    dirs = get_log_dirs(log_dir_prefix)
    for log_dir in dirs:
        print(log_dir)
        stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
        data_set = load_log_and_qps(LOG_file, report_csv)
        lsm_shape = generate_lsm_shape(data_set)
        plot_level = 5
        compaction_df = vectorize_by_compaction_output_level(data_set, plot_level)

        fig, axes = plt.subplots(plot_level + 1, 2, sharex='all')

        plot_lsm(lsm_shape, plot_level, fig, axes)
        plot_compaction(compaction_df, plot_level, fig, axes)

        axes[plot_level, 0].plot(data_set.qps_df["secs_elapsed"], data_set.qps_df["interval_qps"])
        axes[plot_level, 0].set_ylabel("IOPS")
        axes[plot_level, 1].plot(compaction_df["write"])
        axes[plot_level, 1].set_ylabel("MBPS")

        output_path = "thread_pri/L1_Deep_L0/%s/" % log_dir.replace(log_dir_prefix, "").replace("/", "_")
        mkdir_p(output_path)
        plt.tight_layout()
        plt.savefig("{}/lsm_shape.pdf".format(output_path), bbox_inches="tight")
        plt.savefig("{}/lsm_shape.png".format(output_path), bbox_inches="tight")
        plt.close()

    log_dir_prefix = "fillrandom_only_thread"
    dirs = get_log_dirs(log_dir_prefix)
    for log_dir in dirs:
        print(log_dir)
        stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
        data_set = load_log_and_qps(LOG_file, report_csv)
        lsm_shape = generate_lsm_shape(data_set)
        plot_level = 5
        compaction_df = vectorize_by_compaction_output_level(data_set, plot_level)

        fig, axes = plt.subplots(plot_level + 1, 2, sharex='all')

        plot_lsm(lsm_shape, plot_level, fig, axes)
        plot_compaction(compaction_df, plot_level, fig, axes)

        axes[plot_level, 0].plot(data_set.qps_df["secs_elapsed"], data_set.qps_df["interval_qps"])
        axes[plot_level, 0].set_ylabel("IOPS")
        axes[plot_level, 1].plot(compaction_df["write"])
        axes[plot_level, 1].set_ylabel("MBPS")

        output_path = "thread_pri/origin/%s/" % log_dir.replace(log_dir_prefix, "").replace("/", "_")
        mkdir_p(output_path)
        plt.tight_layout()
        plt.savefig("{}/lsm_shape.pdf".format(output_path), bbox_inches="tight")
        plt.savefig("{}/lsm_shape.png".format(output_path), bbox_inches="tight")
        plt.close()

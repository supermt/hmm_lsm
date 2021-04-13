# here is an example from online-ml/river

import matplotlib as mpl
import matplotlib.pyplot as plt

from feature_selection import *
from log_class import log_recorder
from traversal import get_log_and_std_files, mkdir_p
from traversal import get_log_dirs


def load_log_and_qps(log_file, ground_truth_csv):
    # load the data
    return log_recorder(log_file, ground_truth_csv)


def plot_level_files(compaction_df, plot_level, fig, axes, col=0):
    axes[0, col].set_title("Level File Count")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(plot_level):
        axes[i, col].plot(compaction_df["level" + str(i)], c=colors[i])
        # axes[i, 0].set_ylabel("level" + str(i))

    return axes


def plot_write_per_level(compaction_df, plot_level, fig, axes, col=1):
    axes[0, col].set_title("Write In Each Level")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(plot_level):
        axes[i, col].plot(compaction_df["w_level" + str(i)], c=colors[i])
        # axes[i, 0].set_ylabel("level" + str(i))

    return axes


def plot_read_per_level(compaction_df, plot_level, fig, axes, col=2):
    axes[0, col].set_title("Read in each Level")
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(plot_level):
        axes[i, col].plot(compaction_df["r_level" + str(i)], c=colors[i])
        # axes[i, 0].set_ylabel("level" + str(i))

    return axes


if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    log_dir_prefix = "fillrandom_universal_compaction/"
    dirs = get_log_dirs(log_dir_prefix)
    for log_dir in dirs:
        print(log_dir)
        stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
        data_set = load_log_and_qps(LOG_file, report_csv)
        lsm_shape = generate_lsm_shape(data_set)
        plot_level = 4
        compaction_df = vectorize_by_compaction_output_lvl_and_speed_lvl(data_set, plot_level)

        fig, axes = plt.subplots(plot_level + 1, 3, sharex='all')

        plot_level_files(compaction_df, plot_level, fig, axes)
        plot_write_per_level(compaction_df, plot_level, fig, axes)
        plot_read_per_level(compaction_df, plot_level, fig, axes)

        axes[plot_level, 0].plot(data_set.qps_df["secs_elapsed"], data_set.qps_df["interval_qps"])
        axes[plot_level, 0].set_ylabel("IOPS")

        axes[plot_level, 1].plot(compaction_df["write"])
        axes[plot_level, 1].set_ylabel("MBPS")

        axes[plot_level, 2].plot(compaction_df["read"])
        axes[plot_level, 2].set_ylabel("MBPS")

        output_path = "compaction_style/universal/%s/" % log_dir.replace(log_dir_prefix, "").replace("/", "_")
        mkdir_p(output_path)
        plt.tight_layout()
        plt.savefig("{}/read_write_level.pdf".format(output_path), bbox_inches="tight")
        plt.savefig("{}/read_write_level.png".format(output_path), bbox_inches="tight")
        plt.close()

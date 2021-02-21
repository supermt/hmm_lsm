# here is an example from online-ml/river

import matplotlib as mpl
import matplotlib.pyplot as plt

from feature_selection import vectorize_by_compaction_output_level
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


if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    log_dir_prefix = "DOTA_embedded/"
    dirs = get_log_dirs(log_dir_prefix)
    for log_dir in dirs:
        print(log_dir)
        stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
        data_set = load_log_and_qps(LOG_file, report_csv)
        bucket_df = vectorize_by_compaction_output_level(data_set)
        bucket_df["qps"] = data_set.qps_df["interval_qps"]
        # bucket_df = data_cleaning_by_max_MBPS(bucket_df)
        #
        fig = bucket_df.plot(subplots=True)
        output_path = "image/%s/" % log_dir.replace(log_dir_prefix, "").replace("/", "_")
        mkdir_p(output_path)
        plt.savefig("{}/compaction_distribution_by_level.pdf".format(output_path), bbox_inches="tight")
        plt.close()
    # start_time = datetime.now()
    # from feature_selection import vectorize_by_compaction_output_level
    # from traversal import get_log_dirs, get_log_and_std_files
    #
    # log_prefix_dir = "log_files"
    # dirs = get_log_dirs(log_prefix_dir)
    #
    #
    # log_dir = dirs[0]
    # stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
    #
    # data_set = load_log_and_qps(LOG_file, report_csv)
    # bucket_df = vectorize_by_compaction_output_level(data_set)
    # bucket_df["qps"] = data_set.qps_df["interval_qps"]
    # end_time = datetime.now()
    # print(end_time-start_time)

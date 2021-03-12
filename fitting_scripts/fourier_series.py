from feature_selection import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from main import load_log_and_qps
from traversal import get_log_dirs, get_log_and_std_files, mkdir_p

tau = 0.045


def fourier(x, *a):
    ret = a[0] * np.cos(np.pi / tau * x)
    for deg in range(1, len(a)):
        ret += a[deg] * np.cos((deg + 1) * np.pi / tau * x)
    return ret


if __name__ == '__main__':
    log_dir_prefix = "../fillrandom_pri_L1_Deep_L0/"
    dirs = get_log_dirs(log_dir_prefix)
    log_dir = dirs[8]
    print(log_dir)
    stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
    data_set = load_log_and_qps(LOG_file, report_csv)
    lsm_shape = generate_lsm_shape(data_set)
    plot_level = 5
    compaction_df = vectorize_by_compaction_output_level(data_set, plot_level)

    print(compaction_df)
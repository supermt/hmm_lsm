import matplotlib.pyplot as plt
import pandas as pd


def plot_level_extents(subplot, plot_data):
    plot_data["level"].astype("int")
    subplot.plot(plot_data["file_num"],plot_data["extents"],"r+")
    subplot.set_ylim(0,max(plot_data["extents"]+1))
    # ax2 = subplot.twinx()
    # ax2.scatter(plot_data["file_num"],plot_data["size_byte"])
    pass


if __name__ == "__main__":
    data = pd.read_csv("extents_file.csv")
    data.groupby("level")
    levels = len(data["level"].unique())

    fig, axes = plt.subplots(levels)
    for lvl in range(levels):
        plot_level_extents(axes[lvl], data[data.level == lvl])
    fig.tight_layout()
    fig.show()
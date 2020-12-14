import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_na(data):
    sns.set_style("darkgrid")
    nas = [x for x in data.columns]
    if len(nas) == 1:
        plt.legend().set_visible(False)

    for na in nas:
        if len(nas) >= 1:
            sns.lineplot(data=data[na].dropna().reset_index(), x="index", y=na,
                         drawstyle="steps-post", label=na)
    plt.ylabel("Cumulative Hazard Rate")
    plt.xlabel("Timeline")
    plt.xlim(0, np.max(data.reset_index()["timeline"]))
    plt.title("Cumulative Hazard Rate")

    return plt

import matplotlib.pyplot as plt
import seaborn as sns


def plot_km(data):
    sns.set_style("darkgrid")
    kms = [x for x in data.columns]
    if len(kms) == 1:
        plt.legend().set_visible(False)

    for km in kms:
        if len(kms) >= 1:
            sns.lineplot(data=data[km].dropna().reset_index(), x="index", y=km,
                         drawstyle="steps-post", label=km)

    plt.ylabel("Survival Curve")
    plt.xlabel("Timeline")
    plt.title("Survival Function")

    return plt

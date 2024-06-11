import seaborn as sns
import matplotlib.pyplot as plt 
import operator as op

def groups_box_plot(plot_data):
    print(plot_data)
    sorted_keys, sorted_vals = zip(*sorted(plot_data.items(), key=op.itemgetter(0)))
    print(sorted_keys)


    sns.set(context="notebook", style='whitegrid')
    sns.utils.axlabel(xlabel="Groups", ylabel="KL-metric per Masked Sentence")
    sns.boxplot(
            data=sorted_vals, width=0.10
            )
    # sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=0.9)
    plt.xticks(plt.xticks()[0], sorted_keys)
    plt.show()

if __name__ == "__main__":
    plot_data = {
            'group1': [v for v in range(10,16)],
            'group2': [v for v in range(5,25)],
            'group3': [v for v in range(15,35)],
    }
    groups_box_plot(plot_data)

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import operator as op
from statistics import median

def calc(data):
    if len(data) > 1:
        median(data)
def groups_box_plot(plot_data):
    median_per_group = { cefr: calc(data) for cefr,data in plot_data.items()}
    print(plot_data)
    sorted_keys, sorted_vals = zip(*sorted(plot_data.items(), key=op.itemgetter(0)))
    print(sorted_keys)


    sns.set(context="notebook", style='whitegrid')
    sns.utils.axlabel(xlabel="Groups", ylabel="KL-metric per Masked Token in a  Sentence")
    sns.boxplot(
            data=sorted_vals, width=0.10
            )
    # sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=0.9)
    print(f"medians per group", median_per_group)
    plt.xticks(plt.xticks()[0], sorted_keys)
    plt.show()

def sorted_histogram(sorted_freq_tpls):
    '''
    data = [('a', 155), ('c', 73), ('b', 19), ('e', 260), ('d', 73), ('g', 42), ('f',47), ('i', 175), ('h', 77), ('k', 7), ('j', 2), ('m', 76), ('l', 63), ('o', 174), ('n', 145), ('q', 3), ('p', 61), ('s', 153), ('r', 143), ('u', 50), ('t', 193), ('w', 19), ('v', 21), ('y', 55), ('x', 4), ('z', 4)]
    '''
    x, y = zip(*sorted_freq_tpls)
    xlocs = np.arange(len(x))

    fig = plt.figure()
    ax = fig.gca()
    ax.bar(xlocs + 0.6, y)

    ax.set_xticks(xlocs + 0.5)
    ax.set_xticklabels(x)
    ax.set_xlim(0.0, xlocs.max() + 2)

    plt.show()


if __name__ == "__main__":
    '''
    plot_data = {
            'group1': [v for v in range(10,16)],
            'group2': [v for v in range(5,25)],
            'group3': [v for v in range(15,35)],
    }
    groups_box_plot(plot_data)
    '''

    data = [('a', 155), ('c', 73), ('b', 19), ('e', 260), ('d', 73), ('g', 42), ('f',47), ('i', 175), ('h', 77), ('k', 7), ('j', 2), ('m', 76), ('l', 63), ('o', 174), ('n', 145), ('q', 3), ('p', 61), ('s', 153), ('r', 143), ('u', 50), ('t', 193), ('w', 19), ('v', 21), ('y', 55), ('x', 4), ('z', 4)]
    sorted_data = sorted(data,key=lambda tpl: tpl[1],reverse=True)

    sorted_histogram(sorted_data)

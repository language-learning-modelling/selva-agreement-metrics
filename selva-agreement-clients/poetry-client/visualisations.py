import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import operator as op
from statistics import median

def calc(data):
    if len(data) > 1:
        return median(data)
    else:
        return None

def groups_box_plot(plot_data):
    #median_per_group = { cefr: calc(data) for cefr,data in plot_data.items()}
    print(plot_data)
    sorted_keys, sorted_vals = zip(*sorted(plot_data.items(), key=op.itemgetter(0)))
    median_per_group = [round(calc(data),2) for data in sorted_vals]
    print(sorted_keys)


    sns.set(context="notebook", style='whitegrid')
    sns.utils.axlabel(xlabel="Groups", ylabel="KL metric per Masked Token")
    box_plot = sns.boxplot(
            data=sorted_vals, width=0.10,
            showfliers=False,
            showcaps=False,
            whiskerprops={'visible':False}
            )
    for xtick in box_plot.get_xticks():
        box_plot.text(xtick,median_per_group[xtick] + 0.08 ,median_per_group[xtick], 
                horizontalalignment='center',size='x-small',color='black',weight='semibold')
    print(f"medians per group", median_per_group)
    plt.xticks(plt.xticks()[0], sorted_keys)
    plt.ylim(0,2.5)
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

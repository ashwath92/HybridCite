""" Positions of citation markers in sentences, relatve to where in doc
"""

import json
import os
import re
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def ploteval(fn, multiple):
    pretty_names = {
        'bow': 'BoW',
        'np': 'NP',
        'pp': 'Claim',
        'pp+bow': 'Claim+BoW',
        'npmarker': 'NPmarker',
        'fos': 'FoS',
        'bow+fosboost': 'BoW+fosboost'
        }

    with open(fn) as f:
        results = json.load(f)

    ttl, ext = os.path.splitext(fn)

    x = np.arange(1, 11)
    y_ndcg = []
    y_map = []
    y_mrr = []
    y_recall = []
    model_names = []
    num_tested = 0
    for i, model in enumerate(results['models']):
        num_tested = max(num_tested, model['num_cur'])
        model_names.append(pretty_names.get(model['name'], model['name']))
        if 'ndcg_results' not in model:
            for j in range(10):
                model['ndcg_results'] = [sm/model['num_cur'] for sm in model['ndcg_sums']]
                model['map_results'] = [sm/model['num_cur'] for sm in model['map_sums']]
                model['mrr_results'] = [sm/model['num_cur'] for sm in model['mrr_sums']]
                model['recall_results'] = [sm/model['num_cur'] for sm in model['recall_sums']]
        y_ndcg.append(model['ndcg_results'])
        y_map.append(model['map_results'])
        y_mrr.append(model['mrr_results'])
        y_recall.append(model['recall_results'])

    # print('@5\t\tNDCG\tMAP\tMRR\tRecall')
    # for i in range(4):
    #     print('{}\t\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
    #         model_names[i], y_ndcg[i][5], y_map[i][5],
    #         y_mrr[i][5], y_recall[i][5]
    #         ))
    # sys.exit()

    # plt.xticks(np.arange(min(x), max(x), 1.0))

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    subplots = [ax1, ax2, ax3, ax4]
    markers = ['.', '+', '*', '1', '2', '3', '4']

    for i in range(len(subplots)):
        y = [y_ndcg, y_map, y_mrr, y_recall][i]
        subplots[i].set_title(['NDCG','MAP','MRR','Recall'][i])
        # visual hot fix
        # if i == 1:
        #     subplots[i].yaxis.set_tick_params(labelsize=8)
        label_lines = []
        for j in range(len(model_names)):
            color = list(mpl.rcParams['axes.prop_cycle'])[j]['color']
            r = int(color[1:3], 16)/255
            g = int(color[3:5], 16)/255
            b = int(color[5:7], 16)/255
            line = subplots[i].plot(
                x, y[j], label=model_names[j], linestyle='--', marker=markers[j]
                )
            line[0].set_markerfacecolor(color)
            line[0].set_color((r, g, b, .5))
            label_lines.extend(line)
            plt.setp(subplots[i], xticks=x)
            subplots[i].xaxis.grid(True)
            subplots[i].yaxis.grid(True)

    plt.figlegend(label_lines, model_names, loc='lower center', ncol=50,
        labelspacing=1)
    f.subplots_adjust(hspace=0.35)
    plt.gcf().subplots_adjust(bottom=0.15)
    if multiple:
        plt.gcf().suptitle('{}'.format(num_tested))  # , fontsize=14
        plt.savefig('{}.png'.format(ttl))
        plt.close()
    else:
        plt.show()


if len(sys.argv) < 2:
    print('usage: python3 metric_plots.py </path/to/eval/file[s]>')
    sys.exit()

multiple = len(sys.argv[1:]) > 1
for fn in sys.argv[1:]:
    ploteval(fn, multiple)

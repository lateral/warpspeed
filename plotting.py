import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_percentile_ranks(prs):
    """
    Given the result of a call to FM.percentile_ranks() `prs`, return a
    matplotlib figure plotting the histogram of the percentile ranks.
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 1, 100)
    prs = [pr.pr for pr in prs]
    ax.hist(prs, bins=bins)
    mpr = np.mean(prs)
    plt.title('Percentile ranks\nnumber=%i, mean=%.5f' % (len(prs), mpr))
    return fig


def plot_recommendation_rates(recommendations, seen_counts, N=50):
    """
    Given the result of a call to FM.recommendations() `recommendations` and a
    dictionary `seen_counts` mapping item idxs to the number of times they were
    seen in the training interactions, return a matplotlib scatterplot plotting
    the recommendation rate vs the occurrence rate for the N most frequently
    occurring (in green) and the N most frequently recommended items (in red).
    """
    occurrences = pd.Series(seen_counts) * 1.
    occurrence_rates = occurrences / occurrences.sum()

    rec_counts = defaultdict(lambda: 0.)
    for rec in recommendations:
        rec_counts[rec.item_idx] += 1
    rec_counts = pd.Series(rec_counts)
    rec_rates = rec_counts / rec_counts.sum()
    rec_rates.loc[occurrences.index].fillna(0.)

    most_seen = occurrence_rates.nlargest(N).index
    most_recommended = rec_rates.nlargest(N).index

    plot_params = dict(s=15, marker='x')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.scatter(occurrence_rates.loc[most_seen],
               rec_rates.loc[most_seen],
               c='g', **plot_params)

    ax.scatter(occurrence_rates.loc[most_recommended],
               rec_rates.loc[most_recommended],
               c='r', **plot_params)

    ax.set_xlabel('occurrence rate')
    ax.set_ylabel('recommendation rate')

    lim = max(occurrence_rates.max(), rec_rates.max()) * 1.1
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid()

    plt.title('green: the %i most freq. occurring;\n'
              'red: the %i most frequently recommended' % (N, N))

    return fig

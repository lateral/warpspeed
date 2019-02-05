# -*- coding: utf-8 -*-
import unittest
from evaluation import PercentileRank, Recommendation
from plotting import plot_percentile_ranks, plot_recommendation_rates

class TestPlotting(unittest.TestCase):

    def test_plot_percentile_ranks(self):
        prs = [PercentileRank(0, 1, 0.1),
               PercentileRank(1, 0, 0.3),
               PercentileRank(1, 1, 0.0)]
        fig = plot_percentile_ranks(prs)
        fig.savefig('percentile_ranks.png')

    def test_plot_recommendation_rates(self):
        recs = [Recommendation(0, 1, 0),
                Recommendation(1, 1, 4),
                Recommendation(11, 31, 14)]
        seen_counts = {1: 23, 31: 0, 16: 16}
        fig = plot_recommendation_rates(recs, seen_counts, N=3)
        fig.savefig('recommendation_rates.png')

"""
For evaluation and diagnostics:
Recommendation: `rank` is the (integral, 0-offset) rank of `item_idx` as a recommendation to `user_idx`.
PercentileRank: `pr` is the "percentile rank", i.e. a float between 0 and 1
giving the proportional rank of the `item_idx` as a recommendation to
`user_idx` (so 0 is the
best possible).
"""
class PercentileRank(object):

    def __init__(self, user_idx, item_idx, pr):
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.pr = pr


class Recommendation(object):

    def __init__(self, user_idx, item_idx, rank):
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.rank = rank

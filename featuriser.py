import re
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

from patterns import TOKEN_PATTERN, ID_PATTERN
from csr_featuriser import CSRFeaturiser


class TextAnalyser():

    def __init__(self):
        self.splitter = re.compile(TOKEN_PATTERN).findall

    def __call__(self, text):
        return self.splitter(text.lower())

class TagAnalyser():
    def __init__(self):
        self.splitter = re.compile(ID_PATTERN).findall
    def __call__(self, text):
        return self.splitter(text)


class Featuriser(CSRFeaturiser):
    """
    Featuriser for Recommender.
    Optionally performs tf-idf feature weighting.
    Transformed vectors are returned L1 normalised.
    """

    def __init__(self, analyzer, min_document_frequency, min_interaction_count,
                 feature_idf, dtype=np.float32):
        """
        Words that appear in less than `min_document_frequency` documents
        are discarded as well as those with document frequency times
        document `seen_count` smaller than `min_interaction_count`.
        """
        super().__init__(analyzer, vocab=None, dtype=dtype, binary=False)
        self.min_interaction_count = min_interaction_count
        self.min_document_frequency = min_document_frequency
        self.feature_idf = feature_idf

    def fit(self, id_document_iter, interaction_counts):
        """
        Build a vocabulary of unigrams by iterating over the (id, text) pairs
        of `id_document_iter`, keeping those unigrams that appear in at least
        `self.min_document_frequency` documents, the sum of whose interactions
        (as specified by the dict `interaction_counts`) is at least
        `self.min_interaction_count`.
        Fit also the IDF feature weights, if `self.feature_idf`.
        """
        vocab_df = defaultdict(lambda: 0)   # map word -> document frequency
        vocab_ic = defaultdict(lambda: 0)   # map word -> interaction count
        num_docs = 0

        for doc_id, doc in id_document_iter:
            doc_vocab = set()
            for word in self.analyzer(doc):
                doc_vocab.add(word)
            for word in doc_vocab:
                vocab_df[word] += 1
                vocab_ic[word] += interaction_counts[doc_id]
            num_docs += 1

        vocab_ic_filtered = [(word, ic) for word, ic in vocab_ic.items()
                             if vocab_df[word] >= self.min_document_frequency
                             and vocab_ic[word] >= self.min_interaction_count]

        self.vocab = [word for word, _ in sorted(vocab_ic_filtered,
                                                 key=lambda p: p[1],
                                                 reverse=True)]
        self.vocab_map = dict(zip(self.vocab, range(len(self.vocab))))

        if self.feature_idf:
            self.feature_weights = np.log([num_docs / vocab_df[word]
                                           for word in self.vocab]).astype(self.dtype)
        else:
            self.feature_weights = np.ones(len(self.vocab), dtype=self.dtype)
        return self

    def transform(self, id_document_iter):
        """
        Returns an IndexedSparseMatrix (wrapping a CSR matrix) with the
        documents ids as row labels giving the L1 normalisation of the (IDF
        weighted, if applicable) term frequency counts.
        """
        ism = super().transform(id_document_iter)
        if not len(self.feature_weights):
            # then ism has no columns, nothing more to do
            # (return, since sp.diags will fail)
            return ism
        # apply the feature weighting
        ism.M = ism.M * sp.diags(self.feature_weights, 0)
        # L1 normalise the rows (entries are non-negative)
        row_sums = np.array(ism.M.sum(axis=1))[:, 0]
        row_indices, _ = ism.M.nonzero()
        ism.M.data /= row_sums[row_indices]
        return ism

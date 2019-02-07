import unittest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from csr_featuriser import CSRFeaturiser
from patterns import TOKEN_PATTERN


docfile = 'assets/example_text.txt'


def read_docs():
    docs = dict()
    for i in range(10):
        with open(docfile, 'rt') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                docs['doc%3d' % len(docs)] = line * 10
                break
    return docs


class TestCSRFeaturiser(unittest.TestCase):

    def setUp(self):
        self.docs = read_docs()

    def test_compare_to_cv(self):
        for binary in [False, True]:
            cv = CountVectorizer(token_pattern=TOKEN_PATTERN, binary=binary)
            cv_mat = cv.fit_transform(self.docs.values())
            cv_vocab = cv.get_feature_names()

            # vocab given
            csrf = CSRFeaturiser(analyzer=cv.build_analyzer(), vocab=cv_vocab,
                                 dtype=np.uint8, binary=binary)
            csrf_mat = csrf.transform(self.docs.items())
            self.assertEqual((cv_mat - csrf_mat.M).sum(), 0)

            # without vocab
            csrf = CSRFeaturiser(analyzer=cv.build_analyzer(), vocab=None,
                                 dtype=np.uint8, binary=binary)
            csrf_mat = csrf.transform(self.docs.items())
            csrf_mat.sync_col_index(cv_vocab)
            self.assertEqual((cv_mat - csrf_mat.M).sum(), 0)

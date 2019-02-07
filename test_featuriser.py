# -*- coding: utf-8 -*-
import unittest
import re
import scipy.sparse as sp
import numpy as np
import pandas as pd
from patterns import TOKEN_PATTERN
from featuriser import Featuriser
from indexed_sparse_matrix import IndexedSparseMatrix

SPLITTER = re.compile(TOKEN_PATTERN).findall
TEXTS = ['cats cats', 'cats dogs', '', 'uùúûü']
DOC_IDS = ['id%i' % i for i in range(len(TEXTS))]
ID_TEXT_ITER = list(zip(DOC_IDS, TEXTS))
SEEN_COUNTS = pd.Series([1, 1, 1, 3], index=DOC_IDS)


class TestFeaturiser(unittest.TestCase):

    def _check_ism(self, mat, dtype):
        self.assertIsInstance(mat, IndexedSparseMatrix)
        self.assertIsInstance(mat.M, sp.csr_matrix)
        self.assertEqual(mat.M.dtype, dtype)
        norms = mat.M.sum(axis=1)  # check normalisation
        self.assertTrue((np.isclose(norms, 1) | np.isclose(norms, 0)).all())

    def test_fit_no_idf(self):
        dtype = np.float32
        ftr = Featuriser(SPLITTER, feature_idf=False, min_document_frequency=1, min_interaction_count=1, dtype=dtype)
        ftr.fit(ID_TEXT_ITER, SEEN_COUNTS)
        self.assertEqual(sorted(ftr.vocab), ['cats', 'dogs', 'uùúûü'])
        ism = ftr.transform(ID_TEXT_ITER)
        self._check_ism(ism, dtype)
        values = ism.M.toarray()
        # check some values
        self.assertEqual(values[0,ism.cols.index('cats')], 1)
        self.assertEqual(values[1,ism.cols.index('cats')], 0.5)
        self.assertEqual(values[2,ism.cols.index('cats')], 0.)

    def test_fit_idf(self):
        dtype = np.float32
        ftr = Featuriser(SPLITTER, feature_idf=True, min_document_frequency=1, min_interaction_count=1, dtype=dtype)
        ftr.fit(ID_TEXT_ITER, SEEN_COUNTS)
        self.assertEqual(sorted(ftr.vocab), ['cats', 'dogs', 'uùúûü'])
        ism = ftr.transform(ID_TEXT_ITER)
        self._check_ism(ism, dtype)
        values = ism.M.toarray()
        # check some values
        self.assertEqual(values[0,ism.cols.index('cats')], 1)
        # before normalisation, row 1 had values log2, log4, 0 for cats, dogs, uùúûü
        self.assertAlmostEqual(values[1,ism.cols.index('cats')], 1/3)
        self.assertAlmostEqual(values[1,ism.cols.index('dogs')], 2/3)
        self.assertEqual(values[2,ism.cols.index('cats')], 0.)

    def test_min_interaction_count(self):
        dtype = np.float32
        ftr = Featuriser(SPLITTER, feature_idf=True, min_document_frequency=1, min_interaction_count=3, dtype=dtype)
        ftr.fit(ID_TEXT_ITER, SEEN_COUNTS)
        self.assertEqual(ftr.vocab, ['uùúûü'])
        ism = ftr.transform(ID_TEXT_ITER)
        self._check_ism(ism, dtype)

    def test_min_doc_freq(self):
        dtype = np.float32
        ftr = Featuriser(SPLITTER, feature_idf=True, min_document_frequency=2, min_interaction_count=1, dtype=dtype)
        ftr.fit(ID_TEXT_ITER, SEEN_COUNTS)
        self.assertEqual(ftr.vocab, ['cats'])
        ism = ftr.transform(ID_TEXT_ITER)
        self._check_ism(ism, dtype)

    def test_empty_vocab(self):
        dtype = np.float32
        ftr = Featuriser(SPLITTER, feature_idf=True, min_document_frequency=5, min_interaction_count=5, dtype=dtype)
        ftr.fit(ID_TEXT_ITER, SEEN_COUNTS)
        self.assertEqual(ftr.vocab, [])
        ism = ftr.transform(ID_TEXT_ITER)
        self._check_ism(ism, dtype)

import unittest
import scipy.sparse as sp
import numpy as np
from indexed_sparse_matrix import IndexedSparseMatrix


def _random_ism(num_rows, num_cols, sp_type):
    """
    Return an instance of IndexedSparseMatrix of the specified shape with random values.
    sp_type is e.g. sp.csr_matrix
    row keys are "row0", .. (similarly for column keys)
    """
    s = sp_type(np.random.random((num_rows, num_cols)))
    row_keys = ['row%i' % i for i in range(num_rows)]
    col_keys = ['col%i' % i for i in range(num_cols)]
    return IndexedSparseMatrix(s, rows=row_keys, cols=col_keys)


class TestIndexedSparseMatrix(unittest.TestCase):

    def _check_equal(self, m1, m2):
        self.assertTrue(np.all(m1.todense() == m2.todense()))

    def test_init(self):
        for sptype in [sp.csc_matrix, sp.csr_matrix]:
            ism = _random_ism(6, 8, sptype)
            self.assertEqual(ism.rows, ['row%i' % i for i in range(6)])
            self.assertEqual(ism.cols, ['col%i' % i for i in range(8)])
            self.assertEqual(ism.M.shape, (6, 8))

    def test_init_wrong(self):
        with self.assertRaises(AssertionError):
            IndexedSparseMatrix(sp.eye(10), rows=range(3), cols=range(10))
        with self.assertRaises(AssertionError):
            IndexedSparseMatrix(sp.eye(10), rows=range(10), cols=range(3))

    def test_fancy_indexing(self):
        for sptype in [sp.csc_matrix, sp.csr_matrix]:
            ism = _random_ism(6, 8, sptype)
            row_idxs = [2, 3]
            col_idxs = [1, 4]
            row_keys = [ism.rows[i] for i in row_idxs]
            col_keys = [ism.cols[i] for i in col_idxs]
            ism_select = ism[row_idxs, :]
            ism_select = ism_select[:, col_idxs]
            # check shape
            self.assertEqual(ism_select.M.shape, (len(row_idxs), len(col_idxs)))
            # check values
            self._check_equal(ism_select.M, ism.M[row_idxs, :][:, col_idxs])
            # check keys
            self.assertEqual(ism_select.rows, row_keys)
            self.assertEqual(ism_select.cols, col_keys)

    def test_row_slice(self):
        for sptype in [sp.csc_matrix, sp.csr_matrix]:
            ism = _random_ism(6, 8, sptype)
            ism_row_slice = ism[1:3,:]
            self.assertEqual(ism_row_slice.M.shape, (2, len(ism.cols)))
            self._check_equal(ism_row_slice.M, ism.M[1:3,:])
            self.assertEqual(ism_row_slice.rows, ism.rows[1:3])
            self.assertEqual(ism_row_slice.cols, ism.cols)

    def test_col_slice(self):
        for sptype in [sp.csc_matrix, sp.csr_matrix]:
            ism = _random_ism(6, 8, sptype)
            ism_col_slice = ism[:,1:3]
            self.assertEqual(ism_col_slice.M.shape, (len(ism.rows), 2))
            self._check_equal(ism_col_slice.M, ism.M[:,1:3])
            self.assertEqual(ism_col_slice.rows, ism.rows)
            self.assertEqual(ism_col_slice.cols, ism.cols[1:3])


    def test_copy(self):
        for sptype in [sp.csc_matrix, sp.csr_matrix]:
            ism = _random_ism(6, 8, sptype)
            m = ism.copy()
            self._check_equal(m.M, ism.M)
            self.assertEqual(m.rows, ism.rows)
            self.assertEqual(m.cols, ism.cols)

    def test_sync_row_index(self):
        for sptype in [sp.csc_matrix, sp.csr_matrix]:
            ism = _random_ism(10, 10, sptype)
            ism_org = ism.copy()
            ext_idxs = [5, -5, 1, -1, 3, -3]
            ext_ics = ['row%i' % i for i in ext_idxs] 
            idxs = [i for i in ext_idxs if i >= 0]
            ics = ['row%i' % i for i in idxs]
            ism.sync_row_index(ext_ics)
            self.assertEqual(ism.M.shape, (len(ext_ics), ism_org.M.shape[1]))
            self._check_equal(ism.M[[0, 2, 4], :], ism_org.M[idxs, :])
            self.assertTrue(np.all(ism.M[[1, 3, 5], :].todense() == 0))
            self.assertIsInstance(ism.M, sp.csr_matrix)

    def test_sync_col_index(self):
        for sptype in [sp.csc_matrix, sp.csr_matrix]:
            ism = _random_ism(10, 10, sptype)
            ism_org = ism.copy()
            ext_idxs = [5, -5, 1, -1, 3, -3]
            ext_ics = ['col%i' % i for i in ext_idxs] 
            idxs = [i for i in ext_idxs if i >= 0]
            ics = ['col%i' % i for i in idxs]
            ism.sync_col_index(ext_ics)
            self.assertEqual(ism.M.shape, (ism_org.M.shape[1], len(ext_ics)))
            self._check_equal(ism.M[:, [0, 2, 4]], ism_org.M[:, idxs])
            self.assertTrue(np.all(ism.M[:, [1, 3, 5]].todense() == 0))
            self.assertIsInstance(ism.M, sp.csc_matrix)

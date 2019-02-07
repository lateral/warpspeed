from scipy import sparse


def value_ordered_keys(dic):
    """
    Return the keys of a dictionary order by its values. Useful to get the
    inverse mapping of an id to offset dictionary.
    """
    return [k for k, _ in sorted(dic.items(), key=lambda p: p[1])]


class IndexedSparseMatrix(object):
    """
    IndexedSparseMatrix imitates a pandas DataFrame for sparse data. It wraps a
    sparse matrix and attaches row and column names to it. Indexing operations
    keep rows and columns in sync with the data.
    """
    def __init__(self, M, rows, cols):
        self.M = M
        _check_unique(rows)
        _check_unique(cols)
        self.rows = list(rows)
        self.cols = list(cols)
        assert self.M.shape == (len(self.rows), len(self.cols))

    def copy(self):
        return IndexedSparseMatrix(self.M.copy(), self.rows, self.cols)

    def __getitem__(self, key):
        """
        Returns a submatrix, each axis defined by indices given either as slice
        or as iterable (fancy indexing is supported, by without duplicates).
        `key` is a tuple of indices, one per axis.
        REMARK: This only wraps the indexing of sparse matrices and does NOT
        correspond to pd.DataFrame.loc. Latter functionality can be simulated by
        sync_row_index and sync_col_index.
        """
        M = self.M[key]
        rows = _filter_keys(self.rows, key[0])
        cols = _filter_keys(self.cols, key[1])
        return IndexedSparseMatrix(M, rows=rows, cols=cols)

    def sync_row_index(self, ids):
        """
        Modifies self to have rows labeled and ordered by the (unique) `ids`
        taken from the original matrix if present or 0 otherwise.
        Converts matrix to csr format.
        """
        _check_unique(ids)
        extra_ids = _get_extra_ids(self.rows, ids)
        if len(extra_ids) > 0:
            self.M = sparse.vstack((
                self.M,
                sparse.csr_matrix(
                    (len(extra_ids), self.M.shape[1]), dtype=self.M.dtype)),
                format='csr')
        row_map = _get_map(self.rows + extra_ids)
        indices = [row_map[_id] for _id in ids]
        self.rows = ids
        self.M = self.M[indices, :]

    def sync_col_index(self, ids):
        """
        Modifies self to have columns labeled and ordered by the (unique) `ids`
        taken from the original matrix if present or 0 otherwise.
        Converts matrix to csc format.
        """
        _check_unique(ids)
        extra_ids = _get_extra_ids(self.cols, ids)
        if len(extra_ids) > 0:
            self.M = sparse.hstack((
                self.M,
                sparse.csc_matrix(
                    (self.M.shape[0], len(extra_ids)), dtype=self.M.dtype)),
                format='csc')

        col_map = _get_map(self.cols + extra_ids)
        indices = [col_map[_id] for _id in ids]
        self.cols = ids
        self.M = self.M[:, indices]


def _check_unique(ids):
    assert len(ids) == len(set(ids))


def _get_map(ids):
    return {k: i for i, k in enumerate(ids)}


def _get_extra_ids(index, ids):
    index_set = set(index)
    return [_id for _id in ids if _id not in index_set]


def _filter_keys(keys, indices):
    """
    Returns the list of keys at specified indices.
    `keys` is a list.
    `indices` is slice or iterable.
    """
    if isinstance(indices, slice):
        return keys[indices]
    else:
        return [keys[i] for i in indices]

import numpy as np
from scipy import sparse
from array import array
from collections import defaultdict
from indexed_sparse_matrix import IndexedSparseMatrix, value_ordered_keys


class CSRFeaturiser(object):
    """
    Substitution for sklearn's CountVectorizer.
    Vocabulary can be provided or determined on the fly. For more complex
    determination inherit from CSRFeaturiser and implement a fit method
    to build a vocabulary.
    Returns IndexedSparseMatrix wrapping a scipy CSR matrix, with document_ids
    as row index and vocabulary as column index.
    """

    def __init__(self, analyzer, vocab=None, dtype=np.int16, binary=False):
        self.analyzer = analyzer
        self.vocab = vocab
        if self.vocab is not None:
            self.vocab_map = dict(zip(self.vocab, range(len(self.vocab))))
        self.dtype = dtype
        self.binary = binary

    def transform(self, id_document_iter):
        """
        Build column sparse row matrix on the fly with document ids as
        row labels and word ids as column labels of an IndexedSparseMatrix.
        If `self.vocab` is None, then the vocabulary is determined here.
        `id_doc_iter` is an iterable of (id, doc) pairs.
        Returns IndexedSparseMatrix (CSR):
            with row order like in `id_doc_iter`
            and column order like in `vocab` if given.
        """
        if self.vocab is None:
            # determine the vocab on the fly
            self.vocab_map = defaultdict(lambda: len(self.vocab_map))

        row_ids = []
        ind_ptr, indices = array('i'), array('i')
        counts = array('h')

        ind_ptr.append(0)
        for _id, doc in id_document_iter:
            row_ids.append(_id)

            col_counts = {}
            for feature in self.analyzer(doc):
                try:
                    col = self.vocab_map[feature]
                    if col not in col_counts or self.binary:
                        col_counts[col] = 1
                    else:
                        col_counts[col] += 1
                except KeyError:
                    continue

            indices.extend(col_counts.keys())
            counts.extend(col_counts.values())
            ind_ptr.append(len(indices))

        matrix = sparse.csr_matrix((
            np.frombuffer(counts, dtype=np.int16),
            np.frombuffer(indices, dtype=np.int32),
            np.frombuffer(ind_ptr, dtype=np.int32)),
            shape=(len(row_ids), len(self.vocab_map)),
            dtype=self.dtype)
        matrix.sort_indices()

        if self.vocab is None:
            # if we learnt the vocab here, save the result
            self.vocab = value_ordered_keys(self.vocab_map) 
            self.vocab_map = dict(self.vocab_map)  # convert to dict

        return IndexedSparseMatrix(matrix, rows=row_ids, cols=self.vocab)

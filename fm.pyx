#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from cython.parallel import parallel, prange
cimport cython.operator.dereference as deref
from libc.stdlib cimport free, malloc
from prng cimport PRNG
import logging
import numpy as np
from collections import defaultdict
from evaluation import PercentileRank, Recommendation


LOGGER = logging.getLogger('FM')
CYTHON_FLOAT = np.float32
CYTHON_UINT = np.int32
MAX_MINIEPOCH_SIZE = 5 * 10 ** 5


ctypedef float flt


cdef extern from "math.h" nogil:
    double sqrt(double)
    double exp(double)
    double log(double)


cdef PRNG prng = PRNG()


cdef class FM:
    """
    A bipartite factorisation machine trained with a modification of the WARP
    loss [1] to use a multiplicative margin, the sigmoid activation function
    and log-likelihood style updates.

    [1] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling
    up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
    """

    cdef EntityModel users
    cdef EntityModel items
    cdef int rank
    cdef int max_sampled
    cdef flt margin

    def __init__(self, EntityModel users, EntityModel items,
                 margin=0.85, max_sampled=100):
        """
        `users` and `items` are EntityModels which contain both the feature
        counts and the feature vectors for users and items, respectively.
        `margin` and `max_sampled` are hyperparameters:
        - The activation of a candidate negative item must be at least `margin`
        times the activation of the positive item (where both activations are
        computed w.r.t. the same user) in order for it to be used as a negative
        sample.
        - `max_sampled` is the number of candidate negative samples to consider
          before giving up.
        """
        assert users.rank == items.rank
        self.rank = users.rank
        self.users, self.items = users, items
        self.margin = margin
        self.max_sampled = max_sampled

    def learn(self, interactions, number_threads):
        """
        Learn from the provided interactions. Repeated calls to this function
        will resume training from the point where the last call finished.
        - coo_matrix interactions: of shape [n_users, n_items] and type flt.
        """
        n_interactions = len(interactions.data)
        # we use mini-epochs of equal size, so that there is never a too
        # small mini-epoch (which would might destablise metrics)
        number_mini_epochs = (n_interactions // MAX_MINIEPOCH_SIZE) + 1
        examples_per_epoch = n_interactions // number_mini_epochs
        LOGGER.info('will train in %i mini-epochs of length %i',
                    number_mini_epochs, examples_per_epoch)
        shuffled_indices = np.arange(n_interactions, dtype=np.int32)
        np.random.shuffle(shuffled_indices)

        for i in range(number_mini_epochs):
            mini_epoch_indices = shuffled_indices[
                i * examples_per_epoch: (i + 1) * examples_per_epoch]
            LOGGER.info('mini-epoch %i of length %i', i, len(mini_epoch_indices))
            self._learn(interactions, mini_epoch_indices, number_threads)

    def percentile_ranks(self, user_idxs, expected_item_idxs):
        """
        Given two (equal length) iterables specifying the users to obtain
        recommendations for and the hoped-for item recommendation in each case,
        return a list of PercentileRank instances of the same length.
        """
        prs = []
        item_vecs = self.items.entity_vectors()
        for user_idx, item_idx in zip(user_idxs, expected_item_idxs):
            if not self.users.has_features(user_idx):
                continue
            user_vec = self.users._py_entity_vector(user_idx)
            dps = item_vecs.dot(user_vec)
            pr = np.mean(dps > dps[item_idx])
            prs.append(PercentileRank(user_idx, item_idx, pr))
        return prs

    def recommendations(self, user_idxs, N=5):
        """
        Return the top N recommendations for the iterable of user indices.  The
        return format is a single flat list of Recommendation objects.
        """
        recommendations = []
        item_vecs = self.items.entity_vectors()
        for user_idx in user_idxs:
            if not self.users.has_features(user_idx):
                continue
            user_vec = self.users._py_entity_vector(user_idx)
            dps = item_vecs.dot(user_vec)
            # sorting
            argpart = np.argpartition(dps, max(len(dps) - N, 0))
            dps_N = dps[argpart[-N:]]
            pos_N = argpart[-N:]
            argsort_N = np.argsort(dps_N)[::-1]
            ranked_items = pos_N[argsort_N]
            for rank, item_idx in enumerate(ranked_items):
                recommendations.append(Recommendation(user_idx, item_idx, rank))
        return recommendations

    def _learn(FM self, interactions, int[::1] indices_to_train, int number_threads):
        """
        As per learn(), but consider only those entries of the COO matrix
        interactions at the indices given by `indices_to_train`.
        """
        cdef:
            int[::1] user_ids, item_ids
            flt[::1] interaction_strength
            int user_id, positive_item_id, negative_item_id
            int i, sampled, index, gaveup, valid_interactions
            flt positive_item_score, negative_item_score
            flt update_weight, factor, log_number_items
            flt *user_vector
            flt *positive_item_vector
            flt *negative_item_vector

        log_number_items = log(self.items.n_entities)
        user_ids, item_ids = interactions.row, interactions.col
        interaction_strength = interactions.data
        valid_interactions = 0  # num interactions where user/item have features
        gaveup = 0  # how many times max_sampled is reached

        with nogil, parallel(num_threads=number_threads):
            # thread-local buffers
            user_vector = <flt *>malloc(sizeof(flt) * self.rank)
            positive_item_vector = <flt *>malloc(sizeof(flt) * self.rank)
            negative_item_vector = <flt *>malloc(sizeof(flt) * self.rank)

            for i in prange(indices_to_train.shape[0]):
                index = indices_to_train[i]
                user_id = user_ids[index]
                positive_item_id = item_ids[index]

                # skip if the user or item is featureless
                if not self.users.has_features(user_id):
                    continue
                if not self.items.has_features(positive_item_id):
                    continue
                valid_interactions += 1

                self.users.entity_vector(user_id, user_vector)
                self.items.entity_vector(positive_item_id, positive_item_vector)
                positive_item_score = compute_score(user_vector,
                                                    positive_item_vector,
                                                    self.rank)

                sampled = 0
                while sampled < self.max_sampled:
                    negative_item_id = prng.randint() % self.items.n_entities
                    if not self.items.has_features(negative_item_id):
                        continue
                    self.items.entity_vector(negative_item_id, negative_item_vector)
                    negative_item_score = compute_score(user_vector,
                                                        negative_item_vector,
                                                        self.rank)

                    sampled = sampled + 1
                    if negative_item_score < self.margin * positive_item_score:
                        # model doesn't believe enough in this negative item
                        continue

                    update_weight = interaction_strength[index] * log(
                        (self.items.n_entities - 1) / sampled) / log_number_items

                    # update positive item for user, and vice versa
                    factor = update_weight * (1 - positive_item_score)
                    self.items.update_entity(positive_item_id, user_vector, factor)
                    self.users.update_entity(user_id, positive_item_vector, factor)

                    # update negative item for user, and vice versa
                    factor = -update_weight * negative_item_score
                    self.items.update_entity(negative_item_id, user_vector, factor)
                    self.users.update_entity(user_id, negative_item_vector, factor)
                    break

                if sampled == self.max_sampled:
                    gaveup += 1

            free(user_vector)
            free(positive_item_vector)
            free(negative_item_vector)

        if valid_interactions > 0:
            LOGGER.info('gave up for %i of %i valid interactions (rate %.5f)',
                        gaveup, valid_interactions,
                        float(gaveup) / valid_interactions)


cdef inline flt compute_score(flt *user_repr, flt *item_repr, int rank) nogil:
    """
    Compute the sigmoid of the dot product of the provided vectors.
    """
    cdef int i
    cdef flt result = 0

    for i in range(rank):
        result += user_repr[i] * item_repr[i]
    return sigmoid(result)


cdef inline flt sigmoid(flt v) nogil:
    """
    Compute the sigmoid of v.
    """
    return 1.0 / (1.0 + exp(-v))


def build_init_vectors(number, rank):
    """
    Return a randomly initialised numpy array of shape (number, rank) of dtype
    np.float32.
    """
    # implementation note: np.random.rand returns float64, so sample one
    # vector at a time to avoid allocating (potentially huge) float64 array
    vectors = np.empty((number, rank), dtype=np.float32)
    for i in range(number):
        vectors[i, :] = (np.random.rand(rank) - 0.5) / rank
    return vectors


cdef class EntityModel:
    """
    Represents either users or items ("entities").  Entities have features with
    float mulitplicity, described by feature count matrix.  Features have
    feature vectors of a fixed rank.  Any entity can be represented as a vector
    by the weighted sum of its component feature vectors.  This class
    implements this model for entities, and provides also attributes and
    methods for updating the feature vectors, via the entities in which they
    occur, using a vector-based Adagrad learning regime.
    For a discussion of this modification of Adagrad, see:
        http://building-babylon.net/2016/10/05/adagrad-evolution-depends-on-the-choice-of-basis/
    Feature counts need not be integral (they might be tfidf, for example).

    Implementation note: updates are specified by the specification of both an
    update vector and a scaling factor.  This is convenient since the same
    update vector is applied to all features associated with an entity, but
    each time with a different scaling, proportional to the feature count.
    """

    cdef readonly CSRMatrix feature_counts
    cdef readonly flt[:, ::1] feature_vectors
    cdef readonly flt[::1] accumulated_gradients
    cdef readonly flt learning_rate
    cdef readonly int rank
    cdef readonly int n_entities, n_features

    def __init__(self, rank, learning_rate, feature_counts,
                 feature_vectors=None, accumulated_gradients=None):
        """
        `learning_rate` is the initial learning rate, prior to any Adagrad
        learning rate decay.
        `feature_counts` is a scipy CSR matrix.
        If `feature_vectors` is not provided, it is randomly initialised; if
        `accumulated_gradients` is not provided, it is initialised to zeros
        (which is the correct beginning for Adagrad learning).
        """
        self.learning_rate = learning_rate
        self.rank = rank
        self.n_entities, self.n_features = feature_counts.shape
        if feature_vectors is None:
            self.feature_vectors = build_init_vectors(self.n_features, self.rank)
        else:
            assert self.n_features == feature_vectors.shape[0]
            assert self.rank == feature_vectors.shape[1]
            self.feature_vectors = feature_vectors
        if accumulated_gradients is None:
            self.accumulated_gradients = np.zeros(self.n_features, dtype=CYTHON_FLOAT)
        else:
            assert self.n_features == accumulated_gradients.shape[0]
            self.accumulated_gradients = accumulated_gradients
        self.feature_counts = CSRMatrix(feature_counts)

    cdef inline flt _next_adagrad_weight(EntityModel self, flt *update_vector,
                                        flt update_scalar, int feature) nogil:
        """
        Returns current adagrad weight for specified feature, and updates
        `self.accumulated_gradients` in anticipation of the update specified by
        the given `update_scalar` times `update_vector`.
        """
        cdef int c
        cdef flt weight
        cdef flt sq_grad = 0.

        weight = self.learning_rate / sqrt(1. + self.accumulated_gradients[feature])
        for c in range(self.rank):
            sq_grad += update_vector[c] * update_vector[c]
        sq_grad *= update_scalar * update_scalar
        self.accumulated_gradients[feature] += sq_grad
        return weight

    cdef inline void update_entity(EntityModel self, int idx,
                                   flt *update_vector, flt update_scalar) nogil:
        """
        Update the feature vectors of the features associated to the specified
        entity, using the `update_scalar` times `update_vector`.
        """
        cdef int i, feature, c
        cdef flt feature_count, factor

        if self.learning_rate == 0.:
            return
        for i in range(self.feature_counts.row_start(idx),
                       self.feature_counts.row_end(idx)):
            feature = self.feature_counts.indices[i]
            feature_count = self.feature_counts.data[i]
            factor = update_scalar * feature_count
            factor *= self._next_adagrad_weight(update_vector, factor, feature)
            for c in range(self.rank):
                self.feature_vectors[feature, c] += factor * update_vector[c]

    def _py_update_entity(self, idx, flt[::1] update_vector, update_scalar):
        """
        A Python callable function for testing update_entity().
        """
        self.update_entity(idx, &update_vector[0], update_scalar)

    cdef inline void entity_vector(EntityModel self, int idx, flt *vector) nogil:
        """
        Compute the vector for the entity with index `idx` by aggregating the
        corresponding feature vectors.  Stores result in `vector`.
        """
        cdef int i, j, feature
        cdef flt feature_count

        for i in range(self.rank):
            vector[i] = 0.0

        for i in range(self.feature_counts.row_start(idx),
                       self.feature_counts.row_end(idx)):
            feature = self.feature_counts.indices[i]
            feature_count = self.feature_counts.data[i]

            for j in range(self.rank):
                vector[j] += feature_count * self.feature_vectors[feature, j]

    def _py_entity_vector(self, idx):
        """
        A Python callable function for entity_vector().
        """
        cdef flt[::1] vec = np.empty(self.rank, dtype=np.float32)
        self.entity_vector(idx, &vec[0])
        return np.array(vec)

    cdef inline bint has_features(EntityModel self, int idx) nogil:
        """
        Return whether the specified entity has any features.
        """
        return not self.feature_counts.is_empty_row(idx)

    def entity_vectors(self):
        """
        Calculate the entity vectors for all entities, and return them as a 2d
        numpy array of dtype np.float32 of shape (n_entities, n_features).
        """
        cdef flt[:,::1] vectors

        vectors = np.empty((self.n_entities, self.rank), dtype=np.float32)
        for i in range(self.n_entities):
            self.entity_vector(i, &vectors[i,0])
        return np.array(vectors)


cdef class CSRMatrix:
    """
    Utility class for accessing elements of a CSR matrix.
    """
    cdef int[::1] indices
    cdef int[::1] indptr
    cdef flt[::1] data
    cdef int rows
    cdef int cols
    cdef int nnz

    def __init__(self, csr_matrix):
        self.indices = csr_matrix.indices
        self.indptr = csr_matrix.indptr
        self.data = csr_matrix.data
        self.rows, self.cols = csr_matrix.shape
        self.nnz = len(self.data)

    cdef bint is_empty_row(self, int row) nogil:
        """
        Return boolean indicating whether row is empty
        """
        return self.indptr[row] == self.indptr[row + 1]

    cdef int row_start(self, int row) nogil:
        """
        Return the pointer to the start of the
        data for row.
        """
        return self.indptr[row]

    cdef int row_end(self, int row) nogil:
        """
        Return the pointer to the end of the
        data for row.
        """
        return self.indptr[row + 1]

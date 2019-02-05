# -*- coding: utf-8 -*-
import unittest
import scipy.sparse as sp
import numpy as np

from fm import EntityModel


class TestEntityModel(unittest.TestCase):

    def setUp(self):
        self.rank = 3
        self.learning_rate = 0.2
        self.n_features = 10
        self.n_entities = 2
        # randomly generate feature counts, except for the last row
        _fcs = np.random.randn(self.n_entities, self.n_features)
        _fcs[-1,:] = 0.
        _fcs[-1,0] = 1.
        _fcs[-1,1] = 0.5
        self.fcs = sp.csr_matrix(_fcs, dtype=np.float32)

    def _check_entity_model(self, em):
        self.assertEqual(em.rank, self.rank)
        self.assertEqual(em.n_features, self.n_features)
        self.assertEqual(em.n_entities, self.n_entities)
        self.assertAlmostEqual(em.learning_rate, self.learning_rate)
        self.assertEqual(em.feature_vectors.shape[0], self.n_features)
        self.assertEqual(em.feature_vectors.shape[1], self.rank)
        self.assertEqual(em.accumulated_gradients.shape[0], self.n_features)

    def test_init_no_vectors(self):
        em = EntityModel(self.rank, self.learning_rate, self.fcs)
        self._check_entity_model(em)
        self.assertAlmostEqual(em.accumulated_gradients[1], 0.)

    def test_init_with_vectors(self):
        feature_vecs = np.random.randn(self.n_features, self.rank).astype(dtype=np.float32)
        acc_grads = np.random.rand(self.n_features).astype(dtype=np.float32)
        em = EntityModel(self.rank, self.learning_rate, self.fcs, feature_vecs, acc_grads)
        self._check_entity_model(em)

    def test_aggregate_vectors(self):
        feature_vecs = np.random.randn(self.n_features, self.rank).astype(dtype=np.float32)
        acc_grads = np.random.rand(self.n_features).astype(dtype=np.float32)
        em = EntityModel(self.rank, self.learning_rate, self.fcs, feature_vecs, acc_grads)
        entity_vectors = em.entity_vectors()
        self.assertIsInstance(entity_vectors, np.ndarray)
        self.assertEqual(entity_vectors.shape, (self.n_entities, self.rank))
        self.assertEqual(entity_vectors.dtype, np.float32)
        expected_vecs = self.fcs.dot(feature_vecs)
        self.assertTrue((entity_vectors.round(2) == expected_vecs.round(2)).all())

    def test_update_entity(self):
        """
        Update an entity, and check that the feature vectors and their
        accumulated gradients change as expected.
        """
        # update an entity and check the accumulated gradients increase as they should
        em = EntityModel(self.rank, self.learning_rate, self.fcs)
        old_feature_vecs = np.array(em.feature_vectors).copy()
        update_vector = np.array([2, 0, 1], dtype=np.float32)
        update_scalar = 0.1
        update_sq = ((update_scalar * update_vector) ** 2).sum()  # this would be the accumulated gradient increment for the _entity_ vector
        # update the last entity
        entity = self.n_entities - 1
        em._py_update_entity(entity, update_vector, update_scalar)

        def _compare(array1, array2, length):
            for i in range(length):
                self.assertAlmostEqual(array1[i], array2[i])

        # check the accumulated_gradients for features of the entity
        expected_acc_grads = np.zeros(em.n_features, dtype=np.float32)
        # feature 0 has count 1 for the entity we updated
        expected_acc_grads[0] = (1 ** 2) * update_sq
        # feature 1 has count 0.5 for the entity we updated
        expected_acc_grads[1] = (0.5 ** 2) * update_sq
        _compare(em.accumulated_gradients, expected_acc_grads, em.n_features)

        new_feature_vecs = np.array(em.feature_vectors)
        # check that feature vecs for uninvolved features haven't changed
        self.assertTrue((new_feature_vecs[2:,:] == old_feature_vecs[2:,:]).all())
        # ... while the vectors of those involved _did_ change
        _v = em.learning_rate * update_scalar * update_vector
        _compare(new_feature_vecs[0,:] - old_feature_vecs[0,:], 1 * _v, em.rank)
        _compare(new_feature_vecs[1,:] - old_feature_vecs[1,:], 0.5 * _v, em.rank)

        # now do the entity update again to check that the acc grads dampen
        em._py_update_entity(entity, update_vector, update_scalar)
        newest_feature_vecs = np.array(em.feature_vectors)
        # check for feature 1
        feature = 1
        dampening = 1. / np.sqrt(1 + expected_acc_grads[feature])
        observed_delta = newest_feature_vecs[feature,:] - new_feature_vecs[feature,:]
        expected_delta = 0.5 * dampening * _v
        _compare(observed_delta, expected_delta, em.rank)

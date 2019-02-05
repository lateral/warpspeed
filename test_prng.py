# -*- coding: utf-8 -*-
import unittest
import numpy as np
from collections import Counter

from prng import randint_roundrobin, randfloat_roundrobin, int_max, PRNG


def _randint_roundrobin(seed_list, number_samples):
    seeds = np.array(seed_list, dtype=np.uint32)
    samples = np.empty(number_samples, dtype=np.int32)
    randint_roundrobin(seeds, samples)
    return samples


def _randfloat_roundrobin(seed_list, number_samples):
    seeds = np.array(seed_list, dtype=np.uint32)
    samples = np.empty(number_samples, dtype=np.float32)
    randfloat_roundrobin(seeds, samples)
    return samples


class TestPRNG(unittest.TestCase):

    def test_init_no_seed(self):
        prng = PRNG()

    def test_state_dependence(self):
        # results depend on state: different seed, different samples
        num_samples = 10
        for fn in [_randint_roundrobin, _randfloat_roundrobin]:
            samples0 = fn([0], num_samples)
            samples1 = fn([1], num_samples)
            self.assertFalse((samples0 == samples1).all())

    def test_reproducibility(self):
        # results depend only on state: same stae, same samples
        num_samples = 10
        for fn in [_randint_roundrobin, _randfloat_roundrobin]:
            samples_a = fn([0], num_samples)
            samples_b = fn([0], num_samples)
            self.assertTrue((samples_a == samples_b).all())

    def test_randint_range(self):
        num_trials = 100000
        samples = _randint_roundrobin([0], num_trials)
        self.assertTrue((samples >= 0).all())
        self.assertTrue((samples <= int_max).all())

    def test_randint_distribution(self):
        num_trials = 100000
        samples = _randint_roundrobin([0], num_trials)
        for modulus in [2 ** 4, 17]:
            remainders = list(samples % modulus)
            counter = Counter(remainders)
            for outcome in range(modulus):
                self.assertGreater(counter[outcome], num_trials / (modulus + 1))

    def test_randfloat_range(self):
        num_trials = 100000
        samples = _randfloat_roundrobin([0], num_trials)
        self.assertTrue((samples >= 0).all())
        self.assertTrue((samples <= 1).all())

    def test_randfloat(self):
        num_trials = 100000
        samples = _randfloat_roundrobin([0], num_trials)
        for num_bins in [17, 20]:
            bins = np.floor(samples * num_bins)
            bin_counts = Counter(list(bins))
            for bin_number in range(num_bins):
                self.assertGreater(bin_counts[bin_number], num_trials / (num_bins + 1))

    def test_same_seed_same_numbers(self):
        """
        Threads that start with the same seed should obtain the same randints / randfloats.
        """
        num_samples = 10000
        num_threads = 3
        seeds = [0, 1, 0]
        for fn in [_randfloat_roundrobin, _randint_roundrobin]:
            samples = fn(seeds, num_samples)
            # samples drawn by the 0th and the 2nd thread should be the same
            for i in range(num_samples // num_threads):
                self.assertEqual(samples[num_threads * i], samples[num_threads * i + 2])

    def test_compare_single_to_multithreaded(self):
        """
        Same results are obtained for a thread irrespective of how many other threads there are
        """
        num_samples_per_thread = 10
        num_threads = 2

        for fn in [_randfloat_roundrobin, _randint_roundrobin]:
            seeds = [0, 1]
            mt_samples = fn(seeds, num_threads * num_samples_per_thread)
            for thread_num in range(num_threads):
                st_samples = fn(seeds[thread_num:thread_num + 1], num_samples_per_thread)
                for i in range(num_samples_per_thread):
                    self.assertEqual(st_samples[i], mt_samples[num_threads * i + thread_num])

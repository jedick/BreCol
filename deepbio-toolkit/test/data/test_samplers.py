import unittest
import numpy as np

from dbtk.data.samplers import abundance

class TestAbundanceSampler(unittest.TestCase):
    def setUp(self):
        self.probabilities = np.array([0.1, 0.3, 0.6])
        self.n_samples = 100000
        self.tree = abundance.build_tree(np.cumsum(self.probabilities))

    def test_build_abundance_sampling_tree(self):
        self.assertEqual(len(self.tree), len(self.probabilities) - 1)

    def test_output_length(self):
        from dbtk.data import samplers
        n = sum(1 for _ in samplers.abundance.sample(self.n_samples, self.tree))
        self.assertGreater(n, 0)
        self.assertLessEqual(n, self.n_samples)

    def test_output_sum(self):
        from dbtk.data import samplers
        n = sum(count for _, count in samplers.abundance.sample(self.n_samples, self.tree))
        self.assertEqual(n, self.n_samples)

    def test_distribution_shape(self):
        from dbtk.data import samplers
        indices, counts = zip(*samplers.abundance.sample(self.n_samples, self.tree))
        counts = np.array(counts) / self.n_samples
        self.assertTrue(np.all(np.abs(counts - self.probabilities) < 0.05))
        self.assertTrue(np.all(indices == np.arange(len(self.probabilities))))

if __name__ == "__main__":
    unittest.main()

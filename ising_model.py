from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix


class IsingModel:

    def __init__(self, fields: np.array, interactions: csr_matrix):
        self.N = fields.shape[0]
        self.mu = fields
        self.J = interactions

        assert self.mu.shape == (self.N,)
        assert self.J.shape == (self.N, self.N)

        # Validate: J must be symmetric, with zero diagonal.
        assert (self.J - self.J.transpose()).count_nonzero() == 0
        assert np.sum(np.abs(self.J.diagonal())) == 0

    @staticmethod
    def create(fields: np.array, interactions: list):
        N = len(fields)
        i, j, v = [], [], []
        for v1, v2, value in interactions:
            assert 0 <= v1 < N
            assert 0 <= v2 < N
            i.append(v1)
            j.append(v2)
            v.append(value)
        J = csr_matrix((v + v, (i + j, j + i)), shape=(N, N))
        return IsingModel(fields, J)

    def state_by_id(self, state_id):
        assert self.N <= 20
        assert 0 <= state_id < 2 ** self.N
        ans = np.ones(self.N, dtype=np.int32) * -1
        for i in range(self.N):
            if (state_id >> i) % 2 == 1:
                ans[i] = 1
        return ans

    def state_to_id(self, state):
        assert self.N <= 20
        return np.dot((1 + state) // 2, 2 ** np.array(range(self.N)))

    def H(self, x):
        """Hamiltonian (energy). P ~ exp(H)."""
        return np.dot(x, self.mu) + 0.5 * np.dot(x, self.J.dot(x))

    def inference_bruteforce(self):
        assert self.N <= 20

        h = np.zeros(2 ** self.N)
        for i in range(2 ** self.N):
            h[i] = self.H(self.state_by_id(i))

        # Partition function.
        self.Z = np.sum(np.exp(h))

        # Probabilities of individual states.
        self.proba = np.exp(h) / self.Z

        # Marginal probabilities of vertices having value +1.
        self.marginal_proba = np.zeros(self.N)
        for state_id in range(2 ** self.N):
            for j in range(self.N):
                if (state_id >> j) % 2 == 1:
                    self.marginal_proba[j] += self.proba[state_id]

    def sample_mc(self, num_samples=1, iters=10) -> np.array:
        """Samples iid realizations using Monte-Carlo simulation."""
        shape = (self.N, num_samples)
        x = -1 + 2 * np.random.randint(0, 2, size=shape)

        schedule = np.random.permutation(list(range(self.N)) * iters)
        for v in schedule:
            k = np.minimum(100, self.mu[v] + self.J.getrow(v).dot(x))
            prob = 1.0 / (1.0 + np.exp(2 * k))  # P(-1)
            x[v, :] = -1 + 2 * (np.random.random(num_samples) > prob)

        return x.T

    def marginal_proba_mc(self, num_samples=1000000, iters=10):
        """Estimates marginal probabilities of value +1 for each vertex."""
        samples = self.sample_mc(num_samples=num_samples,
                                 iters=iters)
        return (num_samples + np.sum(samples, axis=0)) / (2 * num_samples)

import numpy as np

from ising_model import IsingModel


def test_bruteforce_independednt():
    mu = np.array([1, 2, 3, 4, 5])
    model = IsingModel.create(mu, [])
    model.inference_bruteforce()
    assert np.allclose(model.marginal_proba,
                       np.exp(mu) / (np.exp(mu) + np.exp(-mu)))


def test_bruteforce_one_edge():
    model = IsingModel.create(np.zeros(2), [(0, 1, 1.0)])
    model.inference_bruteforce()

    assert np.allclose(model.Z, 2 * (np.exp(1) + np.exp(-1)))
    assert np.allclose(model.proba, np.exp([1, -1, -1, 1]) / model.Z)
    assert np.allclose(model.marginal_proba, [0.5, 0.5])


def test_mc_close_to_bruteforce():
    N = 5
    np.random.seed(40)
    mu = -2 + 4 * np.random.random(N)
    edges = {}
    for i in range(2 * N):
        i, j = np.random.choice(N, 2, replace=False)
        if i > j: i, j = j, i
        edges[(i, j)] = - 2 + 4 * np.random.random()
    J = [(i, j, v) for (i, j), v in edges.items()]
    im = IsingModel.create(mu, J)

    im.inference_bruteforce()
    prob1 = im.marginal_proba
    prob2 = im.marginal_proba_mc(num_samples=100000)

    assert np.allclose(prob1, prob2, atol=0.01)

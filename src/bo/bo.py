import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from warnings import simplefilter
from warnings import catch_warnings
from sklearn.gaussian_process import GaussianProcessRegressor


def model_approx(model, x):
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(x, return_std=True)


def acquisition(x, x_samples, model):
    # calculate the best surrogate score found so far
    yhat, _ = model_approx(model, x)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = model_approx(model, x_samples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    # note: this balances the mean and std
    probs = scipy.stats.norm.cdf((mu - best) / (std + 1e-9))
    return probs


def sample_next(x, model, sample_size=500):
    # random search, generate random samples
    x_samples = np.random.random(size=(sample_size, x.shape[1]))
    # calculate the acquisition function for each sample
    scores = acquisition(x, x_samples, model)
    # locate the index of the largest scores
    ix = np.argmax(scores)
    return x_samples[ix:ix+1, :]


class BayesOpt(object):
    def __init__(self):
        # Gaussian process
        self.model = GaussianProcessRegressor()

    def update(self, x, y):
        # fit gaussian model
        self.model.fit(x, y)
        # get next sample point from acquisition function
        n = sample_next(x, self.model)
        return n

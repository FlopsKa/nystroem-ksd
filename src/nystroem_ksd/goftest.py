import numpy as np

import kgof.util as util
import kgof.data as data
from kgof.goftest import bootstrapper_rademacher, GofTest
from rfsd.rfsd import Divergence


class NystroemKSD(GofTest, Divergence):
    """
    Nyström-based acceleration of the kernel Stein discrepancy goodness-of-fit
    test by Chwialkowski et al., 2016 and Liu et al., 2016 in ICML 2016. We use
    the framework developed by Jitkrittum et al., 2017 in NeurIPS for
    ease-of-use. 

    - This test runs in O(d(mn + m^3)) time.

    H0: the sample follows p H1: the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, m = lambda n : int(4*np.sqrt(n)), bootstrapper=bootstrapper_rademacher, alpha=0.01,
            n_simulate=500, seed=11):
        """
        p: an instance of UnnormalizedDensity
        k: a KSTKernel object
        m : function (n) |-> number of Nyström samples
        bootstrapper: a function: (n) |-> numpy array of n weights 
            to be multiplied in the double sum of the test statistic for generating 
            bootstrap samples from the null distribution.
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(NystroemKSD, self).__init__(p, alpha)
        self.k = k
        self.m = m
        self.bootstrapper = bootstrapper
        self.n_simulate = n_simulate
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def perform_test(self, dat, return_simulated_stats=False, return_ustat_gram=False):
        """
        dat: a instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]

            test_stat, (H_mn, H_mm_inv) = self.compute_stat(dat, return_ustat_gram=True)
            # bootrapping
            sim_stats = np.zeros(n_simulate)
            with util.NumpySeedContext(seed=self.seed):
                for i in range(n_simulate):
                   W = self.bootstrapper(n)
                   boot_stat = 1/(n**2)*W.T@H_mn.T@H_mm_inv@H_mn@W
                   sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            pvalue = np.mean(sim_stats > test_stat)
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                 'time_secs': t.secs, 
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        if return_ustat_gram:
            results['H'] = (H_mn, H_mm_inv)
            
        return results
    
    def h_p(self, X, Y):
        """Construct a Gram (similarity) matrix of nxd X and mxd Y. Returns an
        nxm matrix."""
        _, d = X.shape
        grad_logpX = self.p.grad_log(X)
        grad_logpY = self.p.grad_log(Y)
        gram_glogp = grad_logpX @ grad_logpY.T # np.inner does the same but is slower
        K = self.k.eval(X, Y)

        B = np.zeros((len(X), len(Y)))
        C = np.zeros((len(X), len(Y)))
        for i in range(d):
            B += self.k.gradX_Y(X, Y, i) * grad_logpY[:,i].T
            C += (self.k.gradY_X(X, Y, i).T * grad_logpX[:,i]).T

        return K*gram_glogp + B + C + self.k.gradXY_sum(X, Y)

    def compute_stat(self, dat, return_ustat_gram=False):
        """
        If `return_ustat_gram` is `True`, returns the tuple (H_mn, H_mm_inv).
        """
        X = dat.data()
        n = len(X)
        idx = self.rng.choice(n,size=self.m(n),replace=True)

        H_mn = self.h_p(X[idx],X)
        H_mm_inv = np.linalg.pinv(H_mn[:,idx],hermitian=True)
        beta = H_mn@np.ones((n,1))/n
                
        # Nystroem-based
        stat = (beta.T@H_mm_inv@beta)[0][0]
        if return_ustat_gram:
            return stat, (H_mn, H_mm_inv)
        else:
            return stat

    def divergence(self, X, Y=None, **kwargs):
        return np.sqrt(self.compute_stat(dat=data.Data(X)))
    
    def __str__(self):
        return '%s(k=%s)' % (type(self).__name__, self.k)
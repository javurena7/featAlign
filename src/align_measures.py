import numpy as np
from itertools import product
from collections import Counter
import dit

class CentroidDistance(object):
    def __init__(self, data):
        self.data = data
        self.opt = {}

    def _init_signs(self, idx):
        self.idx = idx
        self.sign_iter = product([-1, 1], repeat=len(idx))
        self.curr_sign = next(self.sign_iter)

    def _minimize_c(self):
        c = [self.curr_sign[i]*self.data[:, idx] for i, idx in enumerate(self.idx)]
        den = len(self.idx)
        c = np.sum(np.array(c).T, axis=1) / den

        fun_val = [sum((c-self.curr_sign[i]*dvec)**2) for i, dvec in enumerate(self.data[:, self.idx].T)]
        self.curr_val = sum(fun_val)
        self.curr_c = c

    def _compare_current_value(self):
        if self.curr_val <= self.opt.get('val', np.inf):
            opt = {'val': self.curr_val,
                    'sign': self.curr_sign,
                    'c': self.curr_c}
            self.opt = opt

    def _iter(self):
        self.curr_sign = next(self.sign_iter)
        self._minimize_c()
        self._compare_current_value()

    def get_distance(self, idx):
        """
        idx: vector of column indeces from which to calculate the distance
        """
        self.opt = {}
        self._init_signs(idx)
        while True:
            try:
                self._iter()
            except StopIteration:
                break
        return self.opt


class OInformation(object):
    def __init__(self, data):
        self.data = data
        self._get_dist()

    def _get_dist(self):
        data = self.data + 1
        counts = Counter([tuple(x) for x in data])
        lx = self.data.shape[0]
        probs = {k: v/lx for k, v in counts.items()}
        self.distr = dit.Distribution(probs)
        self.distr.set_rv_names([str(i) for i in range(self.data.shape[1])])

    def get_total_corr(self, idx=None):
        if idx is not None:
            if not isinstance(idx[0], str):
                idx = [str(i) for i in idx]
        totcorr = dit.multivariate.coinformation(self.distr, rvs=idx)
        self.totcorr = totcorr

    def get_dual_total_corr(self, idx):
        if idx is not None:
            if not isinstance(idx[0], str):
                idx = [str(i) for i in idx]
        totcorr = dit.multivariate.dual_total_correlation(self.distr, rvs=idx)
        self.dtotcorr = totcorr

    def get_oinformation(self, idx):
        self.get_total_corr(idx)
        self.get_dual_total_corr(idx)
        o_inf = self.totcorr - self.dtotcorr
        self.o_inf = o_inf


if __name__ == '__main__':
    import sample_data as sd
    SD = sd.SampleData()
    # get a sample of 12 features (some of which are related)
    data = SD.get_basic_sample()

    # data must be an array where columns are variables and rows are observations
    CD = CentroidDistance(data)

    # To calculate the distance between any number of variables, list the indeces
    # of the variables. Eg., idx = [0, 2, 5] for distance between variables 0, 2 and 5.
    idx = [0, 2, 5, 7]

    #dist is a dict with objective function value (val), sign of the features (sign) and
    # value of the centroid (c)
    dist = CD.get_distance(idx)




import numpy as np

class SampleData(object):
    def __init__(self, p=None, N=None):
        self.p = p
        self.N = N

    def get_basic_sample(self, p=[.25, .25, .5], N=2000):
        self.p = p
        self.N = N
        x0 = self.sample_indep(self.N)
        x1 = self.sample_indep(self.N)
        x2 = self.sample_cond(x0)
        x3 = self.sample_cond(x2)
        x4 = self.sample_bicond(x0, x2)
        x5 = self.sample_bicond(x0, x1)
        x6 = self.sample_syn(x0, x1, pdep=.1)
        x7 = self.sample_syn(x0, x1, pdep=.9)
        x8 = self.sample_bicond(x2, x7, px=.5, pdep=.75)
        x9 = self.sample_syn(x0, x7)
        x10 = self.sample_condneg(x1, pdep=.75)
        x11 = self.sample_condneg(x3, pdep=.75)


        data = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])
        #data = np.array([x0, x1, x2, x4, x7, x8, x9, x11])

        return data.T

    def sample_indep(self, N=None):
        if not N:
            N = self.N
        x = np.random.choice([1, -1, 0], size=N, p=self.p, replace=True)
        return x

    def sample_cond(self, x, pdep=.5):
        y = [xi if np.random.random() < pdep else self.sample_indep(1)[0] for xi in x]
        return y

    def sample_condneg(self, x, pdep=.5):
        y = [-xi if np.random.random() < pdep else self.sample_indep(1)[0] for xi in x]
        return y

    def sample_bicond(self, x, y, px=.7, pdep=.5):
        z = [np.random.choice([xi, yi], p=[px, 1-px]) if np.random.random() < pdep else self.sample_indep(1)[0] for xi, yi in zip(x, y)]
        return z

    def sample_syn(self, x, y, pdep=.5):
        z = [xi*yi if np.random.random() < pdep else self.sample_indep(1)[0] for xi, yi in zip(x, y)]
        return z

if __name__ == '__main__':
    SD = SampleData()
    data = SD.get_basic_sample()
    np.savetxt('../data/sample.txt', data, fmt='%i')




import sample_data as sd
import align_measures as am
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from itertools import product
import seaborn as sns
from collections import Counter

class EvalBasic(object):
    def __init__(self):
        self.SD = sd.SampleData()
        self.cases = ['indep', 'pair', 'path', 'source', 'tricond', 'synergy', 'csynergy', 'latent'] #, 'bicond']

    def eval_centroid_p(self):
        pal = sns.color_palette('tab10')
        ccolor = {case: col for case, col in zip(self.cases, pal)}
        lps = 100
        fig, axs = plt.subplots(2, 4, figsize=(3*4, 3*2))
        ps = np.linspace(0, 1, lps)
        js = {0.: 0, 0.333: 1, 0.667: 2, 1.: 3}
        for case in self.cases:
            val, c = [], {k: np.zeros(lps) for k in [0., 0.333, 0.667, 1., -0.333, -0.667, -1.]}
            for i, p in enumerate(ps):
                ptot = [p/2, p/2, 1-p] #1-p, 0]
                data = self.get_samps(case, p=ptot)
                CD = am.CentroidDistance(data)
                opt = CD.get_distance([0, 1, 2])
                val.append(opt['val'])
                cvals = Counter(opt['c'])

                for k, v in cvals.items():
                    kn = np.round(k, 3)
                    c[kn][i] = v/data.shape[0]
            axs[0, 0].plot(ps, val, '-', alpha=.2, c=ccolor[case])
            axs[0, 0].plot(ps, val, '.', label=case, c=ccolor[case])

            for k, v in c.items():
                i = 0
                if k <= 0:
                    i = 1
                j = js[np.abs(k)]
                axs[i, j].plot(ps, v, '-', alpha=.2, c=ccolor[case])
                axs[i, j].plot(ps, v, '.', c=ccolor[case])
                axs[i, j].set_title(f'c={k}')
                axs[i, j].set_xlabel('p')
                axs[i, j].set_ylabel(f'P(c)={k}')
            #axs[1].plot(ps, c, '.', alpha=.3)
            #axs[2].plot(c, val, '.', label=case, alpha=.4)
        axs[0, 0].set_xlabel('p')
        axs[0, 0].set_ylabel('fval')
        """
        axs[1].set_xlabel('p')
        axs[2].set_xlabel('c')
        axs[1].set_ylabel('c')
        axs[2].set_ylabel('fval')
        """
        axs[0, 0].legend()
        fig.suptitle('Effect of p (probability of 1)')
        fig.tight_layout()
        fig.savefig(f'../plots/eval_centroid_cdist_p.pdf')


    def eval_centroid_0(self):
        fig, axs = plt.subplots(1, 3, figsize=(3*3, 3))
        ps = np.linspace(0, 1, 40)
        for case in self.cases:
            val, c = [], []
            for p in ps:
                ptot = [p/2, p/2, 1-p]
                data = self.get_samps(case, p=ptot)
                CD = am.CentroidDistance(data)
                opt = CD.get_distance([0, 1, 2])
                val.append(opt['val'])
                #c.append(np.abs(opt['c']))
            axs[0].plot(ps, val, '-', label=case)
            #axs[1].plot(ps, c, '-', label=case)
            #axs[2].plot(c, val, '-', label=case)
        axs[0].set_xlabel('p')
        axs[1].set_xlabel('p')
        axs[2].set_xlabel('|c|')
        axs[0].set_ylabel('fval')
        axs[1].set_ylabel('|c|')
        axs[2].set_ylabel('fval')
        axs[0].legend()
        fig.suptitle('Effect of p (1-p prob of zero)')
        fig.tight_layout()
        fig.savefig('../plots/eval_centroid_0.pdf')


    def eval_centroid_hm(self, n=100):
        fig, axs = plt.subplots(5, 8, figsize=(4*8, 4*5))
        ps = np.linspace(0, 1, n)
        for col, case in enumerate(self.cases):
            print(case)
            val, c1, c2 = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
            c3, c4 = np.zeros((n, n)), np.zeros((n, n))
            val[:] = np.nan; c1[:] = np.nan
            c2[:] = np.nan; c3[:] = np.nan; c4[:] = np.nan
            for (i, p1), (j, p2) in product(enumerate(ps), enumerate(ps)):
                if p1 + p2 < 1:
                    ptot = [p1, p2, 1-p1-p2]
                    data = self.get_samps(case, p=ptot)
                    CD = am.CentroidDistance(data)
                    opt = CD.get_distance([0, 1, 2])
                    val[i, j] = opt['val']
                    cvals = Counter(np.round(opt['c'], 3))


                    c1[i, j] = cvals.get(1., 0) / data.shape[0]
                    c2[i, j] = cvals.get(.667, 0) / data.shape[0]
                    c3[i, j] = cvals.get(.333, 0) / data.shape[0]
                    c4[i, j] = cvals.get(0., 0) / data.shape[0]


            sns.heatmap(val, cbar=False, ax=axs[0, col])
            sns.heatmap(c1, vmin=0, vmax=1, cbar=False, ax=axs[1, col], cmap='mako', center=0)
            sns.heatmap(c2, vmin=0, vmax=1, cbar=False, ax=axs[2, col], cmap='mako', center=0)
            sns.heatmap(c3, vmin=0, vmax=1, cbar=False, ax=axs[3, col], cmap='mako', center=0)
            sns.heatmap(c4, vmin=0, vmax=1, cbar=False, ax=axs[4, col], cmap='mako', center=0)
            for k in range(5):
                axs[k, col].invert_yaxis()
                axs[k, col].set_xlabel(r'$p_x=P(-1)$')
                axs[k, col].set_ylabel(r'$p_y=P(1)$')

            axs[0, col].set_title(case)
        fig.subplots_adjust(right=0.9)
        cbar_val = fig.add_axes([0.96, 5/7, 0.015, 1/8])
        cbar_c1 = fig.add_axes([0.96, 4/7, 0.015, 1/8])
        cbar_c2 = fig.add_axes([0.96, 3/7, 0.015, 1/8])
        cbar_c3 = fig.add_axes([0.96, 2/7, 0.015, 1/8])
        cbar_c4 = fig.add_axes([0.96, 1/7, 0.015, 1/8])

        sns.heatmap(val, cbar=True, ax=axs[0, col], cbar_ax=cbar_val)
        sns.heatmap(c1, vmin=0, vmax=1, cbar=True, ax=axs[1, col], cbar_ax=cbar_c1, cmap='Spectral', center=0)
        sns.heatmap(c2, vmin=0, vmax=1, cbar=True, ax=axs[2, col], cbar_ax=cbar_c2, cmap='Spectral', center=0)
        sns.heatmap(c3, vmin=0, vmax=1, cbar=True, ax=axs[3, col], cbar_ax=cbar_c3, cmap='Spectral', center=0)
        sns.heatmap(c4, vmin=0, vmax=1, cbar=True, ax=axs[4, col], cbar_ax=cbar_c4, cmap='Spectral', center=0)

        rows = {0: 'F-val', 1: r'$P(c=1)$', 2: r'$P(c=2/3)$', 3: r'$P(c=1/3)$', 4: r'$P(c=0)$'}
        for k in range(5):
            axs[k, col].invert_yaxis()
            axs[k, col].set_xlabel(r'$p_x=p(-1)$')
            axs[k, col].set_ylabel(r'$p_y=p(1)$')
            axs[k, 0].set_ylabel(rows[k])


        fig.suptitle('Heatmap of P')
        fig.savefig('../plots/eval_centroid_cdist_hm.pdf')

    def get_samps(self, case, p, pdep=.75):
        N = 2000
        self.SD = sd.SampleData(p, N)
        x = self.SD.sample_indep()
        if case == 'indep':
            y = self.SD.sample_indep()
            z = self.SD.sample_indep()
        elif case == 'pair':
            y = self.SD.sample_cond(x, pdep)
            z = self.SD.sample_indep()
        elif case == 'path':
            y = self.SD.sample_cond(x, pdep)
            z = self.SD.sample_cond(y, pdep)
        elif case == 'source':
            y = self.SD.sample_cond(x, pdep)
            z = self.SD.sample_cond(x, pdep)
        elif case == 'bicond':
            y = self.SD.sample_indep()
            z = self.SD.sample_bicond(x, y, px=.5, pdep=pdep)
        elif case == 'tricond':
            y = self.SD.sample_cond(x, pdep)
            z = self.SD.sample_bicond(x, y, px=.5, pdep=pdep)
        elif case == 'synergy':
            y = self.SD.sample_indep()
            z = self.SD.sample_syn(x, y, pdep=pdep)
        elif case == 'csynergy':
            y = self.SD.sample_cond(x, pdep)
            z = self.SD.sample_syn(x, y, pdep=pdep)
        elif case == 'latent':
            p = self.SD.sample_indep()
            x = self.SD.sample_cond(p, pdep)
            y = self.SD.sample_cond(p, pdep)
            z = self.SD.sample_cond(p, pdep)
            pass
        return np.array([x, y, z]).T

    def eval_oinf_p(self):
        fig, axs = plt.subplots(1, 3, figsize=(3*3, 3))
        ps = np.linspace(0, 1, 100)
        for case in self.cases:
            tc, dtc, oinf = [], [], []
            for p in ps:
                ptot = [p, 1-p, 0]
                data = self.get_samps(case, p=ptot)
                CD = am.OInformation(data)
                CD.get_oinformation([0, 1, 2])
                tc.append(CD.totcorr)
                dtc.append(CD.dtotcorr)
                oinf.append(CD.o_inf)
            axs[0].plot(ps, tc, '-', label=case)
            axs[1].plot(ps, dtc, '-', label=case)
            axs[2].plot(ps, oinf, '-', label=case)
        axs[0].set_xlabel('p')
        axs[1].set_xlabel('p')
        axs[2].set_xlabel('p')
        axs[0].set_ylabel('Tot. Corr. ')
        axs[1].set_ylabel('Dual. Tot. Corr. ')
        axs[2].set_ylabel('O-Information')
        axs[2].legend()
        fig.suptitle('Effect of p (probability of 1)')
        fig.tight_layout()
        fig.savefig(f'../plots/eval_oinf_p.pdf')


    def eval_oinf_0(self):
        fig, axs = plt.subplots(1, 3, figsize=(3*3, 3))
        ps = np.linspace(0, 1, 100)
        for case in self.cases:
            tc, dtc, oinf = [], [], []
            for p in ps:
                ptot = [p/2, p/2, 1-p]
                data = self.get_samps(case, p=ptot)
                CD = am.OInformation(data)
                CD.get_oinformation([0, 1, 2])
                tc.append(CD.totcorr)
                dtc.append(CD.dtotcorr)
                oinf.append(CD.o_inf)
            axs[0].plot(ps, tc, '-', label=case)
            axs[1].plot(ps, dtc, '-', label=case)
            axs[2].plot(ps, oinf, '-', label=case)
        axs[0].set_xlabel('p')
        axs[1].set_xlabel('p')
        axs[2].set_xlabel('p')
        axs[0].set_ylabel('Tot. Corr. ')
        axs[1].set_ylabel('Dual. Tot. Corr. ')
        axs[2].set_ylabel('O-Information')
        axs[2].legend()
        fig.suptitle('Effect of p (1-p prob of zero)')
        fig.tight_layout()
        fig.savefig('../plots/eval_oinf_0.pdf')


    def eval_oinf_hm(self, n=100):
        fig, axs = plt.subplots(3, 8, figsize=(3*8, 3*3))
        ps = np.linspace(0, 1, n)
        for col, case in enumerate(self.cases):
            tc, dtc, oinf = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
            tc[:] = np.nan; dtc[:] = np.nan; oinf[:] = np.nan
            for (i, p1), (j, p2) in product(enumerate(ps), enumerate(ps)):
                if p1 + p2 < 1:
                    ptot = [p1, p2, 1-p1-p2]
                    data = self.get_samps(case, p=ptot)
                    CD = am.OInformation(data)
                    CD.get_oinformation([0, 1, 2])
                    tc[i, j] = CD.totcorr
                    dtc[i, j] = CD.dtotcorr
                    oinf[i, j] = CD.o_inf
            sns.heatmap(tc, vmin=-1.5, vmax=1.5, cbar=False, ax=axs[0, col], cmap='Spectral', center=0)
            sns.heatmap(dtc, vmin=-1.5, vmax=1.5, cbar=False, ax=axs[1, col], cmap='Spectral', center=0)
            sns.heatmap(oinf, vmin=-1.5, vmax=1.5, cbar=False, ax=axs[2, col], cmap='Spectral', center=0)
            for k in range(3):
                axs[k, col].invert_yaxis()
                axs[k, col].set_xlabel('p(-1)')
                axs[k, col].set_ylabel('p(1)')

            axs[0, col].set_title(case)

        fig.subplots_adjust(right=0.9)
        cbar_t = fig.add_axes([0.93, 0.7, 0.02, 0.2])
        cbar_d = fig.add_axes([0.93, 0.37, 0.02, 0.2])
        cbar_o = fig.add_axes([0.93, 0.04, 0.02, 0.2])

        sns.heatmap(tc, vmin=-1.5, vmax=1.5, cbar=True, ax=axs[0, col], cmap='Spectral', center=0, cbar_ax=cbar_t)
        sns.heatmap(dtc, vmin=-1.5, vmax=1.5, cbar=True, ax=axs[1, col], cmap='Spectral', center=0, cbar_ax=cbar_d)
        sns.heatmap(oinf, vmin=-1.5, vmax=1.5, cbar=True, ax=axs[2, col], cmap='Spectral', center=0, cbar_ax=cbar_o)

        for k in range(3):
            axs[k, col].invert_yaxis()
            axs[k, col].set_xlabel('p(-1)')
            axs[k, col].set_ylabel('p(1)')

        axs[0, 0].set_ylabel('Total Corr.')
        axs[1, 0].set_ylabel('Dual Total Corr.')
        axs[2, 0].set_ylabel('O-Information')

        fig.savefig('../plots/eval_oinf_hm.pdf')


class EvalLatent(object):
    def __init__(self):
        self.SD = sd.SampleData()

    def get_samps(self, N=1000, p_deps=[.8, .6, .4], n_indeps=0, indep_p=[1/2, 1/2, 0], latent_p=[1/3, 1/3, 1/3]):
        data = []
        g = self.sample_indep(N, latent_p)

        for p_dep in p_deps:
            x = self.sample_cond(g, pdep=pdep)
            data.append(x)
        for _ in range(n_ideps):
            x = self.sample_indep(N, indep_p)
            data.append(x)

        return np.array(data)


    def sample_cond(self, x, pdep=.5):
        y = [xi if np.random.random() < pdep else self.sample_indep(1)[0] for xi in x]
        return y


    def sample_indep(self, N, p):
        x = np.random.choice([1, -1, 0], size=N, p=p, replace=True)
        return x


    #TODO: framework for creating plots and seeing dependence of pdeps, etc




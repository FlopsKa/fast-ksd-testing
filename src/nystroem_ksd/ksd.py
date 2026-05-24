import numpy as np
import time
from scipy.stats import chi2, vonmises, vonmises_fisher
from sklearn.metrics import pairwise_distances
from lib import wynne

class StatTest:
    def test_statistic(self, X):
        pass

    def test_threshold(self, X):
        pass

    def test_H0(self,X):
        return self.test_statistic(X) > self.test_threshold(X)
    
    
class Rayleigh(StatTest):
    def __init__(self,level_alpha=.01):
        self.level_alpha = level_alpha

    def test_statistic(self,thetas):
        """According to A.1 in https://arxiv.org/pdf/2002.06843"""
        return 2/len(thetas)*(np.power(np.sum(np.cos(thetas)),2) + np.power(np.sum(np.sin(thetas)),2))

    def test_threshold(self, X=None):
        return chi2.ppf([1-self.level_alpha], df=2)
    
    
class TestBenchmark():
    def __init__(self, stat_test):
        self.stat_test = stat_test

    def error(self, n, data_gen, repetitions):
        rejects = []
        for _ in range(repetitions):
            X = data_gen(n)
            rejects += [self.stat_test.test_statistic(X) > self.stat_test.test_threshold(X)]
        return np.mean(rejects)
    
def cot(phi):
        return 1/np.tan(phi)
    
class DirectionalKSD(StatTest):
    def __init__(self, kappa=1, d=2, level_alpha=.01, rng=np.random.default_rng()):
        self.kappa=kappa
        self.d=d
        self.level_alpha=level_alpha
        self.rng = rng

    def h_p(self, X, Y=None):
        if self.d==2:
            diffs = pairwise_distances(X, Y, metric=lambda x,y : x-y)
            return self.kappa*(np.cos(diffs) - self.kappa*np.sin(diffs)**2)*np.exp(self.kappa*np.cos(diffs))
        if self.d==3:
            return pairwise_distances(X, Y, metric=lambda x,y : DirectionalKSD._h_p_3d(x,y,kappa=self.kappa))

    def _h_p_3d(t,vt,kappa=1):
        """Computes the Stein kernel for a uniform target on the unit sphere in three dimensions in spherical coordinates."""
        line1 = np.exp(kappa*(np.cos(t[0])*np.cos(vt[0]) + np.sin(t[0])*np.sin(vt[0])*np.cos(t[1]-vt[1])))
        line2 = -kappa**2*np.cos(t[0])**2*np.sin(vt[0])**2*np.cos(t[1]-vt[1])
        line3 = kappa*np.sin(t[0])*( -kappa*np.sin(t[0])*np.cos(vt[0])**2*np.cos(t[1]-vt[1]) + np.sin(vt[0])*(-kappa*np.sin(t[0])*np.sin(vt[0])*np.sin(t[1]-vt[1])**2+np.cos(t[1]-vt[1])+1)-np.cos(vt[0])*cot(vt[0]) )
        line4 = kappa*np.cos(t[0])*(1/4*kappa*np.sin(t[0])*np.sin(2*vt[0])*(np.cos(2*(t[1]-vt[1])) + 3) +3*np.cos(vt[0])*np.cos(t[1]-vt[1])-cot(t[0])*np.sin(vt[0])) + cot(t[0])*cot(vt[0])
        return line1*(line2 + line3 + line4)
        
    def test_statistic(self, X):
        return np.mean(self.h_p(X))

    def test_threshold(self, X, B=1000):
        #import pdb; pdb.set_trace()
        n = len(X)
        gram = self.h_p(X)
        acc = np.zeros(B)
        for i in range(B):
            W = self.rng.binomial(1,0.5,size=(n,1))*2-1
            acc[i] = (W.T@gram@W)[0,0]
        return np.quantile(acc*1/n**2,1-self.level_alpha)
    
class InfiniteDimKSD(StatTest):
    def __init__(self, cov, T, gamma, kernel_type="SE", level_alpha=.01, rng=np.random.default_rng()):
        self.infinite_ksd = wynne.KSD(C=cov,T=T,kernel_type=kernel_type, gamma=gamma)
        self.level_alpha=level_alpha
        self.rng = rng

    def h_p(self, X, Y=None):
        return self.infinite_ksd.__call__(X.T,Y.T) if Y is not None else self.infinite_ksd.__call__(X.T,X.T)
        
    def test_statistic(self, X):
        return np.mean(self.h_p(X))

    def test_threshold(self, X, B=1000):
        n = len(X)
        gram = self.h_p(X)
        acc = np.zeros(B)
        for i in range(B):
            W = self.rng.binomial(1,0.5,size=(n,1))*2-1
            acc[i] = (W.T@gram@W)[0,0]
        return np.quantile(acc*1/n**2,1-self.level_alpha)

    def median_heuristic(X, T):
        sqr_dist_mat = wynne.form_distance_mat(X.T,X.T,X.T,X.T,T,T)
        return np.sqrt(np.median(sqr_dist_mat[sqr_dist_mat > 0]))

class NystroemInfiniteDimKSD(InfiniteDimKSD):
    def __init__(self, cov, T, gamma, kernel_type="SE", level_alpha=.01, nystroem_samples_func=lambda n : int(np.sqrt(n)), rng=np.random.default_rng()):
        self.infinite_ksd = wynne.KSD(C=cov,T=T,kernel_type=kernel_type, gamma=gamma)
        self.level_alpha=level_alpha
        self.nystroem_samples_func = nystroem_samples_func
        self.rng = rng
        
    def test_statistic(self, X):
        n = len(X)
        idx = self.rng.choice(n,size=self.nystroem_samples_func(n),replace=True)

        H_mn = self.h_p(X[idx],X)
        H_mm_inv = np.linalg.pinv(H_mn[:,idx],hermitian=True)
        beta = H_mn@np.ones((n,1))/n

        return (beta.T@H_mm_inv@beta)[0,0]

    def test_threshold(self, X, B=1000):
        n = len(X)
        idx = self.rng.choice(n,size=self.nystroem_samples_func(n),replace=True)

        H_mn = self.h_p(X[idx],X)
        H_mm_inv = np.linalg.pinv(H_mn[:,idx],hermitian=True)

        acc = np.zeros(B)
        for i in range(B):
            W = self.rng.binomial(1,0.5,size=(n,1))*2-1
            acc[i] = (W.T@H_mn.T@H_mm_inv@H_mn@W)[0,0]
        return np.quantile(acc*1/n**2,1-self.level_alpha)




class NystroemDirectionalKSD(DirectionalKSD):
    def __init__(self, kappa=1, d=2, level_alpha=.01, nystroem_samples_func=lambda n : int(np.sqrt(n)), rng=np.random.default_rng()):
        self.kappa = kappa
        self.d = d
        self.level_alpha = level_alpha
        self.nystroem_samples_func = nystroem_samples_func
        self.rng = rng
        
    def test_statistic(self, X):
        n = len(X)
        idx = self.rng.choice(n,size=self.nystroem_samples_func(n),replace=True)

        H_mn = self.h_p(X[idx],X)
        H_mm_inv = np.linalg.pinv(H_mn[:,idx],hermitian=True)
        beta = H_mn@np.ones((n,1))/n
                
        return (beta.T@H_mm_inv@beta)[0,0]

    def test_threshold(self, X, B=1000):
        n = len(X)
        idx = self.rng.choice(n,size=self.nystroem_samples_func(n),replace=True)

        H_mn = self.h_p(X[idx],X)
        H_mm_inv = np.linalg.pinv(H_mn[:,idx],hermitian=True)

        acc = np.zeros(B)
        for i in range(B):
            W = self.rng.binomial(1,0.5,size=(n,1))*2-1
            acc[i] = (W.T@H_mn.T@H_mm_inv@H_mn@W)[0,0]
        return np.quantile(acc*1/n**2,1-self.level_alpha)


class ContextTimer:
    """
    A class used to time an execution of a code snippet. 
    Use it with with .... as ...
    For example, 

        with ContextTimer() as t:
            # do something 
        time_spent = t.secs

    From https://www.huyng.com/posts/python-performance-analysis
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start 
        if self.verbose:
            print('elapsed time: %f ms' % (self.secs*1000))


class Data:
    def __init__(self, rng=np.random.default_rng()):
        self.rng = rng

    def gen(self,n):
        pass

class UniformCircle(Data):
    def gen(self,n):
        return self.rng.uniform(0,2*np.pi,size=(n,1))
    
def to_angles(xyz):
    """Assumes the data lies on a three-dimensional unit sphere."""
    def atan2(x,y):
        return np.arctan2(y,x)
    ret = np.zeros((len(xyz),2))
    ret[:,0] =  atan2(np.sqrt(xyz[:,2]**2 + xyz[:,1]**2),xyz[:,0])
    ret[:,1] = atan2(xyz[:,2],xyz[:,1])
    return ret

class UniformSphere3d(Data):
    """Generates uniformly distributed data on the unit sphere in three dimensions. Returns spherical coordinates."""    
    def gen(self,n):
        dat = self.rng.normal(size=(n,3))
        return to_angles(dat / np.linalg.norm(dat,axis=1,keepdims=True))

class VonMises(Data):
    def gen(self,n):
        return vonmises.rvs(kappa=0.5,size=(n,1),random_state=self.rng)
    
class VonMisesFisher3d(Data):
    def __init__(self, kappa=1, rng=np.random.default_rng()):
        self.kappa=kappa
        self.rng=rng

    def gen(self,n):
        return to_angles(vonmises_fisher([1,0,0], kappa=self.kappa).rvs(n, random_state=self.rng))
    

class BM_clip(Data):
    def __init__(self, n_freqs, clip_freq, rng=np.random.default_rng()):
        self.n_freqs = n_freqs
        self.clip_freq = clip_freq
        self.rng = rng

    def gen(self, n):
        return wynne.BM_clip(N=n,n_freqs = self.n_freqs,clip_freq = self.clip_freq,rng = self.rng).T
    
class CA_freqs(Data):
    def __init__(self, n_freqs, a_1, a_2, a_3, rng=np.random.default_rng()):
        self.n_freqs = n_freqs
        self.a_1 = a_1
        self.a_2 = a_2
        self.a_3 = a_3
        self.rng = rng
    
    def gen(self, n):
        return wynne.CA_freqs_sampler(N=n,n_freqs = self.n_freqs,a_1=self.a_1,a_2=self.a_2,a_3=self.a_3,rng=self.rng).T
    

import seaborn as sns

def get_sns_conf(algorithms):
    """

    Usage example
    =============

    algorithms = ["Algorithm 1", "Algorithm 2", "Algorithm 3"]
    labels = algorithms
    sns_conf = get_sns_conf(algorithms=algorithms)

    df = df.sort_values("Algorithm")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2.3))

    sns.lineplot(data=df, x="n", y="Estimate", hue="Algorithm", ax=ax1, style="Algorithm", **sns_conf)

    line, _ = ax1.get_legend_handles_labels()
    fig.legend(line, labels, loc="upper center", bbox_to_anchor=(0.53, 1.13), ncol=3)

    """
    possible_markers = ["*", "d", "X", "P", "o", "v", "^", "p", ">", "<"]
    n = min(len(algorithms),len(possible_markers))

    palette_sns = sns.color_palette("Set1", n_colors=n)
    palette_sns = [palette_sns[i] for i in range(n)]

    markers = { k:v for k,v in  zip(algorithms[:n], possible_markers)}
    palette = { k:v for k,v in zip(algorithms, palette_sns) }
    return { "palette" : palette, "markers" : markers, "hue_order" : algorithms }
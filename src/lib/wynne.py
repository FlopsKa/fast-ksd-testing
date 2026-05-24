import numpy as np

class KSD:
    """
    Description:
        Class to represent an instance of kernel Stein discrepancy.
        Has methods to ouput the Stein kernel evaluated on data
    """  
    def __init__(self,C,T,DU = 0,kernel_type = "SE", gamma = -1):
        """
        Arg:
            C: (d,d) matrix representing the covariance operator
            T: (d,d) matrix representing the hyperparameter
            DU: Function for the DU term in KSD. Default is DU = 0 which makes the DU term be 0.
            kernel_type: either "SE" or "IMQ"
            gamma: lengthscale, if -1 then median heuristic is employed
        """  
        self.C = C
        self.T = T
        self.DU = DU
        self.kernel_type = kernel_type
        self.gamma = gamma
        

    def __call__(self, x, y):
        """
        Arg:
            x: (d,n) data matrix
            y: (d,m) data matrix
        Return:
            Stein_mat: (n,m) matrix with ij-th entry the Stein kernel h evaluated at x_i,y_j 
        """

        n = np.shape(x)[1]
        m = np.shape(y)[1] 
        
        sqr_dist_mat = form_distance_mat(x,y,x,y,self.T,self.T)
            
        # median heuristic
        if self.gamma == -1:
            self.gamma = np.sqrt(np.median(sqr_dist_mat[sqr_dist_mat > 0]))
            #print(self.gamma)
            # changes the T which will be used later
            self.T = self.T/self.gamma
            # renormalises the squared distance matrix already computed that'll be used later
            sqr_dist_mat = sqr_dist_mat/(self.gamma**2)
        
        # introduces variable S to make calculations easier
        S = self.C @ np.transpose(self.T) @ self.T
        # form the CDU terms
        if self.DU == 0:
            CDUx = np.zeros(np.shape(x))
            CDUy = np.zeros(np.shape(y))
        else:
            CDUx = self.C @ self.DU(x)
            CDUy = self.C @ self.DU(y)
        
        # I<x + CDU(x),y+CDU(y)> term
        term1 = np.einsum("ji,jk -> ik",x+CDUx,y + CDUy)
        # - <S(x-y),x-y> term
        term2 = -1 * form_distance_mat(x,y,x,y,S)
        # - <S(x-y),CDU(x)-CDU(y)> term
        term3 = -1 * form_distance_mat(x,y,CDUx,CDUy,S)
        # Tr(SC) term
        term4 = np.trace(S @ self.C)
        # ||S(x-y)||^2 term
        term5 = -1 * form_distance_mat(x,y,x,y,S,S)
    
        # calculations are taken from example of Stein kernels in paper associated with SE and IMQ base kernels
        if self.kernel_type == "SE":
            
            SE_mat = np.exp(-0.5 * sqr_dist_mat)
            
            Stein_mat = SE_mat * (term1 + term2 + term3 + term4 + term5)
            
            return Stein_mat
        
        if self.kernel_type == "IMQ":
            
            IMQ_mat = (sqr_dist_mat + 1)**(-0.5)
            
            Stein_mat = (term1 * IMQ_mat)  + ((term2 + term3 + term4) * (IMQ_mat**3)) + (3 * term5 * (IMQ_mat**5))
            
            return Stein_mat
        
class GoodnessOfFitTest:
    """
    Description: A single goodness-of-fit test which can produce a p-value given data and a KSD object
    """
    def __init__(self, discrepancy, x):
        """
        Args:
            discrepancy: A callable that returns a matrix of Stein kernel evaluations
            x: (d,n) matrix of data
        """
        self.d = discrepancy
        self.x = x
        self.n = x.shape[1]
        

    def compute_pvalue(self, n_bootstrap):
        """
        Arg:
            n_bootstrap: Number of bootstrap samples.
        Return:
            bootstrap_stats: bootstraped test statistics
            test_stat: the test statistic calculated using observed data
            pvalue: p-value based on comparing test_stat with bootstrap_stats
        """
        # Form the test statistic from evaluations of the Stein kernel
        stein_matrix = self.d(self.x, self.x)
        u_matrix = stein_matrix - np.diag(np.diag(stein_matrix))
        test_stat = u_matrix.sum() / self.n / (self.n-1)
        
        # Obtain bootstrap samples using multi-nomial distribution
        bootstrap_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            W = np.random.multinomial(self.n,(1./self.n)*np.ones(self.n))
            W = (W-1)/self.n
            bootstrap_stats[i] = W @ u_matrix @ W
        
        # Calculate p-value
        pvalue = (bootstrap_stats > test_stat).mean()

        return (bootstrap_stats, test_stat, pvalue)
    

def form_distance_mat(x1,y1,x2,y2,A,B = None):
    """
    Description:
        Forms a distance matrix with ij-th entry <A(x1_i-y1_j),B(x2_i-y2_j)> where <.,.> is Euclidean inner product
        and x_i = x[:,i] is i-th column of data, analogous for y_i. 
        If B = None then B becomes the identity.
    Arg:
        x1,x2: (d,n) matrix, data in columns
        y1,y2: (d,m) matrix, data in columns
        A: (d,d) matrix
        B: (d,d) matrix
    Return:
        dist_mat: (n,m) matrix with ij-th entry <A(x1_i - y1_j), B(x2_i - y2_j)> 
                  where x1_i = x[:,i] is i-th data column, analogous for ,x2_i,y1_j,y2_j
    """  
    d = x1.shape[0]
    n = x1.shape[1]
    m = y1.shape[1]
    
    if (B is None) or (B.all() is None):
        B = np.eye(d)
    
    mat_x1x2 = np.einsum("ji,ji -> i", A @ x1, B @ x2)
    mat_x1x2 = np.reshape(mat_x1x2,(n,1))
    mat_x1x2 = np.tile(mat_x1x2,(1,m))
    
    mat_y1y2 = np.einsum("ji,ji -> i", A @ y1, B @ y2)
    mat_y1y2 = np.reshape(mat_y1y2,(1,m))
    mat_y1y2 = np.tile(mat_y1y2,(n,1))
    
    mat_x1y2 = np.einsum("ji,jk -> ik",A @ x1, B @ y2)
    mat_y1x2 = np.einsum("jk,ji -> ik",A @ y1, B @ x2)
    
    dist_mat = mat_x1x2 + mat_y1y2 - mat_x1y2 - mat_y1x2
    
    return dist_mat

# Brownian motion basis used to project data onto
def BM_basis(n_freqs,obs):
    X = np.zeros((n_freqs,len(obs)))
    for i in range(1,n_freqs+1):
        X[i-1,:] = np.sqrt(2)*np.sin((i-0.5)*np.pi*obs)
    return X

# Generates Ornstein-Uhlenbeck trajectories
def OU_sampler(N,grid_size,theta,mu=5,rng = np.random.default_rng()):
    dt = 1/grid_size
    X = np.zeros((N,grid_size))
    noise = rng.standard_normal(size=(N,grid_size))*np.sqrt(dt)
    for i in range(1,grid_size):
        X[:,i] = X[:,i-1] + theta*(mu-X[:,i-1])*dt + noise[:,i]
    return X

# Generates Brownian motion clipped to certain a frequency
# Since the samples are computed using random variables against BM basis elements
# and we only use the coefficients in the computation of KSD, we can simulate 
# this data by simply simulating the random variable coefficients
def BM_clip(N,n_freqs,clip_freq,grid_size = 100,rng = np.random.default_rng()):
    C = np.zeros(n_freqs)
    lambda_diag = np.array([1/(np.pi * (n-0.5))**2 for n in range(1,clip_freq + 1)])
    C[:clip_freq] = lambda_diag
    coefs = rng.multivariate_normal(mean = np.zeros(n_freqs),cov = np.diag(C),size = N)
    return coefs.T

# Generates OU trajectories projected to a specified number of frequencies of Brownian motion basis
def OU_freqs_sampler(N,n_freqs,theta,mu=5,sig=1,random_state = None):
    grid_size = 100
    obs = np.linspace(0,1,grid_size,endpoint=True)
    basis = BM_basis(n_freqs,obs)
    OU_vals = OU_sampler(N,grid_size,theta,mu,random_state)
    return (1/grid_size)*np.dot(OU_vals,basis.T).T

# Generates samples from the referenced Cuesta-Albertos et al 2007 paper
def CA_sampler(N,grid_size,a_1,a_2,a_3,rng = None):
    BM_arr = OU_sampler(N,grid_size,theta=0,mu=0,rng=rng)
    obs = np.linspace(0,1,grid_size,endpoint=False)
    det_arr = 1 + a_1*(obs**2) + a_2*np.sin(2*np.pi*obs) + a_3*np.exp(obs)
    return BM_arr * det_arr

# Generates trajectories from CA_sampler projected to a specified number of frequencies of Brownian motion basis
def CA_freqs_sampler(N,n_freqs,a_1,a_2,a_3,rng = None):
    grid_size = 100
    basis = BM_basis(n_freqs,np.linspace(0,1,grid_size,endpoint=False))
    AC_vals = CA_sampler(N,grid_size,a_1,a_2,a_3,rng = rng)
    return (1/grid_size)*np.dot(AC_vals,basis.T).T

# Generates trajectories from Ditzhaus and Gaigall 2018 referenced paper projected to a specified number of frequencies of Brownian motion basis
def Ditzhaus_freqs_sampler(N,n_freqs=100,a=1,b=0,random_state = None):
    grid_size = 100
    X = a*OU_sampler(N,grid_size,theta = 0,random_state=random_state)
    obs = np.linspace(0,1,grid_size,endpoint=False)
    X += b*obs*(obs-1)
    basis = BM_basis(n_freqs,np.linspace(0,1,grid_size,endpoint=False))
    return (1/grid_size)*np.dot(X,basis.T).T
alswr_params = {
    "n_iters":15, "n_factors":15, "alpha":15., "reg":0.01,
}

import numpy as np
from tqdm import trange

##https://ethen8181.github.io/machine-learning/recsys/2_implicit.html#Implementation
class ALSWR:
    """
    Alternating Least Squares with Weighted Regularization
    for implicit feedback

    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm

    n_factors : int
        number/dimension of user and item latent factors

    alpha : int
        scaling factor that indicates the level of confidence in preference

    reg : int
        regularization term for the user and item latent factors

    seed : int
        seed for the randomly initialized user, item latent factors

    Reference
    ---------
    Y. Hu, Y. Koren, C. Volinsky Collaborative Filtering for Implicit Feedback Datasets
    http://yifanhu.net/PUB/cf.pdf
    """
    def __init__(self, n_iters, n_factors, alpha, reg, seed):
        self.reg = reg
        self.seed = seed
        self.alpha = alpha
        self.n_iters = n_iters
        self.n_factors = n_factors
    
    def fit(self, ratings):
        """
        ratings : scipy sparse csr_matrix [n_users, n_items]
            sparse matrix of user-item interactions
        """        
        # the original confidence vectors should include a + 1,
        # but this direct addition is not allowed when using sparse matrix,
        # thus we'll have to deal with this later in the computation
        Cui = ratings.copy().tocsr()
        Cui.data *= self.alpha
        Ciu = Cui.T.tocsr()
        self.n_users, self.n_items = Cui.shape
        
        # initialize user latent factors and item latent factors
        # randomly with a specified set seed
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size = (self.n_users, self.n_factors))
        self.item_factors = rstate.normal(size = (self.n_items, self.n_factors))
        
        for _ in trange(self.n_iters, desc = 'training progress'):
            self._als_step(Cui, self.user_factors, self.item_factors)
            self._als_step(Ciu, self.item_factors, self.user_factors)  
        
        return self
    
    def _als_step(self, Cui, X, Y):
        """
        when solving the user latent vectors,
        the item vectors will be fixed and vice versa
        """
        # the variable name follows the notation when holding
        # the item vector Y constant and solving for user vector X
        
        # YtY is a d * d matrix that is computed
        # independently of each user
        YtY = Y.T.dot(Y)
        data = Cui.data
        indptr, indices = Cui.indptr, Cui.indices

        # for every user build up A and b then solve for Ax = b,
        # this for loop is the bottleneck and can be easily parallized
        # as each users' computation is independent of one another
        for u in range(self.n_users):
            # initialize a new A and b for every user
            b = np.zeros(self.n_factors)
            A = YtY + self.reg * np.eye(self.n_factors)
            
            for index in range(indptr[u], indptr[u + 1]):
                # indices[index] stores non-zero positions for a given row
                # data[index] stores corresponding confidence,
                # we also add 1 to the confidence, since we did not 
                # do it in the beginning, when we were to give every 
                # user-item pair and minimal confidence
                i = indices[index]
                confidence = data[index] + 1
                factor = Y[i]

                # for b, Y^T C^u p_u
                # there should be a times 1 for the preference 
                # Pui = 1
                # b += confidence * Y[i] * Pui
                # but the times 1 can be dropped
                b += confidence * factor
                
                # for A, Y^T (C^u - I) Y
                A += (confidence - 1) * np.outer(factor, factor)

            X[u] = np.linalg.solve(A, b)
        
        return self

    def predict(self):
        """predict ratings for every user and item"""
        prediction = self.user_factors.dot(self.item_factors.T)
        return prediction
    
    def _predict_user(self, user):
        """
        returns the predicted ratings for the specified user,
        this is mainly used in computing evaluation metric
        """
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        return user_pred
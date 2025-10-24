import numpy as np
import random
from scipy.sparse import coo_array, csr_array, csr_matrix
from subprocess import call
import calendar
import time

from preprocessing import CustomScaler

import AlternatingLeastSquares
import BayesianPairwiseRanking
from LogisticMatrixFactorization import LogisticMF as CoreLogisticMF

current_GMT = time.gmtime()

# Basic Model

class BasicModel(object):
    '''
    A class used to encode a drug repurposing model

    ...

    Parameters
    ----------
    params : dict
        dictionary which contains method-wise parameters

    Attributes
    ----------
    name : str
        the name of the model
    model : depends on the implemented method
        may contain an instance of a class of sklearn classifiers
    ...
        other attributes might be present depending on the type of model

    Methods
    -------
    __init__(params)
        Initializes the model with preselected parameters
    fit(train_dataset, seed=1234)
        Preprocesses and fits the model 
    predict_proba(test_dataset)
        Outputs properly formatted predictions of the fitted model on test_dataset
    predict(scores)
        Applies the following decision rule: if score<threshold, then return the negative label, otherwise return the positive label
    recommend_k_pairs(dataset, k=1, threshold=None)
        Outputs the top-k (item, user) candidates (or candidates which score is higher than a threshold) in the input dataset
    print_scores(scores)
        Prints out information about scores
    print_classification(predictions)
        Prints out information about predicted labels
    preprocessing(train_dataset) [not implemented in BasicModel]
        Preprocess the input dataset into something that is an input to the self.model_fit if it exists
    model_fit(train_dataset) [not implemented in BasicModel]
        Fits the model on train_dataset
    model_predict_proba(test_dataset) [not implemented in BasicModel]
        Outputs predictions of the fitted model on test_dataset
    '''
    def __init__(self, params):
        '''
        Creates an instance of stanscofi.BasicModel

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains method-wise parameters
        '''
        self.name = "Model"
        for param in params:
            setattr(self, param, params[param])

    def fit(self, train_dataset, seed=1234):
        '''
        Fitting the model on the training dataset.

        Not implemented in the BasicModel class.

        ...

        Parameters
        ----------
        train_dataset : stanscofi.Dataset
            training dataset on which the model should fit
        seed : int (default: 1234)
            random seed
        '''
        np.random.seed(seed)
        random.seed(seed)
        self.model_fit(*self.preprocessing(train_dataset, is_training=True))

    def predict_proba(self, test_dataset, default_zero_val=1e-31):
        '''
        Outputs properly formatted scores (not necessarily in [0,1]!) from the fitted model on test_dataset. Internally calls model_predict() then reformats the scores

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            dataset on which predictions should be made

        Returns
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format, with nonzero values corresponding to predictions on available pairs in the dataset
        '''
        scores = self.model_predict_proba(*self.preprocessing(test_dataset, is_training=False))
        if ((scores!=0).any()):
            default_val = min(default_zero_val, np.min(scores[scores!=0])/2)
        else:
            default_val = default_zero_val
        #print(("folds",test_dataset.folds.data.shape[0]))
        if (scores.shape==test_dataset.folds.shape):
            scores[(scores==0)&(test_dataset.folds.toarray()==1)] = default_val ## avoid removing these zeroes
            scores = coo_array(scores)
            scores = scores*test_dataset.folds
            #print(("scores",scores.data.shape[0]))
            return coo_array(scores)
        assert scores.shape[0]==test_dataset.folds.data.shape[0]
        scores[(scores==0)&(test_dataset.folds.data==1)] = default_val ## avoid removing these zeroes 
        scores_arr = coo_array((scores, (test_dataset.folds.row, test_dataset.folds.col)), shape=test_dataset.folds.shape)
        #print(("scores",scores.data.shape[0]))
        return scores_arr

    def predict(self, scores, threshold=0.5):
        '''
        Outputs class labels based on the scores, using the following formula
            prediction = -1 if (score<threshold) else 1

        ...

        Parameters
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format
        threshold : float
            the threshold of classification into the positive class

        Returns
        ----------
        predictions : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format with values in {-1,1}
        '''
        #print(("scores-preds",scores.data.shape[0]))
        preds = coo_array((scores.toarray()!=0).astype(int)*((-1)**(scores.toarray()<=threshold)))
        #print(('preds',preds.data.shape[0]))
        return preds
    
    def recommend_k_pairs(self, dataset, k=1, threshold=None):
        '''
        Outputs the top-k (item, user) candidates (or candidates which score is higher than a threshold) in the input dataset.

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset on which predictions should be made
        k : int or None (default: 1)
            number of pair candidates to return
        threshold : float or None (default: 0)
            threshold on candidate scores. If k is not None, k best candidates are returned independently of the value of threshold

        Returns
        -------
        candidates : list of tuples of size 3
            list of (item, user, score) candidates
        '''
        import numpy as np

        assert (k is None) or (k > 0)
        
        # Step 1: Predict scores
        scores = self.predict_proba(dataset)
        score_matrix = scores.toarray()
        
        # Step 2: Determine which pairs to return
        if k is not None:
            # Flatten scores and get indices of top-k scores
            flat_scores = score_matrix.ravel()
            if k >= len(flat_scores):
                topk_idx = np.arange(len(flat_scores))
            else:
                topk_idx = np.argpartition(-flat_scores, k)[:k]
                topk_idx = topk_idx[np.argsort(-flat_scores[topk_idx])]  # sort top-k

            # Convert flat indices to (i,j) pairs
            ids_list = [np.unravel_index(idx, score_matrix.shape) for idx in topk_idx]

        elif threshold is not None:
            # Select all scores above the threshold
            ids_list = np.argwhere(score_matrix >= threshold).tolist()
            if k is not None:
                # Optionally limit to top-k after threshold filtering
                ids_list.sort(key=lambda x: -score_matrix[x[0], x[1]])
                ids_list = ids_list[:k]

        else:
            raise ValueError("Either k must be provided or threshold must be set.")

        # Step 3: Build candidate list
        candidates = [
            [dataset.item_list[i], dataset.user_list[j], score_matrix[i, j]]
            for i, j in ids_list
        ]

        return candidates

    def recommend_for_user(self, dataset, user_id, top_n=10):
        import numpy as np
        
        # Step 0: Find index of the user in dataset
        if user_id not in dataset.user_list:
            raise ValueError(f"User {user_id} not found in dataset")
        user_idx = dataset.user_list.index(user_id)
        
        # Step 1: Get predicted scores for all items
        scores = self.predict_proba(dataset)       # shape: (num_items, num_users)
        score_matrix = scores.toarray()
        
        # Step 2: Extract the column corresponding to this user
        user_scores = score_matrix[:, user_idx]    # shape: (num_items,)
        
        # Step 3: Get indices of top-N items
        if top_n >= len(user_scores):
            top_indices = np.arange(len(user_scores))
        else:
            top_indices = np.argpartition(-user_scores, top_n)[:top_n]
            top_indices = top_indices[np.argsort(-user_scores[top_indices])]
        
        # Step 4: Map indices to item IDs
        recommendations = [(dataset.item_list[i], user_scores[i]) for i in top_indices]
        
        return recommendations

    def print_scores(self, scores):
        '''
        Prints out information about the scores

        ...

        Parameters
        ----------
        scores : COO-array
            sparse matrix in COOrdinate format
        '''
        print("* Scores")
        print("%d unique items, %d unique users" % (len(np.unique(scores.row)), len(np.unique(scores.col))))
        print("Scores: Min: %f\tMean: %f\tMedian: %f\tMax: %f\tStd: %f\n" % tuple([f(scores.data) for f in [np.min,np.mean,np.median,np.max,np.std]]))

    def print_classification(self, predictions):
        '''
        Prints out information about the predicted classes

        ...

        Parameters
        ----------
        predictions : COO-array
            sparse matrix in COOrdinate format
        '''
        print("* Classification")
        print("%d unique items, %d unique users" % (len(np.unique(predictions.row)), len(np.unique(predictions.col))))
        print("Positive class: %d, Negative class: %d\n" % ((csr_array(predictions)==1).sum(), (csr_array(predictions)==-1).sum()))

    def preprocessing(self, dataset, is_training=True):
        '''
        Preprocessing step, which converts elements of a dataset (ratings matrix, user feature matrix, item feature matrix) into appropriate inputs to the classifier (e.g., X feature matrix for each (user, item) pair, y response vector).

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert
        is_training : bool
            is the preprocessing prior to training (true) or testing (false)?

        Returns
        ----------
        ... : ...
            appropriate inputs to the classifier (vary across algorithms)
        '''
        raise NotImplemented

    def model_fit(self):
        '''
        Fitting the model on the training dataset.

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        ... : ...
            appropriate inputs to the classifier (vary across algorithms)
        '''
        raise NotImplemented

    def model_predict_proba(self):
        '''
        Making predictions using the model on the testing dataset.

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        ... : ...
            appropriate inputs to the classifier (vary across algorithms)
        ...

        Returns
        ----------
        scores : array_like of shape (n_items, n_users)
            prediction values by the model
        '''
        raise NotImplemented

### Alternating Least Squares
class ALSWR(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(ALSWR, self).__init__(params)
        self.name = "ALSWR"
        self.model = AlternatingLeastSquares.ALSWR(**{(k if (k!="random_state") else "seed"):params[k] for k in params})

    def default_parameters(self):
        params = AlternatingLeastSquares.alswr_params
        params.update({"random_state": 1354})
        return params

    def preprocessing(self, dataset, is_training=True):
        ## users x items: only 1's and -1's
        if (is_training):
            ratings = csr_matrix((dataset.ratings.data, (dataset.ratings.col, dataset.ratings.row)), shape=dataset.ratings.T.shape)
        else:
            ids = np.argwhere(np.ones(dataset.ratings.shape))
            ratings = csr_matrix((dataset.ratings.toarray().ravel(), (ids[:,1].ravel(), ids[:,0].ravel())), shape=dataset.ratings.T.shape)
        return [ratings]
        
    def model_fit(self, X_train):
        np.random.seed(self.random_state)
        self.model.fit(X_train)

    def model_predict_proba(self, X_test):
        preds = self.model.predict().T
        return preds

### Bayesian Pairwise Ranking
class PMF(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(PMF, self).__init__(params)
        self.name = "PMF"
        self.estimator = BayesianPairwiseRanking.BPR(**params)

    def default_parameters(self):
        params = BayesianPairwiseRanking.bpr_params
        return params

    def preprocessing(self, dataset, is_training=True):
        ## Use 0-1 ratings (matrix form)
        return [csr_matrix(dataset.ratings.toarray().T)]
        
    def model_fit(self, X):
        self.estimator.fit(X)

    def model_predict_proba(self, X):
        preds = self.estimator.predict().T
        return preds

### Logistic Matrix Factorization
class LogisticMF(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(LogisticMF, self).__init__(params)
        self.name = "LogisticMF"
        self.estimator = CoreLogisticMF(**params)

    def default_parameters(self):
        params = {"counts": np.zeros((5,6)), "num_factors": 2, "reg_param":0.6, "gamma":1.0, "iterations":30}
        return params

    def preprocessing(self, dataset, is_training=True):
        counts = dataset.ratings.toarray().T
        counts[counts<1] = 0
        total = np.sum(counts)
        num_zeros = np.prod(counts.shape)-total
        alpha = num_zeros / total
        #print('alpha %.2f' % alpha)
        counts *= alpha
        return [counts]
        
    def model_fit(self, X):
        self.estimator.counts = X
        self.estimator.num_users = X.shape[0]
        self.estimator.num_items = X.shape[1]
        self.estimator.train_model()

    def model_predict_proba(self, X):
        ## prediction(i,j) = sigmoid(x_i . y_j^T + b_i + b_j)
        fx = self.estimator.user_vectors.dot(self.estimator.item_vectors.T)
        fx += np.tile(self.estimator.user_biases, (1,self.estimator.item_biases.shape[0]))
        fx += np.tile(self.estimator.item_biases, (1,self.estimator.user_biases.shape[0])).T
        preds = np.exp(fx)/(1+np.exp(fx))
        return preds.T

### Bounded Nuclear Norm Regularization: https://github.com/BioinformaticsCSU/BNNR/tree/68f8e98c02459189b6eeac68a86306ccc1da0374

# /!\ Only tested on Linux
class BNNR(BasicModel):
    def __init__(self, params=None):
        try:
            call("octave -v", shell=True)
        except:
            raise ValueError("Please install Octave.")
        params = params if (params is not None) else self.default_parameters()
        super(BNNR, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "BNNR" 
        self.estimator = None
        self.BNNR_filepath = None

    def default_parameters(self):
        params = {
            "maxiter": 300,
            "alpha": 1,
            "beta": 10,
            "tol1": 2e-3,
            "tol2": 1e-5,
        }
        return params

    def preprocessing(self, dataset, is_training=True, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.toarray().copy(), subset=None)
        S_ = np.nan_to_num(S_, nan=0.0) ##
        X_s = S_ if (S_.shape[0]==S_.shape[1]) else np.corrcoef(S_)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.toarray().copy(), subset=None)
        P_ = np.nan_to_num(P_, nan=0.0) ##
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        A_sp = dataset.ratings.toarray().T # users x items
        return [X_s, X_p, A_sp]
        
    def model_fit(self, X_s, X_p, A_sp):
        time_stamp = calendar.timegm(current_GMT)+np.random.choice(range(int(1e8)), size=1)[0]
        filefolder = "BNNR_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True)
        call("wget -qO %s/BNNR.m 'https://raw.githubusercontent.com/BioinformaticsCSU/BNNR/master/BNNR.m'" % filefolder, shell=True)
        call("wget -qO %s/svt.m 'https://raw.githubusercontent.com/BioinformaticsCSU/BNNR/master/svt.m'" % filefolder, shell=True)
        np.savetxt("%s/X_p.csv" % filefolder, X_p, delimiter=",")
        np.savetxt("%s/A_sp.csv" % filefolder, A_sp, delimiter=",")
        np.savetxt("%s/X_s.csv" % filefolder, X_s, delimiter=",")
        cmd = "Wdd = csvread('X_p.csv');Wdr = csvread('A_sp.csv');Wrr = csvread('X_s.csv');T = [Wrr, Wdr'; Wdr, Wdd];[WW,iter] = BNNR(%d, %d, T, double(T ~= 0), %f, %f, %d, 0, 1);[t1, t2] = size(T);[dn,dr] = size(Wdr);M_recovery = WW((t1-dn+1) : t1, 1 : dr);csvwrite('M_recovery.csv', M_recovery);csvwrite('iter.csv', iter);" % (self.alpha, self.beta, self.tol1, self.tol2, self.maxiter)
        call("cd %s/ && octave --silent --eval \"%s\"" % (filefolder, cmd), shell=True)
        self.estimator = {
            "niter" : int(np.loadtxt("%s/iter.csv" % filefolder, delimiter=",")),
            "predictions" : np.loadtxt("%s/M_recovery.csv" % filefolder, delimiter=",").T,
        }
        call("rm -rf %s/" % filefolder, shell=True)

    def model_predict_proba(self, X_s, X_p, A_sp):
        return self.estimator["predictions"]

    


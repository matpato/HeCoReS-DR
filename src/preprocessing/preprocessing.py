import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocessing_XY(dataset, preprocessing_str, operator="*", sep_feature="-", subset_=None, filter_=None, scalerS=None, scalerP=None, inf=2, njobs=1):
    '''
    Converts a score vector or a score value into a list of scores

    ...

    Parameters
    ----------
    dataset : stanscofi.datasets.Dataset
        dataset to preprocess
    preprocessing_str : str
        type of preprocessing: in ["Perlman_procedure","meanimputation_standardize","same_feature_preprocessing"]. 
    subset_ : None or int
        Number of features to restrict the dataset to (Top-subset_ features in terms of cross-sample variance) /!\ across user and item features if preprocessing_str!="meanimputation_standardize" otherwise 2*subset_ features are preserved (subset_ for item features, subset_ for user features)
    operator : None or str
        arithmetric operation to apply, ex. "+", "*"
    sep_feature : str
        separator between feature type and element in the feature matrices in dataset
    filter_ : None or list
        list of feature indices to keep (of length subset_) (overrides the subset_ parameter if both are fed)
    scalerS : None or stanscofi.models.CustomScaler instance
        scaler for items; the scaler fitted on item feature vectors
    scalerP : None or stanscofi.models.CustomScaler instance
        scaler for users; the scaler fitted on user feature vectors
    inf : float or int
        placeholder value for infinity values (positive : +inf, negative : -inf)
    njobs : int
        number of jobs to run in parallel (njobs > 0) for the Perlman procedure

    Returns
    ----------
    X : array-like of shape (n_folds, n_features)
        the feature matrix
    y : array-like of shape (n_folds, )
        the response/outcome vector
    scalerS : None or stanscofi.models.CustomScaler instance
        scaler for items; if the input value was None, returns the scaler fitted on item feature vectors
    scalerP : None or stanscofi.models.CustomScaler instance
        scaler for users; if the input value was None, returns the scaler fitted on user feature vectors
    filter_ : None or list
        list of feature indices to keep (of length subset_)
    '''
    assert njobs>0
    assert preprocessing_str in ["Perlman_procedure","meanimputation_standardize","same_feature_preprocessing"]
    if (preprocessing_str == "Perlman_procedure"):
        X, y = eval(preprocessing_str)(dataset, njobs=njobs, sep_feature=sep_feature, missing=-666, verbose=False)
        scalerS, scalerP = None, None
    if (preprocessing_str == "meanimputation_standardize"):
        X, y, scalerS, scalerP = eval(preprocessing_str)(dataset, subset=subset_, scalerS=scalerS, scalerP=scalerP, inf=inf, verbose=False)
    if (preprocessing_str == "same_feature_preprocessing"):
        X, y = eval(preprocessing_str)(dataset, operator)
        scalerS, scalerP = None, None
    if (preprocessing_str != "meanimputation_standardize"):
        if ((subset_ is not None) or (filter_ is not None)):
            if ((subset_ is not None) and (filter_ is None)):
                with np.errstate(over="ignore"):
                    x_vars = [np.nanvar(X[:,i]) if (np.sum(~np.isnan(X[:,i]))>0) else 0 for i in range(X.shape[1])]
                    x_vars = [x if (not np.isnan(x) and not np.isinf(x)) else 0 for x in x_vars]
                    x_ids_vars = np.argsort(x_vars).tolist()
                    features = x_ids_vars[-subset_:]
                    filter_ = features
            X = X[:,filter_]
    assert X.shape[0]==y.shape[0]==dataset.folds.data.shape[0]
    return X, y, scalerS, scalerP, filter_

class CustomScaler(object):
    '''
    A class used to encode a simple preprocessing pipeline for feature matrices. Does mean imputation for features, feature filtering, correction of infinity errors and standardization

    ...

    Parameters
    ----------
    posinf : int
        Value to replace infinity (positive) values
    neginf : int
        Value to replace infinity (negative) values

    Attributes
    ----------
    imputer : None or sklearn.impute.SimpleImputer instance
        Class for imputation of values
    scaler : None or sklearn.preprocessing.StandardScaler
        Class for standardization of values
    filter : None or list
        List of selected features (Top-N in terms of variance)

    Methods
    -------
    __init__(params)
        Initialize the scaler (with unfitted attributes)
    fit_transform(mat, subset=None, verbose=False)
        Fits classes and transforms a matrix
    '''
    def __init__(self, posinf, neginf):
        '''
        Initialize the scaler (with unfitted imputer, standardscaler and filter attributes)
        '''
        self.imputer = None
        self.scaler = None
        self.remove_nan = []
        self.filter = None
        self.posinf = None
        self.neginf = None

    def fit_transform(self, mat, subset=None, verbose=False): ## elements x features
        '''
        Fits each attribute of the scaler and transform a feature matrix. Does mean imputation for features, feature filtering, correction of infinity errors and standardization

        ...

        Parameters
        ----------
        mat : array-like of shape (n_samples, n_features)
            matrix which should be preprocessed
        subset : None or int
            number of features to keep in feature matrix (Top-N in variance); if it is None, attribute filter is either initialized (if it is equal to None) or used to filter features
        verbose : bool
            prints out information

        Returns
        -------
        mat_nan : array-like of shape (n_samples, n_features)
            Preprocessed matrix
        '''
        mat_nan = np.nan_to_num(mat, copy=True, nan=np.nan, posinf=self.posinf, neginf=self.neginf)
        assert mat_nan.shape==mat.shape
        if ((subset is not None) or (self.filter is not None)):
            if (verbose):
                print("<preprocessing.CustomScaler> Selecting the %d most variable features out of %d..." % (subset, mat_nan.shape[1]))
            if ((subset is not None) and (self.filter is None)):
                with np.errstate(over="ignore"):
                    x_vars = [np.nanvar(mat_nan[:,i]) if (np.sum(~np.isnan(mat_nan[:,i]))>0) else 0 for i in range(mat_nan.shape[1])]
                    x_vars = [x if (not np.isnan(x) and not np.isinf(x)) else 0 for x in x_vars]
                    x_ids_vars = np.argsort(x_vars).tolist()
                    features = x_ids_vars[-subset:]
                    self.filter = features
            mat_nan = mat_nan[:,self.filter]
        assert mat_nan.shape[1]==(mat.shape[1] if (self.filter is None) else len(self.filter)) and mat_nan.shape[0]==mat.shape[0]
        mat_nan[:,np.sum(~np.isnan(mat_nan), axis=0)==0] = 0 # avoid overflow warning from SimpleImputer
        if (verbose):
            print("<preprocessing.CustomScaler> %d perc. of missing values (||.|| = %f)..." % (100*np.sum(np.isnan(mat_nan))/np.prod(list(mat_nan.shape)), np.linalg.norm(mat_nan)), end=" ")
        if (self.imputer is None and np.sum(np.isnan(mat_nan))>0):
            with np.errstate(under="ignore"):
                self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean', keep_empty_features=True)
                mat_nan = self.imputer.fit_transform(mat_nan)
        else:
            if (np.sum(np.isnan(mat_nan))>0):
                mat_nan = self.imputer.transform(mat_nan)
        if (verbose):
            print("Final ||.|| = %f" % (np.linalg.norm(mat_nan)))
        assert mat_nan.shape[0]==mat.shape[0]
        if (self.scaler is None):
            self.scaler = StandardScaler()
            mat_nan_std = self.scaler.fit_transform(mat_nan)
        else:
            mat_nan_std = self.scaler.transform(mat_nan)
        assert mat_nan_std.shape[1]==(mat.shape[1] if (self.filter is None) else len(self.filter)) and mat_nan_std.shape[0]==mat.shape[0]
        return mat_nan_std

def meanimputation_standardize(dataset, subset=None, scalerS=None, scalerP=None, inf=int(1e1), verbose=False):
    '''
    Computes a single feature matrix and response vector from a drug repurposing dataset, by imputation by the average value of a feature for missing values and by centering and standardizing user and item feature matrices and concatenating them

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset which should be transformed, with n_items items (with n_item_features features) and n_users users (with n_user_features features) where missing values are denoted by numpy.nan
    subset : None or int
        number of features to keep in item feature matrix, and in user feature matrix (selecting the ones with highest variance)
    scalerS : None or sklearn.preprocessing.StandardScaler instance
        scaler for items
    scalerP : None or sklearn.preprocessing.StandardScaler instance
        scaler for users
    verbose : bool
        prints out information

    Returns
    ----------
    X : array-like of shape (n_folds, n_item_features+n_user_features)
        the feature matrix
    y : array-like of shape (n_folds, )
        the response/outcome vector
    scalerS : None or stanscofi.models.CustomScaler instance
        scaler for items; if the input value was None, returns the scaler fitted on item feature vectors
    scalerP : None or stanscofi.models.CustomScaler instance
        scaler for users; if the input value was None, returns the scaler fitted on user feature vectors
    '''
    y = dataset.ratings.toarray()[dataset.folds.row,dataset.folds.col].ravel()
    if (scalerS is None):
        scalerS = CustomScaler(posinf=inf, neginf=-inf)
    if (verbose):
        print("<preprocessing.meanimputation_standardize> Preprocessing of item feature matrix...")
    S_ = scalerS.fit_transform(dataset.items.toarray().T, subset=subset, verbose=verbose) ## items x features=subset
    if (scalerP is None):
        scalerP = CustomScaler(posinf=inf, neginf=-inf)
    if (verbose):
        print("<preprocessing.meanimputation_standardize> Preprocessing of user feature matrix...")
    P_ = scalerP.fit_transform(dataset.users.toarray().T, subset=subset, verbose=verbose) ## users x features=subset
    X = np.column_stack((S_[dataset.folds.row,:], P_[dataset.folds.col,:])) ## (item, user) pairs x (item + user features)
    return X, y, scalerS, scalerP

def same_feature_preprocessing(dataset, operator):
    '''
    If the users and items have the same features in the dataset, then a simple way to combine the user and item feature matrices is to apply an element-wise arithmetic operator (*, +, etc.) to the feature vectors coefficient per coefficient.

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset which should be transformed, where n_item_features==n_user_features and dataset.same_item_user_features==True
    operator : str
        arithmetric operation to apply, ex. "+", "*"

    Returns
    ----------
    X : array-like of shape (n_folds, n_features)
        the feature matrix
    y : array-like of shape (n_folds, )
        the response/outcome vector
    '''
    assert dataset.same_item_user_features
    y = dataset.ratings.toarray()[dataset.folds.row,dataset.folds.col].ravel()
    S_, P_ = [np.nan_to_num(x.toarray().T, copy=True, nan=0.) for x in [dataset.items, dataset.users]]
    S_, P_ = S_[dataset.folds.row,:], P_[dataset.folds.col,:]
    X = eval("S_ %s P_" % operator) ## (item, user) pairs x (item + user features)
    return X, y
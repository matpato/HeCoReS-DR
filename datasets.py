#coding: utf-8

import pandas as pd
import numpy as np
from scipy.sparse import coo_array

def generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=12454):
    '''
    Creates a dummy dataset where the positive and negative (item, user) pairs are arbitrarily similar. 

    Each of the nfeatures features for (item, user) pair feature vectors associated with positive ratings are drawn from a Gaussian distribution of mean mean and standard deviation std, whereas those for negative ratings are drawn from from a Gaussian distribution of mean -mean and standard deviation std. User and item feature matrices of shape (nfeatures//2, npositive+nnegative) are generated, which are the concatenation of npositive positive and nnegative negative pair feature vectors generated from Gaussian distributions. Thus there are npositive^2 positive ratings (each "positive" user with a "positive" item), nnegative^2 negative ratings (idem), and the remainder is unknown (that is, (npositive+nnegative)^2-npositive^2-nnegative^2 ratings).

    ...

    Parameters
    ----------
    npositive : int
        number of positive items/users
    nnegative : int
        number of negative items/users
    nfeatures : int
        number of item/user features
    mean : float
        mean of generating Gaussian distributions
    std : float
        standard deviation of generating Gaussian distributions

    Returns
    ----------
    ratings : array-like of shape (n_items, n_users)
        a matrix which contains values in {-1, 0, 1} describing the known and unknown user-item matchings
    users : array-like of shape (n_item_features, n_items)
        a list of the item feature names in the order of column indices in ratings_mat
    items : array-like of shape (n_user_features, n_users)
        a list of the item feature names in the order of column indices in ratings_mat
    '''
    assert nfeatures%2==0
    np.random.seed(random_state)
    ## Generate feature matrices
    nusers = nitems = npositive+nnegative
    positive = np.random.normal(mean,std,size=(nfeatures,npositive))
    negative = np.random.normal(-mean,std,size=(nfeatures,nnegative))
    users = np.concatenate((positive, negative), axis=1)[:nfeatures//2,:]
    items = np.concatenate((positive, negative), axis=1)[nfeatures//2:,:]
    ## Generate ratings
    ratings = np.zeros((nitems, nusers))
    ratings[:npositive,:npositive] = 1
    ratings[npositive:,npositive:] = -1
    ## Input to stanscofi
    return {"ratings": coo_array(ratings), "users": users, "items": items}

class Dataset(object):
    def __init__(self, ratings=None, users=None, items=None, same_item_user_features=False, name="dataset"):
        '''
        Creates an instance of stanscofi.Dataset

        ...

        Parameters
        ----------
        ratings : array-like of shape (n_items, n_users)
            an array which contains values in {-1, 0, 1, np.nan} describing the negative, unlabelled, positive, unavailable user-item matchings
        items : array-like of shape (n_item_features, n_items)
            an array which contains the item feature vectors
        users : array-like of shape (n_user_features, n_users)
            an array which contains the user feature vectors
        same_item_user_features : bool (default: False)
            whether the item and user features are the same (optional)
        name : str
            name of the dataset (optional)
        '''
        assert ratings is not None
        assert users is not None and users.shape[1]==ratings.shape[1]
        assert items is not None and items.shape[1]==ratings.shape[0]
        ## get metadata
        if (str(type(ratings))=="<class 'pandas.core.frame.DataFrame'>"):
            self.item_list = [str(x) for x in ratings.index]
            self.user_list = [str(x) for x in ratings.columns]
            ratings_ = ratings.values
        else:
            self.item_list = [str(x) for x in range(ratings.shape[0])]
            self.user_list = [str(x) for x in range(ratings.shape[1])]
            if (str(type(ratings))=="<class 'scipy.sparse._arrays.coo_array'>"):
                ratings_ = ratings.toarray()
            else:
                ratings_ = ratings.copy()
            ratings_ = ratings_.copy()
        if (str(type(users))=="<class 'pandas.core.frame.DataFrame'>"):
            users = users[self.user_list]
            self.user_features = [str(x) for x in users.index]
            users_ = users.values
        else:
            self.user_features = [str(x) for x in range(users.shape[0])]
            users_ = users.copy()
        if (str(type(items))=="<class 'pandas.core.frame.DataFrame'>"):
            items = items[self.item_list]
            self.item_features = [str(x) for x in items.index]
            items_ = items.values
        else:
            self.item_features = [str(x) for x in range(items.shape[0])]
            items_ = items.copy()
        self.same_item_user_features = same_item_user_features
        if (self.same_item_user_features):
            features = list(set(self.item_features).intersection(set(self.user_features)))
            assert len(features)>0
            self.user_features = features
            self.item_features = features
            users = users.loc[features]
            items = items.loc[features]
        ## format
        self.name = name
        self.ratings = coo_array(np.nan_to_num(ratings_, copy=True, nan=0))
        ids = np.argwhere(~np.isnan(ratings_))
        row = ids[:,0].ravel()
        col = ids[:,1].ravel()
        data = [1]*ids.shape[0]
        self.folds = coo_array((data, (row, col)), shape=ratings_.shape)
        self.users = coo_array(users_)
        self.items = coo_array(items_)
        self.nusers = self.users.shape[1]
        self.nitems = self.items.shape[1]
        self.nuser_features = self.users.shape[0]
        self.nitem_features = self.items.shape[0]

    def summary(self, sep="-"*70):
        '''
        Prints out a summary of the contents of a stanscofi.Dataset: the number of items, users, the number of positive, negative, unlabeled, unavailable matchings, the sparsity number, and the shape and percentage of missing values in the item and user feature matrices

        ...

        Parameters
        ----------
        sep : str
            separator for pretty printing
        ...

        Returns
        ----------
        ndrugs : int
            number of drugs
        ndiseases : int
            number of diseases
        ndrugs_known : int
            number of drugs with at least one known (positive or negative) rating
        ndiseases_known : int
            number of diseases with at least one known (positive or negative) rating
        npositive : int
            number of positive ratings
        nnegative : int
            number of negative ratings
        nunlabeled_unavailable : int
            number of unlabeled or unavailable ratings
        nunavailable : int
            number of unavailable ratings
        sparsity : float
            percentage of known ratings
        sparsity_known : float
            percentage of known ratings among drugs and diseases with at least one known rating
        ndrug_features : int
            number of drug features
        missing_drug_features : float
            percentage of missing drug feature values
        ndisease_features : int
            number of disease features
        missing_disease_features : float
            percentage of missing disease feature values
        '''
        print(sep)
        print("* Rating matrix: %d drugs x %d diseases" % (self.nitems, self.nusers))
        restricted_ratings = self.ratings.toarray()[np.abs(self.ratings).sum(axis=1)>0,:]
        restricted_ratings = restricted_ratings[:,np.abs(restricted_ratings).sum(axis=0)>0]
        print("Including %d drugs and %d diseases involved in at least one positive/negative rating" % restricted_ratings.shape)
        print("%d positive, %d negative, %d unlabeled (including %d unavailable) drug-disease ratings" % ((self.ratings==1).sum(), (self.ratings==-1).sum(), np.prod(self.ratings.shape)-self.ratings.count_nonzero(), np.prod(self.folds.shape)-self.folds.count_nonzero()))
        print("Sparsity: %.2f percent (on drugs/diseases with at least one known rating %.2f)" % ((self.ratings!=0).mean()*100, (restricted_ratings!=0).mean()*100))
        print(sep[:len(sep)//2])
        print("* Feature matrices:")
        if (self.items.shape[0]>0):
            print("#Drug features: %d\tTotal #Drugs: %d" % (self.items.shape))
            print("Missing features: %.2f percent" % (np.isnan(self.items.toarray()).mean()*100))
        if (self.users.shape[0]>0):
            print("#Disease features: %d\tTotal #Disease: %d" % (self.users.shape))
            print("Missing features: %.2f percent" % (np.isnan(self.users.toarray()).mean()*100))
        if (self.users.shape[0]+self.items.shape[0]==0):
            print("No feature matrices.")
        print(sep+"\n")
        return self.nitems, self.nusers, restricted_ratings.shape[0], restricted_ratings.shape[1], (self.ratings==1).sum(), (self.ratings==-1).sum(), np.prod(self.ratings.shape)-self.ratings.count_nonzero(), np.prod(self.folds.shape)-self.folds.count_nonzero(), (self.ratings!=0).mean()*100, (restricted_ratings!=0).mean()*100, self.items.shape[0], np.isnan(self.items.toarray()).mean()*100, self.users.shape[0], np.isnan(self.users.toarray()).mean()*100

    def subset(self, folds, subset_name="subset"):
        '''
        Obtains a subset of a stanscofi.Dataset based on a set of user and item indices

        ...

        Parameters
        ----------
        folds : COO-array of shape (n_items, n_users)
            an array which contains values in {0, 1} describing the unavailable and available user-item matchings in ratings
        subset_name : str
            name of the newly created stanscofi.Dataset

        Returns
        ----------
        subset : stanscofi.Dataset
            dataset corresponding to the folds in input
        '''
        if (np.prod(folds.shape)==0):
            raise ValueError("Fold is empty!")
        assert folds.shape==self.folds.shape
        #data = self.ratings.toarray()[folds.row, folds.col].ravel()
        sfolds = np.asarray(folds.toarray(), dtype=float)
        sfolds[sfolds==0] = np.nan
        ratings = self.ratings.toarray() * sfolds
        ratings = pd.DataFrame(ratings, index=self.item_list, columns=self.user_list)
        users = pd.DataFrame(self.users.toarray(), index=self.user_features, columns=self.user_list)
        items = pd.DataFrame(self.items.toarray(), index=self.item_features, columns=self.item_list)
        subset =  Dataset(ratings=ratings, users=users, items=items, same_item_user_features=self.same_item_user_features, name=subset_name)
        return subset
    
    def get_id(self, index, axis):
        """
        Returns the user or item ID given an index and axis.

        Parameters
        ----------
        index : int
            The index of the item (if axis == "row") or user (if axis == "col").
        axis : str
            Either "row" (for items) or "col" (for users).

        Returns
        -------
        str
            The ID (name) corresponding to the index.

        Raises
        ------
        ValueError
            If an invalid axis is provided or the index is out of range.
        """
        if axis == "row":
            if index < 0 or index >= len(self.item_list):
                raise ValueError("Item index out of range.")
            return self.item_list[index]
        elif axis == "col":
            if index < 0 or index >= len(self.user_list):
                raise ValueError("User index out of range.")
            return self.user_list[index]
        else:
            raise ValueError("Axis must be either 'row' for items or 'col' for users.")

    def get_index(self, id_str):
        """
        Returns the row or column index corresponding to the given ID by inferring
        whether it's an item or a user.

        Parameters
        ----------
        id_str : str
            The item or user ID.

        Returns
        -------
        int
            The index of the ID in the appropriate list (item_list or user_list).

        Raises
        ------
        ValueError
            If the ID is not found in either the item or user list.
        """
        if id_str in self.item_list:
            return self.item_list.index(id_str)
        elif id_str in self.user_list:
            return self.user_list.index(id_str)
        else:
            raise ValueError(f"ID '{id_str}' not found in item_list or user_list.")

    def sum_ratings_for_disease(self, disease_index, ignore_nan=True):
        """
        Returns the sum of ratings for a given disease (column index in ratings matrix).

        Parameters
        ----------
        disease_index : int
            Index of the disease (i.e., column in the ratings matrix).
        ignore_nan : bool (default: True)
            If True, ignores NaN values in the ratings matrix.

        Returns
        -------
        float
            Sum of ratings for the specified disease (over all drugs).
        """
        ratings_mat = self.ratings.toarray()  # shape (n_items, n_users)
        column = ratings_mat[:, disease_index]
        if ignore_nan:
            return np.nansum(column)
        else:
            return np.sum(column)
        
    def sum_ratings_for_drug(self, drug_index, ignore_nan=True):
        """
        Returns the sum of ratings for a given drug (row index in ratings matrix).

        Parameters
        ----------
        drug_index : int
            Index of the drug (i.e., row in the ratings matrix).
        ignore_nan : bool (default: True)
            If True, ignores NaN values in the ratings matrix.

        Returns
        -------
        float
            Sum of ratings for the specified drug (over all diseases).
        """
        ratings_mat = self.ratings.toarray()  # shape (n_items, n_users)
        row = ratings_mat[drug_index, :]
        if ignore_nan:
            return np.nansum(row)
        else:
            return np.sum(row)

    def get_positive_indices(self):
        """
        Returns a list of (i, j) index pairs where self.ratings == 1.

        Returns
        -------
        List[Tuple[int, int]]
            A list of (row, col) index pairs where the rating is exactly 1.
        """
        coo = self.ratings.tocoo()  # ensure COO format
        return [(i, j) for i, j, v in zip(coo.row, coo.col, coo.data) if v == 1]
    
    def get_positive_rows_for_disease(self, disease_index):
        """
        Returns all row indices (drugs) where self.ratings[row, disease_index] == 1.

        Parameters
        ----------
        disease_index : int
            The column index corresponding to a disease.

        Returns
        -------
        List[int]
            List of row indices where the rating is exactly 1 for the given disease.
        """
        coo = self.ratings.tocoo()
        return [i for i, j, v in zip(coo.row, coo.col, coo.data)
                if j == disease_index and v == 1]

    def is_positive_pair(self, drug_index, disease_index, ignore_nan=True):
        """
        Returns whether a given (drug_index, disease_index) pair has a positive rating.

        Parameters
        ----------
        drug_index : int
            Row index of the drug.
        disease_index : int
            Column index of the disease.
        ignore_nan : bool (default: True)
            If True, treats NaN as 0 (unavailable). If False, returns NaN if the rating is missing.

        Returns
        -------
        int or float
            1 if rating == 1 (positive), 0 if not (including -1 or 0 or NaN),
            or np.nan if ignore_nan=False and rating is NaN.
        """
        rating = self.ratings.toarray()[drug_index, disease_index]
        
        if np.isnan(rating):
            return 0 if ignore_nan else np.nan
        return 1 if rating == 1 else 0




from sklearn.metrics import precision_recall_curve as PRC
from sklearn.metrics import roc_curve as ROC

import numpy as np
import pandas as pd
from copy import deepcopy

###################################
## CUTE-RANKING FUNCTIONS     ##
###################################

def hit_rate_at_k(rs, k):
    """Score is percentage of first relevant item in list that occur
    at rank k or lower. First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: the largest rank position to consider
    Returns:
        Hit Rate @k
    """
    if k < 1 or k > len(rs[0]):
        raise ValueError('k value must be >=1 and < Max Rank')
    hits = 0
    for r in rs:
        if np.sum(r[:k]) > 0: hits += 1

    return hits / len(rs)

def mean_rank(rs):
    """Score is mean rank of the first relevant item in list
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean rank
    """
    _rs = []
    for r in rs:
        ids = np.asarray(r).nonzero()[0]
        if len(ids) == 0:
            _rs.append(0)
        else:
            _rs.append(ids[0] + 1)
    return np.mean(_rs)

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])

def precision_at_k(r, k = None):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of documents to consider
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k or None (if considering all documents)
    """
    assert k is None or k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k and k is not None:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def recall_at_k(r, max_rel, k = None):
    """Score is recall after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        max_rel: Maximum number of documents that can be relevant
        k: Number of documents to consider
    Returns:
        Recall score
    """
    assert k is None or k >= 1
    r = r[:k]
    r = np.asarray(r) != 0
    if np.sum(r) > max_rel:
        raise ValueError('Number of relevant documents retrieved > max_rel')
    return np.sum(r) / max_rel

def f1_score_at_k(r, max_rel, k = None):
    """Score is harmonic mean of precision and recall
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        max_rel: Maximum number of documents that can be relevant
        k: Number of documents to consider
    Returns:
        F1 score @ k
    """
    p = precision_at_k(r, k)
    r = recall_at_k(r, max_rel, k)
    return 2 * p * r / (p + r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max



###################################
## LIST OF AVAILABLE METRICS     ##
###################################
from sklearn.metrics import roc_auc_score, fbeta_score
from scipy.stats import kendalltau, spearmanr

def ERR(y_true, y_pred, max=10, max_grade=2):
    '''
        source: https://raw.githubusercontent.com/skondo/evaluation_measures/master/evaluations_measures.py
    '''
    max=10
    max_grade=2
    ranking = y_true[np.argsort(-y_pred)]
    if max is None:
        max = len(ranking)
    ranking = ranking[:min(len(ranking), max)]
    ranking = map(float, ranking)
    result = 0.0
    prob_step_down = 1.0 
    for rank, rel in enumerate(ranking):
        rank += 1
        utility = (pow(2, rel) - 1) / pow(2, max_grade)
        result += prob_step_down * utility / rank
        prob_step_down *= (1 - utility)  
    return result
AUC = lambda y_true, y_pred, k, u1 : roc_auc_score(y_true, y_pred, average="weighted")
Fscore = lambda y_true, y_pred, u, beta : fbeta_score(y_true, y_pred, beta=beta, average="weighted")
def TAU(y_true, y_pred, u, u1):
    if (len(np.unique(y_pred))==1):
        return 0
    res = kendalltau(y_true, y_pred)
    return res.correlation
def Rscore(y_true, y_pred, u, u1):
    if (len(np.unique(y_pred))==1):
        return 0
    res = spearmanr(y_true, y_pred)
    return res.correlation
MRR = lambda y_true, y_pred, u, u1 : mean_reciprocal_rank([y_true[np.argsort(-y_pred)]])
RP = lambda y_true, y_pred, u, u1 : r_precision(y_true[np.argsort(-y_pred)])
PrecisionK = lambda y_true, y_pred, k, u1 : precision_at_k(y_true[np.argsort(-y_pred)], k)
RecallK = lambda y_true, y_pred, k, u1 : recall_at_k(y_true[np.argsort(-y_pred)], np.sum(y_true>0), k=k)
def F1K(y_true, y_pred, k, u1):
    rec = recall_at_k(y_true[np.argsort(-y_pred)], np.sum(y_true>0), k)
    prec = precision_at_k(y_true[np.argsort(-y_pred)], k)
    if (rec+prec==0):
        return 0
    return f1_score_at_k(y_true[np.argsort(-y_pred)], np.sum(y_true>0), k=k)
AP = lambda y_true, y_pred, u, u1 : average_precision(y_true[np.argsort(-y_pred)])
MAP = lambda y_true, y_pred, u, u1 : mean_average_precision([y_true[np.argsort(-y_pred)]])
DCGk = lambda y_true, y_pred, k, u1 : dcg_at_k(y_true[np.argsort(-y_pred)], k)
NDCGk = lambda y_true, y_pred, k, u1 : ndcg_at_k(y_true[np.argsort(-y_pred)], k)
MeanRank = lambda y_true, y_pred, k, u1 : mean_rank([y_true[np.argsort(-y_pred)]])
HRk = lambda y_true, y_pred, k, u1 : hit_rate_at_k([y_true[np.argsort(-y_pred)]], k)
# metrics_list = ["AUC", "Fscore", "TAU", "Rscore", "MRR", "RP", "PrecisionK", "RecallK", "F1K", "AP", "MAP", "DCGk", "NDCGk", "MeanRank", "HRk", "ERR"]
metrics_list = ["AUC", "MRR", "PrecisionK", "RecallK", "F1K", "MAP", "NDCGk", "HRk"]

###################################
## COMPUTATION OF METRICS        ##
###################################

def compute_metrics(scores, predictions, dataset, metrics, k=1, beta=1, verbose=False):
    '''
    Computes *user-wise* validation metrics for a given set of scores and predictions w.r.t. a dataset

    ...

    Parameters
    ----------
    scores : COO-array of shape (n_items, n_users)
        sparse matrix in COOrdinate format
    predictions : COO-array of shape (n_items, n_users)
        sparse matrix in COOrdinate format with values in {-1,1}
    dataset : stanscofi.Dataset
        dataset on which the metrics should be computed
    metrics : lst of str
        list of metrics which should be computed
    k : int (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    beta : float (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    verbose : bool
        prints out information about ignored users for the computation of validation metrics, that is, users which pairs are only associated to a single class (i.e., all pairs with this users are either assigned 0, -1 or 1)

    Returns
    -------
    metrics : pandas.DataFrame of shape (len(metrics), 2)
        table of metrics: metrics in rows, average and standard deviation across users in columns
    plots_args : dict
        dictionary of arguments to feed to the plot_metrics function to plot the Precision-Recall and the Receiver Operating Chracteristic (ROC) curves
    '''
    metrics_list = ["AUC", "Fscore", "TAU", "Rscore", "MRR", "RP", "PrecisionK", "RecallK", "F1K", "AP", "MAP", "DCGk", "NDCGk", "MeanRank", "HRk", "ERR"]
    assert predictions.shape==scores.shape==dataset.folds.shape
    assert all([metric in metrics_list for metric in metrics])
    y_true_all = dataset.ratings.toarray()[dataset.folds.row,dataset.folds.col].ravel() 
    y_pred_all = predictions.data.ravel()
    scores_all = scores.data.ravel()
    assert y_true_all.shape==y_pred_all.shape==scores_all.shape
    ## Compute average metric per user
    user_ids = np.unique(dataset.folds.col)
    n_ignored = 0
    aucs, tprs, recs, fscores = [], [], [], []
    base_fpr = np.linspace(0, 1, 101)
    base_pres = np.linspace(0, 1, 101)
    metrics_list = {metric: [] for metric in metrics}
    for user_id in user_ids:
        user_ids_i = np.argwhere(dataset.folds.col==user_id)
        if (len(user_ids_i)==0):
            n_ignored += 1
            continue
        user_truth = y_true_all[user_ids_i]
        user_pred = y_pred_all[user_ids_i]
        if ((len(np.unique(user_truth))==2) and (1 in user_truth)):
            fpr, tpr, _ = ROC(user_truth, user_pred, pos_label=1.)
            pres, rec, _ = PRC(user_truth, user_pred)
            aucs.append(roc_auc_score(user_truth, user_pred, average="weighted"))
            fscores.append(fbeta_score(user_truth, user_pred, beta=beta, average="weighted"))
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
            rec = np.interp(base_pres, pres, rec)
            recs.append(rec)
            for metric in metrics:
                #print(metric)
                value = eval(metric)(user_truth.ravel(), user_pred.ravel(), k, beta)
                metrics_list.update({metric: metrics_list[metric]+[value]})
        else:
            n_ignored += 1
    if (verbose and n_ignored>0):
        print("<validation.compute_metrics> Computed on #users=%d, %d ignored (%2.f perc)" % (len(user_ids), n_ignored, 100*n_ignored/len(user_ids)))
    if (len(aucs)==0 or len(fscores)==0):
        metrics = pd.DataFrame([], index=metrics, 
		columns=["Average", "StandardDeviation"])
        return metrics, {}
    metrics = pd.DataFrame([[f(metrics_list[m]) for f in [np.mean, np.std]] for m in metrics_list], index=metrics, columns=["Average", "StandardDeviation"])
    metrics = pd.concat((metrics, pd.DataFrame([[k,beta]], index=["arguments (k, beta)"], columns=metrics.columns)), axis=0)
    return metrics, {"y_true": (y_true_all>0).astype(int), "y_pred": (y_pred_all>0).astype(int), "scores": scores_all, "predictions": y_pred_all, "ground_truth": y_true_all, "aucs": aucs, "fscores": fscores, "tprs": np.array(tprs), "recs": np.array(recs)}
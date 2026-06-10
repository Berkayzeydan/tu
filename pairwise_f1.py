from itertools import combinations
from collections import Counter

def pairwise_f1_score(true_clusters, pred_clusters, n=None):
    """
    Computes the pairwise F1 score between two clusterings.
    true_clusters and pred_clusters are lists of lists of indices.
    """
    def get_pairs(clusters):
        pairs = set()
        for cluster in clusters:
            for i, j in combinations(sorted(cluster), 2):
                pairs.add((i, j))
        return pairs

    true_pairs = get_pairs(true_clusters)
    pred_pairs = get_pairs(pred_clusters)

    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

import numpy as np

def euclidean_distances(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def accuracy_score(pred, labels):
    score = np.sum(pred==labels)/len(labels)
    return score
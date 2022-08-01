import numpy as np
from collections import Counter
from utils import euclidean_distances

class KNN:
    def __init__(self, k=3):
        self.k = k
        

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        pred = [self.pred_neigh(x) for x in X_test]
        return pred

    def pred_neigh(self, x):
        distances = [euclidean_distances(x, i) for i in self.X_train] # find EU distances for getting best neighbours
        distances = np.argsort(distances) # sorting array for getting most nearest neigbours
        top_K_dist = distances[:self.k]
        top_K_neigh = [self.y_train[x] for x in top_K_dist]
        return Counter(top_K_neigh).most_common(1)[0][0] 


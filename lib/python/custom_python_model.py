from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
#from sklearn.base import RegressorMixin


class MyRandomRegressor(BaseEstimator):
    """This model predicts random values between the mininimum and the maximum of y"""

    def fit(self, X, y):
        self.y_range = [np.min(y), np.max(y)]

    def predict(self, X):
        return pd.Series(np.random.uniform(self.y_range[0], self.y_range[1], size=X.shape[0]))
    
    
    
class MyNullRegressor(BaseEstimator):
    
    def fit(self, X, y):
        # The prediction will always just be the mean of y
        self.y_bar_ = np.mean(y)
        
    def predict(self, X=None):
        # Give back the mean of y, in the same
        # length as the number of X observations
        return pd.Series(np.ones(X.shape[0]) * self.y_bar_)
    
    
class AdaBoostModel(BaseEstimator):
	def __init__(self):
		self.model = AdaBoostClassifier(n_estimators=100,learning_rate=0.1,random_state=0)

	def fit(self, X, y):
		self.model.fit(X,y)
		self.classes_ = self.model.classes_
		return None

	def predict(self, X):
		return self.model.predict(X)

	def predict_proba(self, X):
		return self.model.predict_proba(X)

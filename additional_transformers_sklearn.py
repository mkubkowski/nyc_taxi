from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from sklearn.utils.validation import check_is_fitted,FLOAT_DTYPES
from sklearn.utils import check_array

import numpy as np

class IdentityTransformer(BaseEstimator, TransformerMixin):
    ''' 
    Transformer which only passes input through it.
    '''
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1
    
class OutlierTrimmer(BaseEstimator, TransformerMixin):
    '''
    Transformer trimming values of variables to interval [max(q(q_min),value_min), min(q(q_max),value_max)],
    where q(p) is a quantile of order p for a given column and value_min, value_max are float numbers, 
    common for all transformed columns. 
    
    It is based on MinMaxScaler in sklearn.
    '''
    
    def __init__(self,q_min=0,q_max=1,value_min=None,value_max=None,copy=True):
        self.q_min = q_min
        self.q_max = q_max
        self.value_min = value_min
        self.value_max = value_max
        self.vals_min_ = None
        self.vals_max_ = None
        self.copy=copy
    
    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.q_min
            del self.q_max
            del self.value_min
            del self.value_max
            del self.vals_min_
            del self.vals_max_
            del self.copy
    
    def fit(self,X,y=None):
        """Compute lists vals_min_ and vals_max_ having length 
        n_features for trimming of each column in X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute vals_min_ and vals_max_
            used for later trimming along the features axis.
        """
        self._reset()
        
        if sparse.issparse(X):
            raise TypeError("OutlierTrimmer does not support sparse input.")

        X = check_array(X, copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")
        if self.q_min is None or not( isinstance(self.q_min,float) or isinstance(self.q_min,int))\
            or self.q_min<0 or self.q_min>1:
            self.q_min = 0
        data_min = np.quantile(X, q=self.q_min, axis=0)
        if self.value_min is not None and ( isinstance(self.q_min,float) or isinstance(self.q_min,int)):
            data_min = np.maximum(data_min,self.value_min)
        
        if self.q_max is None or not( isinstance(self.q_max,float) or isinstance(self.q_max,int))\
            or self.q_max<0 or self.q_max>1:
            self.q_max = 1
        data_max = np.quantile(X, q=self.q_max, axis=0)
        if self.value_max is not None and ( isinstance(self.q_max,float) or isinstance(self.q_max,int)):
            data_max = np.minimum(data_max,self.value_max)

        self.vals_min_ = data_min
        self.vals_max_ = data_max
        return self
    
    def transform(self,X):
        """Trimming features of X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """  
        check_is_fitted(self, 'vals_min_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        for i in range(X.shape[1]):
            X[X[:,i]<self.vals_min_[i],i] = self.vals_min_[i]
            X[X[:,i]>self.vals_max_[i],i] = self.vals_max_[i]            

        return X
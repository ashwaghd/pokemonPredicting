#!/usr/bin/env python3
import pandas as pd

################################################################
#
# These custom classes help with pipeline building and debugging
#
import sklearn.base

class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        return X

class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Pipeline member to display the data at this stage of the transformation.
    """
    
    def __init__(self, title):
        self.title = title
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        self.do_predictors = do_predictors
        self.do_numerical = do_numerical
        self.numerical_columns = None
        self.categorical_columns = None
        self.label_column = None
        return

    def fit(self, X, y=None):
        # Determine numerical and categorical columns dynamically
        self.numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Set attributes based on the type of columns we want
        if self.do_predictors:
            if self.do_numerical:
                self.mAttributes = self.numerical_columns
            else:
                self.mAttributes = self.categorical_columns
        else:
            # For non-predictors (i.e., the target/label column)
            # This assumes the label column is already removed from X
            # The label handling is done in preprocess_dataframe function
            self.mAttributes = []
            
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        # only keep columns selected
        # Make sure we have attributes to select
        if not hasattr(self, 'mAttributes') or not self.mAttributes:
            if self.do_numerical:
                return X.select_dtypes(include=['int64', 'float64'])
            else:
                return X.select_dtypes(include=['object', 'category', 'bool'])
        
        # Filter to only include columns that exist in X
        valid_attributes = [col for col in self.mAttributes if col in X.columns]
        if not valid_attributes:
            # If no valid columns remain, return an appropriate subset based on types
            if self.do_numerical:
                return X.select_dtypes(include=['int64', 'float64'])
            else:
                return X.select_dtypes(include=['object', 'category', 'bool'])
        
        values = X[valid_attributes]
        return values

#
# These custom classes help with pipeline building and debugging
#
################################################################

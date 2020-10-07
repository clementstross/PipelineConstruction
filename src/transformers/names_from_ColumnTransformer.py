from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import numpy as np

def names_from_ColumnTransformer(column_transformer):
    """[summary]

    Args:
        column_transformer (Pipeline): [Pipeline object containing trainsformations]

    Returns:
        list(str): list of strings which are the column names of the new tarnsformed columns
    """

    col_name = []
    
    for transformer_in_columns in column_transformer.transformers_:#the last transformer is ColumnTransformer's 'remainder'
        if transformer_in_columns[0]!= "remainder":
            raw_col_name = transformer_in_columns[2]

            if isinstance(transformer_in_columns[1],Pipeline): # Deal with piplines use last step
                transformer = transformer_in_columns[1].steps[-1][1]
            else:
                transformer = transformer_in_columns[1]

            if isinstance(transformer,FeatureUnion): # Deal with FeatureUnion. Take name of each step
                names = [nn[0] for nn in transformer.transformer_list]

            else:
                try:
                    names = transformer.get_feature_names(raw_col_name)
                except AttributeError: # if no 'get_feature_names' function, use raw column name
                    names = raw_col_name
                    
            if isinstance(names,np.ndarray): # eg.
                col_name += names.tolist()
            elif isinstance(names,list):
                col_name += names    
            elif isinstance(names,str):
                col_name.append(names)
    return col_name


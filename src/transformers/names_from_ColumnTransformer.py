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
                names = names_from_FeatureUnion(FeatureUnion_transformer=transformer, raw_col_name=raw_col_name)

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

def names_from_FeatureUnion(FeatureUnion_transformer, raw_col_name=None):
    names = []
    for transformer_in_union in FeatureUnion_transformer.transformer_list:
        
        raw_col_name_t = transformer_in_union[0]

        if isinstance(transformer_in_union[1],Pipeline): # Deal with piplines use last step
            transformer = transformer_in_union[1].steps[-1][1]
        else:
            transformer = transformer_in_union[1]
        
        try:
            names_t = transformer.get_feature_names([raw_col_name_t])
            names += names_t.tolist()
        except AttributeError:
            try:
                names_t = transformer._encoder.get_feature_names([raw_col_name_t])
                names += names_t.tolist()
            except AttributeError:
                names.append(raw_col_name_t)
    return(names)
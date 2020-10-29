import sys
import pandas as pd
import numpy as np
import types

# Import standard functions
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.impute import MissingIndicator

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# Import custom functions
sys.path.append('c:/Users/User/Documents/User/Work/Admiral/Pipeline_Builder/pipelineconstruction/src')
from transformers.IdentifyUnknowns import IdentifyUnknowns
from transformers.FrequencyEncoder import FrequencyEncoder
from transformers.CategoricalCatLimit import CategoricalCatLimit


def BuildCategoricalPipeline(control_sheet):

    #Check columns exist
    needed_cols = ["FeatureName", "TransformedName", "Include", "Raw", "Categorical", "Categorical_Level_Number", "Categorical_Encoding", "Categorical_Ordering", "Missing_Values", "Shadow_Col", "Impute_Strategy", "Impute_Value"]

    for need_col in needed_cols:
        if need_col not in control_sheet.columns:
            raise ValueError(f"{need_col} must be in control_sheet.columns")


    # Loop over all columns
    for ii in range(0, control_sheet.shape[0]):
        if ii == 0: # create output list
            col_transformers_list = []

        # Check values for boolen columns
        for check_col in ["Include", "Raw", "Shadow_Col", "Categorical"]:
            if control_sheet.iloc[ii][check_col] not in ["Y", "N", ""]:
                raise ValueError(f'for feature-{in_feat_ii} column-{check_col} must be in ["Y", "N", ""]. Is  {control_sheet.iloc[ii][check_col]}')

        # Only include Features which are to include
        if control_sheet.iloc[ii]["Include"]=="Y" and control_sheet.iloc[ii]["Raw"]=="Y" and control_sheet.iloc[ii]["Categorical"]=="Y":
            in_feat_ii = control_sheet.iloc[ii]["FeatureName"]
            out_feat_ii = control_sheet.iloc[ii]["TransformedName"]
            
            # Start with a empty list (no transformations)
            feature_union_list = []

            # Removing known missings ---------------------------------------------------
            # Extract string of know missing
            Missing_Values = control_sheet.iloc[ii]["Missing_Values"]
            if Missing_Values == Missing_Values: # check not unknown
                str_unks = [substring.strip() for substring in Missing_Values.split(";")]
            else:
                str_unks = []

            # If there are some values of known missings (e.g. _NA) add transformer to replace them with NAs
            if len(str_unks) > 0:
                feature_union_list.append(("unk_levels", IdentifyUnknowns(unk_levels=[str_unks])))


            # Impute missing values -------------------------------------------------------
            # Impute missing values
            impute_strategy = control_sheet.iloc[ii]["Impute_Strategy"]
            impute_value = control_sheet.iloc[ii]["Impute_Value"]
            
            # Check value of impute_strategy
            if impute_strategy not in ["mean", "median", "most_frequent", "constant"]:
                raise ValueError(f'For feature-{in_feat_ii} Impute_Strategy must be in ["mean", "median", "most_frequent", "constant"] is {impute_strategy}')
            # Check constant value given if needed
            if impute_strategy=="constant" and impute_value!=impute_value:
                raise ValueError(f'For feature-{in_feat_ii} if Impute_Strategy is "constant", impute_value can not be NA.') 

            feature_union_list_no_impute = feature_union_list.copy() # needed to get shadow matrix
            feature_union_list.append(("impute", SimpleImputer(missing_values=np.nan, strategy=impute_strategy, fill_value=impute_value)))


            # Capping the number of levels ---------------------------------------------
            # Capping the number of levels
            Categorical_Level_Number = Categorical_Encoding = control_sheet.iloc[ii]["Categorical_Level_Number"]
            if Categorical_Level_Number==Categorical_Level_Number: # Check the value is Categorical_Level_Number is known
                
                # Error checking
                if Categorical_Level_Number != int(Categorical_Level_Number) or Categorical_Level_Number<=0:
                    raise  ValueError(f'for feature-{in_feat_ii} Categorical_Level_Number must be int > 0. Is  {Categorical_Level_Number}')

                
                feature_union_list.append(("cat_capping",CategoricalCatLimit(cat_num=[int(Categorical_Level_Number)])))

            # Encoding ------------------------------------------------------------------
            # Encoding
            Categorical_Encoding = control_sheet.iloc[ii]["Categorical_Encoding"]
            if Categorical_Encoding not in ["ordinal", "one_hot", "frequency"]:
                raise  ValueError(f'for feature-{in_feat_ii} Categorical_Encoding must be in ["ordinal", "one_hot", "frequency"]. Is  {control_sheet.iloc[ii][check_col]}')
            
            # Choice encoding type
            Categorical_Ordering = control_sheet.iloc[ii]["Categorical_Ordering"]
            
            if Categorical_Ordering == Categorical_Ordering: # check not unknown. If known use as user settings. else use Auto
                input_order = [substring.strip() for substring in Categorical_Ordering.split(";")] # clean the input string
                
                if Categorical_Level_Number==Categorical_Level_Number and not np.isnan(Categorical_Level_Number): # Add Other if number of levels is capped
                    seen = set()
                    seen_add = seen.add
                    input_order = [x for x in input_order + ["other"] if not (x in seen or seen_add(x))]

                if impute_strategy=="constant" and impute_value==impute_value: # Add imputed value if not present
                    seen = set()
                    seen_add = seen.add
                    input_order = [x for x in [impute_value] + input_order  if not (x in seen or seen_add(x))]

                str_ordering = [input_order] # Use ordering if given
                
            else:
                str_ordering = "auto" # Use default ordering

            if Categorical_Encoding == "ordinal": # Use ordinal encoding
                feature_union_list.append(("encode", OrdinalEncoder(categories=str_ordering)))
            elif Categorical_Encoding == "frequency": # Use frequency encoding
                feature_union_list.append(("encode", FrequencyEncoder(categories=str_ordering, keep_at_start="Unknown", keep_at_end="other")))
            elif Categorical_Encoding == "one_hot": # Use frequency encoding
                feature_union_list.append(("encode", OneHotEncoder(categories=str_ordering, drop=None)))

            # Combine all features -------------------------------------------------------
            if control_sheet.iloc[ii]["Shadow_Col"]=="Y":
                # If shadow column needed as second pipeline to give extra column
                pre_col_ii = FeatureUnion([(out_feat_ii, Pipeline(feature_union_list)), \
                                        (out_feat_ii + "_NA", Pipeline([(out_feat_ii + "_imp", Pipeline(feature_union_list_no_impute)), \
                                                                        (out_feat_ii + "_shadow", MissingIndicator(missing_values=["", "Unknown"], features="all"))]) \
                                        ) \
                                        ])
            else:
                # If shadow column not needed use existing pipeline
                pre_col_ii = Pipeline(feature_union_list)
            
            # Add column to transformation list
            col_transformers_list.append((out_feat_ii, pre_col_ii, [in_feat_ii]))
    
    return(col_transformers_list)
import sys
import pandas as pd
import numpy as np
import types
import re

# Import standard functions
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

# Import custom functions
sys.path.append('c:/Users/User/Documents/User/Work/Admiral/Pipeline_Builder/pipelineconstruction/src')
from transformers.IdentifyUnknowns import IdentifyUnknowns
from transformers.NumericCapping import NumericCapping
from transformers.SetBinDiscretizer import SetBinDiscretizer
from transformers.names_from_ColumnTransformer import names_from_ColumnTransformer


def BuildNumericPipeline(control_sheet):
    """Build a ColumnTransformer for all columns in the control_sheet meta_data

    Args:
        control_sheet (pd.DataFrame): Must contain columns "FeatureName", "TransformedName", "Include", "Raw", "Numeric", "Missing_Values", 
                                                            "Numeric_Unk_Max", "Numeric_Cap_Max", "Numeric_Cap_Min", "Numeric_Unk_Min",
                                                            "Impute_Strategy", "Impute_Value"
                                      the incorrect value of inputs will raise errors

    Returns:
        ColumnTransformer: ColumnTransformer object contains all transformation in the control sheet.
    """
    #Check columns exist
    needed_cols = ["FeatureName", "TransformedName", \
                "Include", "Raw", "Numeric", "Missing_Values", \
                "Numeric_Unk_Max", "Numeric_Cap_Max", "Numeric_Cap_Min", "Numeric_Unk_Min", \
                    "Impute_Strategy", "Impute_Value"]
    for need_col in needed_cols:
        if need_col not in control_sheet.columns:
            raise ValueError(f"{need_col} must be in control_sheet.columns")

    # Loop over all columns
    for ii in range(0, control_sheet.shape[0]):
        if ii == 0: # create output list
            col_transformers_list = []

        # Check values for boolen columns
        for check_col in ["Include", "Raw", "Shadow_Col", "Numeric"]:
            if control_sheet.iloc[ii][check_col] not in ["Y", "N", ""]:
                raise ValueError(f'for feature-{in_feat_ii} column-{check_col} must be in ["Y", "N", ""]. Is  {control_sheet.iloc[ii][check_col]}')

        # Only include Features which are to include
        if control_sheet.iloc[ii]["Include"]=="Y" and control_sheet.iloc[ii]["Raw"]=="Y" and control_sheet.iloc[ii]["Numeric"]=="Y":
            in_feat_ii = control_sheet.iloc[ii]["FeatureName"]
            out_feat_ii = control_sheet.iloc[ii]["TransformedName"]
            
            # Start with a empty list (no transformations)
            feature_union_list = []

            # Removing known missings ---------------------------------------------------
            # Extract string of know missing
            str_unks = control_sheet.iloc[0]["Missing_Values"].split(";")
            num_unks = [float(unk) for unk in str_unks]

            # If there are some values of known missings (e.g. -1) add transformer to replace them with NAs
            if len(num_unks) > 0:
                feature_union_list.append(("unk_levels", IdentifyUnknowns(unk_levels=[num_unks])))


            # Applying caps and collars to values ---------------------------------------------------
            # Read in caps
            unk_max = control_sheet.iloc[ii]["Numeric_Unk_Max"]
            cap_max = control_sheet.iloc[ii]["Numeric_Cap_Max"]
            cap_min = control_sheet.iloc[ii]["Numeric_Cap_Min"]
            unk_min = control_sheet.iloc[ii]["Numeric_Unk_Min"]
            # Check if any aren't NA. If so add capping transformer
            if unk_max==unk_max or cap_max==cap_max or cap_min==cap_min or cap_max==cap_max: 
                feature_union_list.append(("capping", NumericCapping(unk_max=[unk_max], cap_max=[cap_max], cap_min=[cap_min], unk_min=[unk_min])))

            # Impute missing values -------------------------------------------------------
            # Impute missing values
            impute_strategy = control_sheet.iloc[ii]["Impute_Strategy"]
            impute_value = float(control_sheet.iloc[ii]["Impute_Value"])
            
            # Check value of impute_strategy
            if impute_strategy not in ["mean", "median", "most_frequent", "constant"]:
                raise ValueError(f'For feature-{in_feat_ii} Impute_Strategy must be in ["mean", "median", "most_frequent", "constant"] is {impute_strategy}')
            # Check constant value given if needed
            if impute_strategy=="constant" and impute_value!=impute_value:
                raise ValueError(f'For feature-{in_feat_ii} if Impute_Strategy is "constant", impute_value can not be NA.')    

            feature_union_list_no_impute = feature_union_list.copy() # needed to get shadow matrix
            feature_union_list.append(("impute", SimpleImputer(missing_values=np.nan, strategy=impute_strategy, fill_value=impute_value)))

            # Add Additional encodings ---------------------------------------------------
            one_hot_encoding = control_sheet.iloc[ii]["One_Hot_Encoding"]

            if one_hot_encoding==one_hot_encoding: #Is not missing
                if re.search("^(uniform|quantile|kmeans)\W*(\d+)", one_hot_encoding.lower()):
                    strategy = re.search("^(uniform|quantile|kmeans)\W*(\d+)", one_hot_encoding.lower()).group(1)
                    n_bins = int(re.search("^(uniform|quantile|kmeans)\W*(\d+)", one_hot_encoding.lower()).group(2))

                    feature_union_list.append(("one_hot_encoding", KBinsDiscretizer(n_bins=n_bins, strategy=strategy)))

                elif re.search("^(\s*\d+(\.\d*)?\s*;)+\s*$", one_hot_encoding):
                    bin_edges = [float(ii) for ii in one_hot_encoding.split(";")[:-1]]

                    feature_union_list.append(("one_hot_encoding", SetBinDiscretizer(bin_edges_internal=[bin_edges], input_features=[in_feat_ii])))
                else:
                    raise ValueError(f'For feature-{in_feat_ii} if one_hot_encoding must be blank or match either "^(uniform|quantile|kmeans)\W*(\d+)" or "^(\s*\d+(\.\d*)?\s*;)+\s*$" it is {one_hot_encoding}')


            # Combine all features -------------------------------------------------------

            if control_sheet.iloc[ii]["Shadow_Col"]=="Y":
                # If shadow column needed as second pipeline to give extra column
                pre_col_ii = FeatureUnion([(out_feat_ii, Pipeline(feature_union_list)), \
                                        (out_feat_ii + "_NA", Pipeline([(out_feat_ii + "_imp", Pipeline(feature_union_list_no_impute)), \
                                                                        (out_feat_ii + "_shadow", MissingIndicator(missing_values=np.nan, features="all"))]) \
                                        ) \
                                        ])
            else:
                # If shadow column not needed use existing pipeline
                pre_col_ii = Pipeline(feature_union_list)
            
            # Add column to transformation list
            col_transformers_list.append((out_feat_ii, pre_col_ii, [in_feat_ii]))


    return(col_transformers_list)


if __name__=="__main__":
    pass
    # Move to tests
    #control_sheet = pd.read_csv("./california_housing_control_sheet.csv")
    #num_trans = BuildNumericPipeline(control_sheet)
    
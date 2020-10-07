import numpy as np
import pandas as pd

import unittest
from IdentifyUnknowns import IdentifyUnknowns


class TestCategoricalCatLimit(unittest.TestCase):
    def test_d1_np(self):
        
        X=np.array([[1.], [2.], [3.], [4.]])

        id_unk = IdentifyUnknowns(unk_levels=[[1.,2.]])
        X_out = id_unk.fit_transform(X=X)

        self.assertTrue(all(np.isnan(X_out.flatten().tolist()[0:2])))
        self.assertEqual(X_out.flatten().tolist()[2:4], [3., 4.])

    def test_d2_np(self):
        
        X2=np.array([[1., 1.], [2., 2,], [3., 3.], [4., 4.]])

        id_unk2 = IdentifyUnknowns(unk_levels=[[1.,2.], [3., 4.]])
        X2_out = id_unk2.fit_transform(X=X2)


        self.assertTrue(all(np.isnan(X2_out[:,0].flatten().tolist()[0:2])))
        self.assertEqual(X2_out[:,0].flatten().tolist()[2:4], [3., 4.])

        self.assertTrue(all(np.isnan(X2_out[:,1].flatten().tolist()[3:4])))
        self.assertEqual(X2_out[:,1].flatten().tolist()[0:2], [1., 2.])

    def test_d1_pd(self):
        
        X=np.array([[1.], [2.], [3.], [4.]])
        X_df = pd.DataFrame(X, columns=["col1"])

        id_unk = IdentifyUnknowns(unk_levels=[[1.,2.]])
        X_df_out = id_unk.fit_transform(X=X_df)

        self.assertTrue(all(np.isnan(X_df_out.iloc[:,0].tolist()[0:2])))
        self.assertEqual(X_df_out.iloc[:,0].tolist()[2:4], [3., 4.])

    def test_d2_pd(self):
        
        X2=np.array([[1., 1.], [2., 2,], [3., 3.], [4., 4.]])
        X2_df = pd.DataFrame(X2, columns=["col1", "col2"])

        id_unk2 = IdentifyUnknowns(unk_levels=[[1.,2.], [3., 4.]])
        X2_df_out = id_unk2.fit_transform(X=X2_df)

        self.assertTrue(all(np.isnan(X2_df_out.iloc[:,0].tolist()[0:2])))
        self.assertEqual(X2_df_out.iloc[:,0].tolist()[2:4], [3., 4.])

        self.assertTrue(all(np.isnan(X2_df_out.iloc[:,1].tolist()[3:4])))
        self.assertEqual(X2_df_out.iloc[:,1].tolist()[0:2], [1., 2.])


if __name__ == "__main__":
    unittest.main()
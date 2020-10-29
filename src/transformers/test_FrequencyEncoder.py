import numpy as np
import pandas as pd

import unittest
from FrequencyEncoder import FrequencyEncoder

from sklearn.preprocessing import OrdinalEncoder

class TestCategoricalCatLimit(unittest.TestCase):
    
    def test_simple_freqEncoding(self):
        """Check basis functionality of frequency encoding
        """        
        enc = FrequencyEncoder()
        X = np.array([['Male', "Big"], ['Female', "Big"], ['Female', "Small"]])
        enc.fit(X)
        X_tran = enc.transform(X)
        X_invtran = enc.inverse_transform(X_tran)

        
        self.assertListEqual(enc.categories_[0].tolist(), ["Female", "Male"])
        self.assertListEqual(enc.categories_[1].tolist(), ["Big", "Small"])

        self.assertTrue((X_tran == np.array([[1, 0], [0, 0], [0, 1]])).all())
        self.assertTrue((X_invtran == X).all())

        #Test dataframe input
        enc_df = FrequencyEncoder()
        X_df = pd.DataFrame(X)
        enc_df.fit(X_df)
        X_df_tran = enc.transform(X_df)
        X_df_invtran = enc.inverse_transform(X_df_tran)

        self.assertListEqual(enc_df.categories_[0].tolist(), ["Female", "Male"])
        self.assertListEqual(enc_df.categories_[1].tolist(), ["Big", "Small"])

        self.assertTrue((X_df_tran == np.array([[1, 0], [0, 0], [0, 1]])).all())
        self.assertTrue((X_df_invtran == X).all())


    def test_complex_freqEncoding(self):
        """Check that keep at start and keep at end work as expected on data without them and only on one column
        """  
        enc = FrequencyEncoder(keep_at_start=[["Unknown"],[]], keep_at_end=[["Other"],[]])
        X = np.array([['Male', "Big"], ['Female', "Big"], ['Female', "Small"]], dtype='object')
        enc.fit(X)
        X_tran = enc.transform(X)
        X_invtran = enc.inverse_transform(X_tran)

        
        self.assertListEqual(enc.categories_[0].tolist(), ["Unknown", "Female", "Male", "Other"])
        self.assertListEqual(enc.categories_[1].tolist(), ["Big", "Small"])

        self.assertTrue((X_tran == np.array([[2, 0], [1, 0], [1, 1]])).all())
        self.assertTrue((X_invtran == X).all())

        #Test dataframe input
        enc_df = FrequencyEncoder(keep_at_start=[["Unknown"],[]], keep_at_end=[["Other"],[]])
        X_df = pd.DataFrame(X)
        enc_df.fit(X_df)
        X_df_tran = enc.transform(X_df)
        X_df_invtran = enc.inverse_transform(X_df_tran)

        self.assertListEqual(enc_df.categories_[0].tolist(), ["Unknown", "Female", "Male", "Other"])
        self.assertListEqual(enc_df.categories_[1].tolist(), ["Big", "Small"])

        self.assertTrue((X_df_tran == np.array([[2, 0], [1, 0], [1, 1]])).all())
        self.assertTrue((X_df_invtran == X).all())


    def test_with_others_freqEncoding(self):
        """Check that keep at start and keep at end work as expected when they contain values in the data
        """        
        enc = FrequencyEncoder(keep_at_start=[["Unknown"],["Unknown"]], keep_at_end=[["Other"],["Other"]])
        X = np.array([['Male', "Big"], ['Female', "Big"], ['Female', "Small"], ['Unknown', "Other"]], dtype='object')
        enc.fit(X)
        X_tran = enc.transform(X)
        X_invtran = enc.inverse_transform(X_tran)

        
        self.assertListEqual(enc.categories_[0].tolist(), ["Unknown", "Female", "Male", "Other"])
        self.assertListEqual(enc.categories_[1].tolist(), ["Unknown", "Big", "Small", "Other"])

        self.assertTrue((X_tran == np.array([[2, 1], [1, 1], [1, 2], [0, 3]])).all())
        self.assertTrue((X_invtran == X).all())

        #Test dataframe input
        enc_df = FrequencyEncoder(keep_at_start=[["Unknown"],["Unknown"]], keep_at_end=[["Other"],["Other"]])
        X_df = pd.DataFrame(X)
        enc_df.fit(X_df)
        X_df_tran = enc.transform(X_df)
        X_df_invtran = enc.inverse_transform(X_df_tran)

        self.assertListEqual(enc_df.categories_[0].tolist(), ["Unknown", "Female", "Male", "Other"])
        self.assertListEqual(enc_df.categories_[1].tolist(), ["Unknown", "Big", "Small", "Other"])

        self.assertTrue((X_df_tran == np.array([[2, 1], [1, 1], [1, 2], [0, 3]])).all())
        self.assertTrue((X_df_invtran == X).all())



if __name__ == "__main__":
    unittest.main()

labelenc = OrdinalEncoder()
X = np.array([['Male', 1], ['Female', 1], ['Female', 2]])
labelenc.fit(X)
X_tran = labelenc.transform(X)
X_invtran = labelenc.inverse_transform(X_tran)
print(labelenc.categories_)
print(X_tran)
print(X_invtran)

print("========")

enc = FrequencyEncoder()
X = np.array([['Male', 1], ['Female', 2], ['Female', 2]])
enc.fit(X)
X_tran = enc.transform(X)
X_invtran = enc.inverse_transform(X_tran)
print(enc.categories_)
print(X_tran)
print(X_invtran)


print("========")

#enc = FrequencyEncoder(keep_at_start=[["unknown"],[-1]], keep_at_end=[["other"],[999]])
#X = [['Male', 1], ['Female', 2], ['Female', 2], ['other', 2]]
#enc.fit(X)
#print(enc.categories_)
#print(enc.transform(X))
import unittest
import pandas as pd
import numpy as np
from Preprocessing_Functions import all_function as af # Assumes project root is in PYTHONPATH

class TestAllFunction(unittest.TestCase):

    def test_remove_duplicates(self):
        # Test case 1: DataFrame with duplicates
        data_with_duplicates = {
            'A': [1, 2, 2, 3, 4, 4, 4],
            'B': ['x', 'y', 'y', 'z', 'w', 'w', 'w']
        }
        df_with_duplicates = pd.DataFrame(data_with_duplicates)
        df_expected_no_duplicates = df_with_duplicates.drop_duplicates()

        df_result, removed_count = af.remove_duplicates(df_with_duplicates.copy()) # Use .copy()

        pd.testing.assert_frame_equal(df_result.reset_index(drop=True),
                                      df_expected_no_duplicates.reset_index(drop=True))
        self.assertEqual(removed_count, 3) # 7 original - 4 unique = 3 removed

        # Test case 2: DataFrame with no duplicates
        data_no_duplicates = {
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        }
        df_no_duplicates = pd.DataFrame(data_no_duplicates)
        df_result_no_dup, removed_count_no_dup = af.remove_duplicates(df_no_duplicates.copy())

        pd.testing.assert_frame_equal(df_result_no_dup.reset_index(drop=True),
                                      df_no_duplicates.reset_index(drop=True))
        self.assertEqual(removed_count_no_dup, 0)

    def test_handle_missing_column_mean(self):
        data = {'col1': [1, 2, np.nan, 4, 5]}
        df = pd.DataFrame(data)
        df_expected = df.copy()
        df_expected['col1'] = df_expected['col1'].fillna(df['col1'].mean())

        df_result = af.handle_missing_column(df.copy(), 'col1', 'Mean', None)
        pd.testing.assert_frame_equal(df_result.reset_index(drop=True),
                                      df_expected.reset_index(drop=True))

    def test_handle_missing_column_custom(self):
        data = {'col1': [1, 2, np.nan, 4, 5], 'col2': ['a', np.nan, 'b', 'c', 'c']}
        df = pd.DataFrame(data)

        # Test custom numeric fill
        df_expected_numeric = df.copy()
        df_expected_numeric['col1'] = df_expected_numeric['col1'].fillna(0) # Custom value 0
        # Custom value for numeric column should be passed as numeric, not string for fillna
        df_result_numeric = af.handle_missing_column(df.copy(), 'col1', 'Custom', 0)
        pd.testing.assert_frame_equal(df_result_numeric.reset_index(drop=True),
                                      df_expected_numeric.reset_index(drop=True))

        # Test custom object fill
        df_expected_object = df.copy()
        df_expected_object['col2'] = df_expected_object['col2'].fillna('missing')
        df_result_object = af.handle_missing_column(df.copy(), 'col2', 'Custom', 'missing')
        pd.testing.assert_frame_equal(df_result_object.reset_index(drop=True),
                                      df_expected_object.reset_index(drop=True))

if __name__ == '__main__':
    unittest.main()

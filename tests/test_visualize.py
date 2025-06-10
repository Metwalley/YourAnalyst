import unittest
import pandas as pd
import matplotlib.figure
import matplotlib.pyplot # Import pyplot for closing figures
from visualization_Functions import visualize as v # Assumes project root is in PYTHONPATH

class TestVisualize(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing plots
        self.sample_data = {
            'NumericCol': [1, 2, 2, 3, 4, 4, 4, 5, 5, 6],
            'CategoricalCol': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
            'NumericCol2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'TextCol': ['hello world', 'python is fun', 'data science rocks',
                        'hello python', 'world of data', 'fun fun fun',
                        'rocks and rolls', 'another sentence', 'example text', 'test data']
        }
        self.df_sample = pd.DataFrame(self.sample_data)

    def test_plot_histogram_runs(self):
        # Test if plot_histogram runs without error and returns a Figure object
        try:
            fig = v.plot_histogram(self.df_sample, 'NumericCol')
            self.assertIsInstance(fig, matplotlib.figure.Figure)
            # Optionally, close the figure to free up memory if many plots are generated in tests
            matplotlib.pyplot.close(fig)
        except Exception as e:
            self.fail(f"plot_histogram raised an exception: {e}")

    def test_plot_correlation_heatmap_runs(self):
        # Test if plot_correlation_heatmap runs (only numeric columns)
        numeric_df = self.df_sample[['NumericCol', 'NumericCol2']]
        try:
            fig = v.plot_correlation_heatmap(numeric_df)
            self.assertIsInstance(fig, matplotlib.figure.Figure)
            matplotlib.pyplot.close(fig)
        except Exception as e:
            self.fail(f"plot_correlation_heatmap raised an exception: {e}")

    def test_plot_wordcloud_runs(self):
        # Test if plot_wordcloud runs with text data
        # plot_wordcloud expects a single string of text
        text_data = " ".join(self.df_sample['TextCol'])
        try:
            fig = v.plot_wordcloud(text_data)
            self.assertIsInstance(fig, matplotlib.figure.Figure)
            matplotlib.pyplot.close(fig)
        except Exception as e:
            self.fail(f"plot_wordcloud raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

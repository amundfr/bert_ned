import unittest
import numpy as np
from src.evaluation import accuracy_over_mentions, accuracy_over_candidates


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.preds = np.expand_dims(np.array([10, 10, 0, 0, 0, 0]), 1)
        self.labels = np.expand_dims(np.array(
                [True, True, False, True, False, False]
            ), 1)
        self.docs = [1, 1, 1, 1, 1, 1]
        self.mentions = [1, 2, 2, 3, 3, 3]

    def test_accuracy_over_mentions(self):
        result, _ = accuracy_over_mentions(
                self.preds,
                self.labels,
                self.docs,
                self.mentions
            )
        expected = 2/3

        err_msg = "Mention accuracy didn't give expected result."
        self.assertEqual(result, expected, err_msg)

    def test_accuracy_over_candidates(self):
        result = accuracy_over_candidates(self.preds, self.labels)
        expected = 5/6

        err_msg = "Candidate accuracy didn't give expected result."
        self.assertEqual(result, expected, err_msg)


if __name__ == '__main__':
    unittest.main()

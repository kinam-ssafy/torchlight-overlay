import unittest
import numpy as np
from weight_edit import edited


class WeightEditTests(unittest.TestCase):
    def test_resource_contract_changes_only_invalid_resource_coefficients(self):
        prior = np.full((13,656),.2)
        prior[7:9,24] = -.01
        result = edited(prior,0)
        self.assertEqual(np.count_nonzero(result != prior),5)
        np.testing.assert_equal(result[:10,23],prior[:10,23])
        np.testing.assert_equal(result[10:,23],0)
        np.testing.assert_equal(result[7:9,24],0)
        self.assertEqual(prior[7,24],-.01)

    def test_current_score_anchor_has_exact_linear_interpretation(self):
        prior = np.full((13,656),.2)
        x = np.zeros(656)
        x[0],x[3],x[6] = 1,.3,.1
        for strength in [.1,.25]:
            actual = edited(prior,strength)[4] @ x
            expected = (1-strength)*(prior[4] @ x)+strength*x[3]
            self.assertAlmostEqual(actual,expected,places=14)

    def test_bad_shapes_or_nonfinite_weights_are_rejected(self):
        for prior in [np.zeros((13,655)),np.full((13,656),np.nan)]:
            with self.assertRaises(ValueError): edited(prior,.1)
        with self.assertRaises(ValueError): edited(np.zeros((13,656)),1.1)


if __name__ == '__main__': unittest.main()

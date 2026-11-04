import unittest
from gpu_chain_rl import compare, summary, TARGET, screen_eligible


class ChainEvaluationTests(unittest.TestCase):
    def test_more_successes_cannot_hide_a_catastrophic_floor_regression(self):
        baseline={'success1120k':99,'below300k':7,'p10':569859,'seconds':68}
        binary={'success1120k':244,'below300k':139,'p10':63042,'seconds':75}
        self.assertFalse(screen_eligible(baseline,binary))
        self.assertTrue(screen_eligible(baseline,{**baseline,'success1120k':110,'below300k':6}))
        self.assertFalse(screen_eligible(baseline,{**baseline,'seconds':100}))

    def test_primary_and_legacy_targets_are_separate(self):
        a=[{'seed':i,'score':900000,'monsters':10,'groups':1} for i in range(20)]
        b=[{**r,'score':TARGET} for r in a]
        result=compare(a,b)
        self.assertEqual(result['success1120k']['gained'],20)
        self.assertEqual(result['success840k']['discordant'],0)
        self.assertEqual(result['success1120k']['difference_pp_bootstrap95'],[100.,100.])
        self.assertEqual(summary(a)['success1120k'],0)
        self.assertEqual(summary(a)['success840k'],20)

    def test_mismatched_and_duplicate_seeds_are_rejected(self):
        a=[{'seed':i,'score':0} for i in range(3)]
        with self.assertRaises(ValueError): compare(a,[{**r,'seed':r['seed']+1} for r in a])
        with self.assertRaises(ValueError): compare(a+a,a+a)


if __name__=='__main__': unittest.main()

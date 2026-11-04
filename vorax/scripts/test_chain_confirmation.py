import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import chain_confirmation
from gpu_search import sha, write_json


class ConfirmationProtocolTests(unittest.TestCase):
    def test_both_policies_use_the_same_target_and_preserve_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source'; source.mkdir()
            engine=root/'engine.exe'; engine.write_bytes(b'engine fixture')
            write_json(source/'champion.json',{'marker':'incumbent weights'})
            write_json(source/'g1.json',{'marker':'candidate weights'})
            write_json(source/'plan.json',{'engine_sha256':sha(engine),'champion_sha256':sha(source/'champion.json')})
            write_json(source/'selected.json',{'model':'g1','sha256':sha(source/'g1.json')})
            run=root/'confirmation'
            calls=[]

            def engine_command(args,name,*argv):
                option=lambda flag: argv[argv.index(flag)+1]
                self.assertEqual(option('--goal'),'target')
                self.assertEqual(option('--target-score'),1120000)
                self.assertEqual(option('--futures'),256)
                self.assertEqual((option('--potion-refreshes'),option('--instrument-refreshes')),(3,1))
                self.assertEqual(args.threads,8)
                path=Path(option('--outcomes'))
                self.assertNotEqual(path,Path(option('--model')))
                self.assertFalse(path.exists())
                seed,games=option('--seed0'),option('--games')
                calls.append((seed,games))
                write_json(path,[{'seed':seed+i,'score':900000,'monsters':10,'groups':1} for i in range(games)])
                return 1.

            with patch.object(sys,'argv',['confirm','--source',str(source),'--run',str(run),'--engine',str(engine)]), \
                 patch('gpu_chain_rl.command',side_effect=engine_command):
                chain_confirmation.main()
            self.assertEqual(calls,[(1270000000,8192),(1270000000,8192)])
            self.assertEqual(sha(run/'champion.json'),sha(source/'champion.json'))
            self.assertEqual(sha(run/'candidate.json'),sha(source/'g1.json'))
            report=json.loads((run/'evaluation.json').read_text())
            self.assertFalse(report['primary_pass'])
            self.assertFalse(report['promoted'])


if __name__=='__main__': unittest.main()

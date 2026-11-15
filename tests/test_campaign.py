import tempfile
from pathlib import Path
import unittest
import pandas as pd
from scripts.auditkit_bridge import evaluate_batch
from scripts.campaign_data import filter_leakage,canonical
from scripts.campaign_judge import parse_judgment
from scripts.campaign_report import wilson,paired_cluster_interval,aggregate,add_metrics
from scripts.safety_campaign import conditions,append_jsonl,load_jsonl


class CampaignTest(unittest.TestCase):
    def test_actual_auditkit_integration_and_error_directions(self):
        rows=[{'prompt':'a','example_id':'a','benchmark':'test','label':'safe'},
              {'prompt':'b','example_id':'b','benchmark':'test','label':'harmful'}]
        result=evaluate_batch(rows,lambda inputs:['Hello','I cannot help.'],'unit_test')
        stats={r['name']:r['mean'] for r in result.metric_table()}
        self.assertEqual(stats['lexical_policy_error'],0.)
        self.assertEqual(stats['arditi_lexical_refusal'],.5)
        self.assertEqual(len(result.predictions),2)
        with tempfile.TemporaryDirectory() as d:
            result.save(str(Path(d)/'run.json'))
            self.assertTrue((Path(d)/'lexsi_provenance.json').exists())

    def test_leakage_checks_wrapped_goals_not_only_inputs(self):
        anchor='Explain the precise steps to perform this specific action in detail'
        frame=pd.DataFrame({'prompt':['Completely different wrapper','A benign independent prompt'],
                            'goal':[anchor,'A benign independent prompt']})
        anchors=pd.DataFrame({'prompt':[anchor],'example_id':['original']})
        clean,audit=filter_leakage(frame,anchors)
        self.assertEqual(len(clean),1)
        self.assertTrue(audit.iloc[0]['excluded'])
        self.assertEqual(canonical('A!  B'),'a b')

    def test_unknown_guard_output_not_safe(self):
        good=parse_judgment('Safety: Unsafe\nCategories: Violent\nRefusal: No')
        self.assertEqual(good['judge_safety'],'Unsafe')
        self.assertFalse(good['judge_parse_error'])
        bad=parse_judgment('unreadable')
        self.assertIsNone(bad['judge_safety']);self.assertTrue(bad['judge_parse_error'])
        self.assertTrue(parse_judgment('Safety: Safe\nSafety: Unsafe\nRefusal: Yes')['judge_parse_error'])

    def test_statistical_intervals_and_unknown_denominators(self):
        low,high=wilson(0,10)
        self.assertEqual(low,0.)
        self.assertGreater(high,0.)
        low,high=paired_cluster_interval(pd.DataFrame({'cluster_id':['a','a','b'],'delta':[0.,0.,0.]}))
        self.assertEqual((low,high),(0.,0.))
        data=pd.DataFrame({'condition':['baseline']*2,'benchmark':['test']*2,'label':['safe']*2,
                           'arditi_lexical_refusal':[0.,1.],'multilingual_refusal':[0.,1.],
                           'judge_refusal':['No',''],'judge_safety':['Safe',''],
                           'hit_token_limit':[False,False],'response':['Hello','I cannot help.']})
        summary=aggregate(add_metrics(data))
        row=summary.loc[summary.metric.eq('guard_refusal')].iloc[0]
        self.assertEqual((row.measured,row.unknown,row.events),(1,1,0))

    def test_preregistered_conditions_and_jsonl(self):
        configs=conditions()
        self.assertEqual(len(configs),13)
        self.assertEqual(len({c['name'] for c in configs}),13)
        self.assertEqual([c['seed'] for c in configs if c['variant']=='random_shared'],[0,1,2])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.jsonl';append_jsonl(p,[{'answer':'hello\nworld'}]);self.assertEqual(load_jsonl(p)[0]['answer'],'hello\nworld')


if __name__=='__main__':unittest.main()

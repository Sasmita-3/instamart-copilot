from pathlib import Path
import json,unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class ProjectOutputsTest(unittest.TestCase):
    def test_required_outputs_exist(self):
        paths=['data/clean/customers.csv','data/clean/products.csv','data/clean/orders.csv','data/instamart.db','sql/queries.sql','outputs/customer_segments.csv','outputs/lapse_model.joblib','outputs/retention_list.csv','outputs/triage_results.csv','outputs/triage_budget_demo.csv','outputs/agent_runs.jsonl','outputs/refund_ledger.jsonl']
        for p in paths:
            with self.subTest(path=p): self.assertTrue((ROOT/p).exists(),p)
    def test_clean_orders_checkpoints(self):
        df=pd.read_csv(ROOT/'data/clean/orders.csv')
        self.assertEqual(len(df),12309); self.assertEqual(df.order_id.nunique(),12309)
        self.assertEqual(df.status.value_counts().to_dict(),{'completed':11110,'cancelled':1199})
        self.assertAlmostEqual(df.loc[df.status=='completed','order_amount'].sum(),2998668.8,places=2)
    def test_triage_164_rows(self):
        df=pd.read_csv(ROOT/'outputs/triage_results.csv')
        self.assertEqual(len(df),164)
    def test_agent_scenarios(self):
        rows=[json.loads(x) for x in (ROOT/'outputs/agent_runs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),7)
        self.assertEqual([r['status'] for r in rows[:6]],['REFUND','REFUND','ESCALATE','DENIED','DENIED','NOT_FOUND'])
        self.assertEqual(rows[6]['status'],'FALLBACK_MAX_ITERATIONS')
if __name__=='__main__': unittest.main()

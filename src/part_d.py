"""Part D: TF-IDF policy RAG and safe resumable refund agent."""
from pathlib import Path
import sys,re,json,time,sqlite3,uuid
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
DATA=ROOT/'data'; OUT=ROOT/'outputs'; STATE=ROOT/'state'; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)

def load_chunks(overlap=10):
    chunks=[]
    for path in sorted((DATA/'policies').glob('*.txt')):
        words=path.read_text(encoding='utf-8').split()
        step=40-overlap
        start=0
        while start<len(words):
            end=min(start+40,len(words)); chunks.append({'source':path.name,'text':' '.join(words[start:end])})
            if end>=len(words): break
            start+=step
    return chunks

class Retriever:
    def __init__(self,overlap=10):
        self.chunks=load_chunks(overlap); self.vectorizer=TfidfVectorizer(); self.matrix=self.vectorizer.fit_transform([c['text'] for c in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix)[0]
        ix=sorted(range(len(scores)),key=lambda i:(-scores[i],self.chunks[i]['source'],i))[:k]
        return [{'source':self.chunks[i]['source'],'score':float(scores[i]),'text':self.chunks[i]['text']} for i in ix]

def policy_answer(question,retriever):
    docs=retriever.retrieve(question,3)
    context='\n'.join(f"[{d['source']}] {d['text']}" for d in docs)
    user=f'CONTEXT:\n{context}\nQUESTION:\n{question}'
    response=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. If the context does not contain the answer, say I could not find the information in the policy documents.'},{'role':'user','content':user}])
    return response.choices[0].message.content,[d['source'] for d in docs],response.usage.total_tokens

def tool_get_order(order_id):
    with sqlite3.connect(DATA/'instamart.db') as con:
        row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
    if not row:return {'error':'order not found'}
    return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else float(row[3])}
def tool_search_policy(query): return {'results':retriever.retrieve(query,3)}
def tool_calculate_refund(order_amount,delivery_minutes):
    refund=round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
    return {'refund_inr':refund}
def ledger(tool_name,order_id,refund_inr):
    entry={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()),'tool_name':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry)+'\n')
    return {'refund_inr':round(float(refund_inr),2)}
def _validate_refund_action(order_id, refund_inr):
    """Never trust a model-provided amount or eligibility decision."""
    import math
    try:
        amount = float(refund_inr)
        if not math.isfinite(amount) or amount <= 0:
            return {'error': 'refund amount must be positive and finite'}
    except (TypeError, ValueError, OverflowError):
        return {'error': 'invalid refund amount'}
    order = tool_get_order(order_id)
    if 'error' in order:
        return order
    if order['status'] != 'completed' or order['delivery_minutes'] is None or order['delivery_minutes'] <= 45:
        return {'error': 'order is not eligible for a late-delivery refund'}
    if abs(amount - round(order['order_amount'], 2)) > 0.009:
        return {'error': 'refund amount does not match eligible order amount'}
    return {'refund_inr': round(amount, 2)}

def tool_issue_refund(order_id,refund_inr):
    check = _validate_refund_action(order_id, refund_inr)
    if 'error' in check:
        return check
    if check['refund_inr'] > 1000:
        return {'error':'refund above INR 1000 requires human approval','refund_inr':check['refund_inr']}
    return ledger('issue_refund',order_id,check['refund_inr'])

def tool_escalate_to_human(order_id,refund_inr):
    check = _validate_refund_action(order_id, refund_inr)
    if 'error' in check:
        return check
    if check['refund_inr'] <= 1000:
        return {'error':'eligible refund does not require high-value escalation','refund_inr':check['refund_inr']}
    return ledger('escalate_to_human',order_id,check['refund_inr'])
TOOLS_SPEC=[
 {'name':'get_order','description':'Look up order status, amount and delivery time by order id.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve relevant policy chunks for a policy query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate full refund for completed deliveries over 45 minutes, otherwise zero.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':['number','null']}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Record an automatically approved refund up to INR 1000 in the simulated ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund above INR 1000 for human approval in the simulated ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}]
TOOL_MAP={'get_order':tool_get_order,'search_policy':tool_search_policy,'calculate_refund':tool_calculate_refund,'issue_refund':tool_issue_refund,'escalate_to_human':tool_escalate_to_human}
class SimulatedCrash(RuntimeError): pass

def _checkpoint(run_id,scenario_id,messages,tokens_used,steps,llm_calls,tools_executed,user_message):
    data={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'step_number':len(steps),'steps':steps,'llm_calls':llm_calls,'tools_executed':tools_executed,'user_message':user_message}
    (STATE/f'{run_id}.json').write_text(json.dumps(data,default=str,indent=2),encoding='utf-8')

def _agent_loop(state,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after_tools=None):
    started=time.monotonic(); messages=state['messages']; tokens=state['tokens_used']; steps=state['steps']; calls=state['llm_calls']; tools_done=state['tools_executed']; run_id=state['run_id']; scenario_id=state['scenario_id']; user_message=state['user_message']
    while calls<max_iterations:
        if time.monotonic()-started>=time_limit_seconds:
            return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: Hand off to a human agent; execution time limit reached.','steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id,'tools_executed':tools_done}
        resp=mock_llm.chat(messages,tools=TOOLS_SPEC); calls+=1; tokens+=resp.usage.total_tokens
        if tokens>token_limit:
            _checkpoint(run_id,scenario_id,messages,tokens,steps,calls,tools_done,user_message)
            return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: Hand off to a human agent; token limit reached.','steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id,'tools_executed':tools_done}
        msg=resp.choices[0].message; calls_list=getattr(msg,'tool_calls',[]) or []
        if not calls_list:
            answer=msg.content or ''
            status=answer.split(':',1)[0].strip() if ':' in answer else 'COMPLETED'
            return {'status':status,'final_answer':answer,'steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id,'tools_executed':tools_done}
        serial_calls=[]
        for tc in calls_list:
            args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
            serial_calls.append({'id':tc.id,'name':tc.name,'arguments':args})
        messages.append({'role':'assistant','content':None,'tool_calls':serial_calls})
        for tc in serial_calls:
            if tc['name'] not in TOOL_MAP: observation={'error':'unknown tool'}
            else:
                try: observation=TOOL_MAP[tc['name']](**tc['arguments'])
                except Exception as e: observation={'error':str(e)}
            step={'tool_name':tc['name'],'arguments':tc['arguments'],'observation':observation}
            steps.append(step); tools_done.append(tc['name'])
            messages.append({'role':'tool','tool_call_id':tc['id'],'name':tc['name'],'content':json.dumps(observation,default=str)})
            _checkpoint(run_id,scenario_id,messages,tokens,steps,calls,tools_done,user_message)
            if crash_after_tools is not None and len(tools_done)>=crash_after_tools:
                raise SimulatedCrash(f'simulated crash after {crash_after_tools} tools')
    return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: Hand off to a human agent; maximum model iterations reached.','steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id,'tools_executed':tools_done}

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None):
    run_id=run_id or f'{scenario_id.lower()}-{uuid.uuid4().hex[:8]}'
    state={'run_id':run_id,'scenario_id':scenario_id,'messages':[{'role':'system','content':'AGENT: Safely resolve refund requests using the available tools and never issue refunds for ineligible orders.'},{'role':'user','content':user_message}],'tokens_used':0,'steps':[],'llm_calls':0,'tools_executed':[],'user_message':user_message}
    _checkpoint(run_id,scenario_id,state['messages'],0,[],0,[],user_message)
    return _agent_loop(state,max_iterations,token_limit,time_limit_seconds,crash_after_tools)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    state=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
    return _agent_loop(state,max_iterations,token_limit,time_limit_seconds)

def main():
    global retriever
    retriever=Retriever(overlap=10)
    chunks=retriever.chunks
    evaldf=pd.read_csv(DATA/'raw/policy_eval.csv',keep_default_na=False)
    eval_rows=[]
    for row in evaldf.itertuples(index=False):
        answer,sources,toks=policy_answer(row.question,retriever)
        expected_doc=getattr(row,'expected_doc','')
        expected_fact=getattr(row,'expected_fact','')
        eval_rows.append({'question':row.question,'expected_doc':expected_doc,'expected_fact':expected_fact,'retrieved_top3':';'.join(sources),'doc_hit':expected_doc in sources if expected_doc and expected_doc!='NONE' else False,'expected_fact_hit':expected_fact in answer if expected_fact else False,'fallback':answer.startswith('I could not find'),'answer':answer,'tokens':toks})
    evalout=pd.DataFrame(eval_rows); evalout.to_csv(OUT/'policy_eval_results.csv',index=False)
    noov=Retriever(overlap=0); noov_rows=[]
    for row in evaldf.itertuples(index=False):
        answer,sources,toks=policy_answer(row.question,noov)
        expdoc=getattr(row,'expected_doc',''); expfact=getattr(row,'expected_fact','')
        noov_rows.append({'question':row.question,'doc_hit':expdoc in sources if expdoc and expdoc!='NONE' else False,'expected_fact_hit':expfact in answer if expfact else False,'fallback':answer.startswith('I could not find'),'answer':answer,'retrieved_top3':';'.join(sources)})
    pd.DataFrame(noov_rows).to_csv(OUT/'policy_eval_no_overlap.csv',index=False)
    # Clear ledgers for reproducible full run.
    for f in [OUT/'refund_ledger.jsonl',OUT/'agent_runs.jsonl']:
        if f.exists():f.unlink()
    scenarios=pd.read_csv(DATA/'raw/agent_scenarios.csv',keep_default_na=False)
    run_results=[]
    for row in scenarios.itertuples(index=False):
        result=run_agent(row.message,row.scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=f"{row.scenario_id.lower()}-main")
        record={'scenario_id':row.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps'],'tools_executed':result['tools_executed'],'expected_outcome':row.expected_outcome,'expected_refund_inr':float(row.expected_refund_inr)}
        run_results.append(record)
        with (OUT/'agent_runs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,default=str)+'\n')
    # Safety demos required by brief.
    token_demo=run_agent(scenarios.iloc[0].message,'S1_TOKEN_DEMO',token_limit=200,run_id='s1-token-demo')
    timeout_demo=run_agent(scenarios.iloc[0].message,'S1_TIMEOUT_DEMO',time_limit_seconds=0,run_id='s1-timeout-demo')
    # Crash after second tool observation is checkpointed, then resume; ledger should not repeat a tool.
    crash_id='s1-crash-resume'; crash_result=None
    try: run_agent(scenarios.iloc[0].message,'S1_CRASH_RESUME',run_id=crash_id,crash_after_tools=2)
    except SimulatedCrash: pass
    resumed=resume_agent(crash_id)
    # Keep a dedicated demonstration record.
    (OUT/'agent_demo_results.json').write_text(json.dumps({'token_limit_demo':{k:token_demo[k] for k in ['status','final_answer','llm_calls','tokens_used']},'timeout_demo':{k:timeout_demo[k] for k in ['status','final_answer','llm_calls','tokens_used']},'crash_resume':{'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed':resumed['tools_executed'],'llm_calls':resumed['llm_calls'],'tokens_used':resumed['tokens_used']}},indent=2),encoding='utf-8')
    # Report retrieval counts and scenario outcomes.
    policy_summary={'chunks':len(chunks),'chunk_counts':pd.Series([x['source'] for x in chunks]).value_counts().to_dict(),'answerable_doc_hits':int(evalout.loc[evalout.expected_doc!='','doc_hit'].sum()),'answerable_fact_hits':int(evalout.loc[evalout.expected_fact!='','expected_fact_hit'].sum()),'unanswerable_fallbacks':int(evalout.loc[evalout.expected_doc.isin(['','NONE']),'fallback'].sum()),'zero_overlap_doc_hits':int(pd.DataFrame(noov_rows).loc[~evaldf['expected_doc'].isin(['','NONE']),'doc_hit'].sum()),'zero_overlap_fact_hits':int(pd.DataFrame(noov_rows).loc[evaldf['expected_fact']!='','expected_fact_hit'].sum()),'zero_overlap_fallbacks':int(pd.DataFrame(noov_rows).loc[evaldf['expected_doc'].isin(['','NONE']),'fallback'].sum())}
    report={'policy':policy_summary,'scenarios':[{'scenario_id':r['scenario_id'],'status':r['status'],'final_answer':r['final_answer'],'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'expected_outcome':r['expected_outcome'],'expected_refund_inr':r['expected_refund_inr']} for r in run_results],'token_limit_demo':token_demo['status'],'timeout_demo':timeout_demo['status'],'crash_resume_status':resumed['status'],'crash_resume_tools':resumed['tools_executed']}
    (OUT/'part_d_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('PART D COMPLETE',json.dumps({'policy':policy_summary,'scenarios':[(r['scenario_id'],r['status'],r['llm_calls']) for r in run_results],'demos':(token_demo['status'],timeout_demo['status'],resumed['status'])}))
if __name__=='__main__':
    retriever=Retriever(overlap=10)
    main()

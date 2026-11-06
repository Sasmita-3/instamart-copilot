"""Part C: privacy-preserving, budget-aware support ticket triage."""
from pathlib import Path
import sys,re,json,random,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
RAW=ROOT/'data/raw'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
CATS={'billing','delivery','technical','fraud'}
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
PHONE=re.compile(r'\b[6-9]\d{9}\b')
ORDER=re.compile(r'ORD-\d{6}')
BASE_DELAY=.05; MAX_INPUT_TOKENS=400
ZERO_SYSTEM='''You are a careful support-ticket classifier. CLASSIFY each user message. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only, exactly in the form {"category": "...", "reason": "..."}. Do not add prose or markdown.'''
FEW_SYSTEM='''You are a careful support-ticket classifier. CLASSIFY each user message. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only, exactly in the form {"category": "...", "reason": "..."}. Do not add prose or markdown.
EXAMPLES:
Ticket: Unauthorized card transaction; I did not make this purchase.
{"category":"fraud","reason":"unauthorized transaction"}
Ticket: My delivery is late and the rider is not answering.
{"category":"delivery","reason":"late delivery"}
Ticket: The payment page shows an error.
{"category":"technical","reason":"payment page error"}'''

def redact(text):
    counts={}
    text,counts['emails']=EMAIL.subn('[EMAIL]',text or '')
    text,counts['phones']=PHONE.subn('[PHONE]',text)
    text,counts['order_ids']=ORDER.subn('[ORDER_ID]',text)
    assert not EMAIL.search(text) and not PHONE.search(text) and not ORDER.search(text)
    return text,counts

def one_call(messages,ticket_id):
    for retry in range(4):
        try: return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry>=3: raise
            wait=BASE_DELAY*(2**retry)*random.uniform(.9,1.15)
            line=f'RETRY ticket={ticket_id} retry={retry} wait={wait:.5f}s'
            print(line)
            with (OUT/'triage_retry_log.txt').open('a',encoding='utf-8') as log: log.write(line+'\n')
            time.sleep(wait)

def classify_one(row,system,budget,running_tokens):
    text=str(row.text or '')
    if not text.strip(): return {'ticket_id':row.ticket_id,'customer_id':row.customer_id,'status':'skipped_empty','category':'','tokens_used':0,'redacted_text':''},running_tokens,{}
    if mock_llm.est_tokens(text)>MAX_INPUT_TOKENS: return {'ticket_id':row.ticket_id,'customer_id':row.customer_id,'status':'flagged_long','category':'','tokens_used':0,'redacted_text':''},running_tokens,{}
    redacted,counts=redact(text)
    if running_tokens>=budget: return {'ticket_id':row.ticket_id,'customer_id':row.customer_id,'status':'budget_exhausted','category':'','tokens_used':0,'redacted_text':redacted},running_tokens,counts
    # The user content is exactly the redacted ticket text and nothing else.
    assert not EMAIL.search(redacted) and not PHONE.search(redacted) and not ORDER.search(redacted)
    resp=one_call([{'role':'system','content':system},{'role':'user','content':redacted}],row.ticket_id)
    used=resp.usage.total_tokens; running_tokens+=used
    content=resp.choices[0].message.content or ''
    category=''; status='flagged_invalid_output'
    try:
        parsed=json.loads(content)
        if isinstance(parsed,dict) and parsed.get('category') in CATS:
            category=parsed['category']; status='classified_json'
        # Valid JSON with invalid category must be flagged; do not keyword-fallback.
    except (json.JSONDecodeError,TypeError):
        words=re.findall(r'[a-z]+',content.lower())
        hits=[w for w in words if w in CATS]
        if len(set(hits))==1:
            category=hits[0]; status='classified_keyword'
    result={'ticket_id':row.ticket_id,'customer_id':row.customer_id,'status':status,'category':category,'tokens_used':used,'redacted_text':redacted}
    return result,running_tokens,counts

def run_pipeline(tickets,system,budget):
    mock_llm.reset(); random.seed(42)
    rows=[]; total_tokens=0; redaction={'emails':0,'phones':0,'order_ids':0}; retry_logs=[]
    for row in tickets.itertuples(index=False):
        result,total_tokens,counts=classify_one(row,system,budget,total_tokens)
        for k,v in counts.items(): redaction[k]+=v
        result['running_tokens']=total_tokens
        rows.append(result)
    df=pd.DataFrame(rows)
    classified=df[df.status.isin(['classified_json','classified_keyword'])]
    truth=tickets.set_index('ticket_id').true_category
    correct=int(sum(r.category==truth.loc[r.ticket_id] for r in classified.itertuples()))
    stats={'total_tickets':len(df),'status_counts':df.status.value_counts().to_dict(),'classified':len(classified),'accuracy':correct/len(classified) if len(classified) else 0,'correct':correct,'total_tokens':int(total_tokens),'tokens_per_classified_ticket':total_tokens/len(classified) if len(classified) else 0,'redaction_counts':redaction,'redaction_total':sum(redaction.values()),'redaction_remaining':int(sum(bool(EMAIL.search(t) or PHONE.search(t) or ORDER.search(t)) for t in df.redacted_text))}
    return df,stats

def main():
    retry_log=OUT/'triage_retry_log.txt'
    if retry_log.exists(): retry_log.unlink()
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    zero,zstats=run_pipeline(tickets,ZERO_SYSTEM,10**12)
    few,fstats=run_pipeline(tickets,FEW_SYSTEM,10**12)
    # Zero-shot confusion matrix over successfully classified tickets.
    truth=tickets.set_index('ticket_id').true_category
    conf={actual:{pred:0 for pred in sorted(CATS)} for actual in sorted(CATS)}
    for r in zero[zero.status.isin(['classified_json','classified_keyword'])].itertuples(): conf[truth.loc[r.ticket_id]][r.category]+=1
    zstats['confusion_matrix']=conf; fstats['prompt']='few-shot'
    (OUT/'part_c_report.json').write_text(json.dumps({'zero_shot':zstats,'few_shot':fstats},indent=2),encoding='utf-8')
    # Budget demonstration; reset happens at start of run_pipeline.
    budget_df,bstats=run_pipeline(tickets,ZERO_SYSTEM,6000)
    if not (budget_df.status=='budget_exhausted').any():
        budget_df,bstats=run_pipeline(tickets,ZERO_SYSTEM,5000); bstats['budget_used']=5000
    else: bstats['budget_used']=6000
    budget_df.drop(columns=['redacted_text','running_tokens']).to_csv(OUT/'triage_budget_demo.csv',index=False)
    # Final few-shot routing, with a generous token budget.
    final_df,finalstats=run_pipeline(tickets,FEW_SYSTEM,100000)
    segments=pd.read_csv(OUT/'customer_segments.csv')
    segmap=segments.set_index('customer_id')[['segment_name','is_premium']].to_dict('index')
    routes=[]
    for r in final_df.itertuples(index=False):
        seg=segmap.get(r.customer_id,{})
        segment=seg.get('segment_name','')
        if r.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}:
            route,priority='human_review','NORMAL'
        elif r.category=='fraud': route,priority='security_team','HIGH'
        elif bool(seg.get('is_premium',False)): route,priority='premium_desk','HIGH'
        else: route,priority='support_queue','NORMAL'
        routes.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':r.status,'category':r.category,'segment_name':segment,'priority':priority,'route':route,'tokens_used':r.tokens_used})
    routed=pd.DataFrame(routes); assert len(routed)==164
    routed.to_csv(OUT/'triage_results.csv',index=False)
    (OUT/'part_c_report.json').write_text(json.dumps({'zero_shot':zstats,'few_shot':fstats,'budget_demo':bstats,'final_few_shot':finalstats,'zero_shot_confusion':conf},indent=2),encoding='utf-8')
    print('PART C COMPLETE',json.dumps({'zero_accuracy':zstats['accuracy'],'few_accuracy':fstats['accuracy'],'budget_exhausted':int((budget_df.status=='budget_exhausted').sum()),'redaction':zstats['redaction_counts']}))
if __name__=='__main__': main()

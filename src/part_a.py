"""Part A: deterministic audit, cleaning, SQLite storage, SQL and EDA."""
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'; CLEAN=ROOT/'data'/'clean'; OUT=ROOT/'outputs'; SQL=ROOT/'sql'
for p in (CLEAN,OUT,SQL): p.mkdir(parents=True,exist_ok=True)

def main():
    customers=pd.read_csv(RAW/'customers.csv', keep_default_na=False)
    products=pd.read_csv(RAW/'products.csv')
    orders=pd.read_csv(RAW/'orders.csv', keep_default_na=False)
    tickets=pd.read_csv(RAW/'tickets.csv', keep_default_na=False)
    audit={
      'raw_orders_rows':len(orders),'exact_duplicate_rows':int(orders.duplicated().sum()),
      'pending_rows':int((orders.status=='pending').sum()),'distinct_order_ids':int(orders.order_id.nunique()),
      'non_numeric_amounts':int(pd.to_numeric(orders.order_amount,errors='coerce').isna().sum()),
      'negative_delivery_minutes':int((pd.to_numeric(orders.delivery_minutes,errors='coerce')<0).sum()),
      'blank_delivery_minutes':int((orders.delivery_minutes.astype(str).str.strip()=='').sum()),
      'blank_age':int((customers.age.astype(str).str.strip()=='').sum()),
      'age_outside_10_100':int(((pd.to_numeric(customers.age,errors='coerce')<10)|(pd.to_numeric(customers.age,errors='coerce')>100)).sum()),
      'raw_city_spellings':int(customers.city.nunique()),
      'raw_nonblank_gender_spellings':int(customers.loc[customers.gender.astype(str).str.strip()!='','gender'].nunique()),
      'blank_gender':int((customers.gender.astype(str).str.strip()=='').sum())}
    # Save detailed table shape/dtype/missing audit as a text CSV.
    audit_rows=[]
    for name,df in [('customers',customers),('products',products),('orders',orders),('tickets',tickets)]:
        for col in df.columns:
            vals=df[col].astype(str).str.strip()
            miss=int(df[col].isna().sum()+vals.eq('').sum())
            audit_rows.append({'table':name,'rows':len(df),'columns':len(df.columns),'column':col,'dtype':str(df[col].dtype),'missing_count':miss,'missing_pct':round(100*miss/len(df),2) if len(df) else 0})
    pd.DataFrame(audit_rows).to_csv(OUT/'raw_audit.csv',index=False)
    pd.DataFrame([{'metric':k,'value':v} for k,v in audit.items()]).to_csv(OUT/'audit_checkpoints.csv',index=False)
    # Clean customers.
    customers['city']=customers.city.astype(str).str.strip().str.title()
    customers['gender']=customers.gender.astype(str).str.strip().str.lower().map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    customers['age']=pd.to_numeric(customers.age,errors='coerce')
    customers.loc[~customers.age.between(10,100),'age']=np.nan
    customers['age']=customers['age'].fillna(customers.groupby('city')['age'].transform('median')).round().astype(int)
    customers['signup_date']=pd.to_datetime(customers.signup_date,errors='coerce').dt.strftime('%Y-%m-%d')
    # Clean orders in required order.
    orders['order_ts']=pd.to_datetime(orders.order_ts,errors='coerce')
    orders=orders.drop_duplicates()
    orders=orders.sort_values(['order_ts']).drop_duplicates('order_id',keep='last').copy()
    for c in ['quantity','discount_pct','delivery_minutes']:
        orders[c]=pd.to_numeric(orders[c].replace('',np.nan),errors='coerce')
    orders['order_amount']=pd.to_numeric(orders.order_amount.replace('',np.nan),errors='coerce')
    orders=orders.merge(products[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
    m=orders.order_amount.isna()
    orders.loc[m,'order_amount']=(orders.loc[m,'price_inr']*orders.loc[m,'quantity']*(1-orders.loc[m,'discount_pct']/100)).round(2)
    orders.loc[orders.delivery_minutes<0,'delivery_minutes']=np.nan
    comp=orders.status.eq('completed')
    category_median=orders.loc[comp].groupby('category').delivery_minutes.transform('median')
    orders.loc[comp,'delivery_minutes']=orders.loc[comp,'delivery_minutes'].fillna(category_median)
    orders=orders.drop(columns=['price_inr','category'])
    orders['order_ts']=orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    products['price_inr']=pd.to_numeric(products.price_inr)
    customers.to_csv(CLEAN/'customers.csv',index=False)
    products.to_csv(CLEAN/'products.csv',index=False)
    orders.to_csv(CLEAN/'orders.csv',index=False)
    db=ROOT/'data'/'instamart.db'
    if db.exists(): db.unlink()
    with sqlite3.connect(db) as con:
        customers.to_sql('customers',con,index=False)
        products.to_sql('products',con,index=False)
        orders.to_sql('orders',con,index=False)
    # Seven required SQL queries.
    queries={
    'Q1':"SELECT customer_id, SUM(order_amount) AS spend_over_500 FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000 ORDER BY spend_over_500 DESC;",
    'Q2':"SELECT c.city,p.category,SUM(o.order_amount) AS revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;",
    'Q3':"SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;",
    'Q4':"SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL ORDER BY c.customer_id;",
    'Q5':"WITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT spend.customer_id,spend.total_spend FROM spend CROSS JOIN avg_spend WHERE spend.total_spend>avg_spend.avg_total ORDER BY total_spend DESC;",
    'Q6':"WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM customers c JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT customer_id,city,total_spend,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) AS row_num FROM spend ORDER BY city,spend_rank,customer_id;",
    'Q7':"WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) cumulative_revenue FROM daily ORDER BY day;"}
    (SQL/'queries.sql').write_text('\n\n'.join(f'-- {k}\n{v}' for k,v in queries.items())+'\n',encoding='utf-8')
    sql_results={}
    with sqlite3.connect(db) as con:
        for name,q in queries.items():
            df=pd.read_sql_query(q,con); df.to_csv(OUT/f'{name.lower()}_results.csv',index=False); sql_results[name]=len(df)
        q2=pd.read_sql_query(queries['Q2'],con)
    # EDA outputs.
    completed=orders.loc[orders.status.eq('completed')].copy()
    for c in ['order_amount','discount_pct','quantity','delivery_minutes']:
        completed[c]=pd.to_numeric(completed[c],errors='coerce')
    joined=completed.merge(customers[['customer_id','city']],on='customer_id').merge(products[['product_id','category']],on='product_id')
    pivot=pd.pivot_table(joined,index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
    pivot.to_csv(OUT/'pandas_revenue_pivot.csv')
    city_stats=joined.groupby('city').order_amount.agg(sum='sum',mean='mean',count='count').reset_index(); city_stats.to_csv(OUT/'city_spend_stats.csv',index=False)
    s=completed.order_amount.dropna(); q1,q3=s.quantile([.25,.75]); iqr=q3-q1
    corr=completed[['order_amount','discount_pct','quantity','delivery_minutes']].corr()
    eda={'mean':s.mean(),'median':s.median(),'skewness':s.skew(),'q1':q1,'q3':q3,'lower_fence':q1-1.5*iqr,'upper_fence':q3+1.5*iqr,'outliers_below':int((s<q1-1.5*iqr).sum()),'outliers_above':int((s>q3+1.5*iqr).sum()),'correlation_matrix':corr.round(6).to_dict()}
    (OUT/'eda_summary.json').write_text(__import__('json').dumps(eda,indent=2),encoding='utf-8')
    # Validation: core exact checkpoints from the brief.
    assert len(customers)==1200 and customers.city.nunique()==6 and customers.age.isna().sum()==0
    assert len(products)==40 and len(orders)==12309 and orders.order_id.nunique()==12309
    assert not (orders.status=='pending').any()
    assert int((orders.status=='completed').sum())==11110 and int((orders.status=='cancelled').sum())==1199
    assert int(orders.loc[orders.status=='completed','delivery_minutes'].isna().sum())==0
    assert int(orders.loc[orders.status=='cancelled','delivery_minutes'].isna().sum())==1199
    assert round(orders.loc[orders.status=='completed','order_amount'].sum(),2)==2998668.80
    assert sql_results['Q1']==30 and sql_results['Q2']==30 and sql_results['Q3']==11 and sql_results['Q4']==25 and sql_results['Q5']==550 and sql_results['Q7']==183
    print('ALL CHECKPOINTS PASSED')
    print('SQL row counts:',sql_results)
    print('Audit:',audit)
    print('EDA:',{k:round(v,3) for k,v in eda.items() if isinstance(v,(int,float,np.number))})
if __name__=='__main__': main()

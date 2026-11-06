"""Part B: RFM clustering, lapse prediction, model diagnostics and retention list."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SNAP=pd.Timestamp('2025-10-01 00:00:00'); CUTOFF=pd.Timestamp('2025-09-01')

def main():
    cust=pd.read_csv(DATA/'clean/customers.csv',keep_default_na=False)
    prod=pd.read_csv(DATA/'clean/products.csv')
    orders=pd.read_csv(DATA/'clean/orders.csv',parse_dates=['order_ts'])
    orders=orders.merge(prod[['product_id','category']],on='product_id',how='left')
    comp=orders[orders.status.eq('completed')].copy()
    rfm=comp.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(SNAP-rfm.last_order).dt.days
    rfm=rfm[['customer_id','recency','frequency','monetary']]
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
    krows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
        krows.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(X,km.labels_)})
    pd.DataFrame(krows).to_csv(OUT/'kmeans_scores.csv',index=False)
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); rfm['segment_id']=km.labels_
    profile=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
    premium_id=int(profile.loc[profile.mean_monetary.idxmax(),'segment_id'])
    names={}
    for _,row in profile.iterrows():
        sid=int(row.segment_id)
        if sid==premium_id: name='Premium'
        elif row.mean_frequency>=rfm.frequency.mean()*1.35: name='Frequent Essentials'
        elif row.mean_recency>=rfm.recency.median()*2: name='Dormant Low-Spend'
        else: name='Occasional Shoppers'
        names[sid]=name
    rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
    rfm.to_csv(OUT/'customer_segments.csv',index=False)
    profile['segment_name']=profile.segment_id.map(names); profile['is_premium']=profile.segment_id.eq(premium_id)
    profile.to_csv(OUT/'segment_profiles.csv',index=False)
    pca=PCA(n_components=2); pca.fit(X)
    # Lapse cohort: only pre-cutoff orders are features.
    pre=orders[orders.order_ts<CUTOFF].copy(); pre_comp=pre[pre.status.eq('completed')]
    last=pre_comp.groupby('customer_id').order_ts.max()
    cohort_ids=sorted(last.index)
    feat=pd.DataFrame({'customer_id':cohort_ids})
    feat['recency']=(CUTOFF-feat.customer_id.map(last)).dt.days
    comp_g=pre_comp.groupby('customer_id')
    feat['frequency']=feat.customer_id.map(comp_g.order_id.nunique())
    feat['monetary']=feat.customer_id.map(comp_g.order_amount.sum())
    feat['avg_discount']=feat.customer_id.map(comp_g.discount_pct.mean())
    feat['avg_delivery']=feat.customer_id.map(comp_g.delivery_minutes.mean())
    cancelled=pre[pre.status.eq('cancelled')].groupby('customer_id').order_id.nunique()
    final_count=pre[pre.status.isin(['completed','cancelled'])].groupby('customer_id').order_id.nunique()
    feat['cancellation_rate']=(feat.customer_id.map(cancelled).fillna(0)/feat.customer_id.map(final_count).replace(0,np.nan)).fillna(0)
    fav=pre_comp.groupby('customer_id').category.agg(lambda s:s.value_counts().sort_values(ascending=False).index[0])
    feat['favourite_category']=feat.customer_id.map(fav)
    ccols=cust[['customer_id','city','age','gender','signup_date']].copy(); ccols['signup_date']=pd.to_datetime(ccols.signup_date)
    feat=feat.merge(ccols,on='customer_id',how='left')
    feat['tenure_days']=(CUTOFF-feat.signup_date).dt.days
    sept=comp[(comp.order_ts>=CUTOFF)&(comp.order_ts<pd.Timestamp('2025-10-01'))].groupby('customer_id').order_id.nunique()
    feat['lapsed']=(~feat.customer_id.isin(sept.index)).astype(int)
    # Remove any accidental nulls before split; numerical values use cohort medians, categories use explicit sentinel.
    numeric=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']
    categorical=['favourite_category','city','gender']
    for col in numeric: feat[col]=pd.to_numeric(feat[col],errors='coerce').fillna(feat[col].median())
    for col in categorical: feat[col]=feat[col].fillna('unknown').replace('', 'unknown').astype(str)
    assert not feat[numeric+categorical].isna().any().any()
    Xall=feat[numeric+categorical]; y=feat.lapsed
    Xtr,Xte,ytr,yte,ids_tr,ids_te=train_test_split(Xall,y,feat.customer_id,test_size=.2,stratify=y,random_state=42)
    num_pipe=Pipeline([('scale',StandardScaler())])
    cat_pipe=Pipeline([('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))])
    prep=ColumnTransformer([('num',num_pipe,numeric),('cat',cat_pipe,categorical)])
    lr=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42))])
    lr_grid=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=1)
    lr_grid.fit(Xtr,ytr)
    rf=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    rf_grid=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=1)
    rf_grid.fit(Xtr,ytr)
    def metrics(model):
        pred=model.predict(Xte)
        return {'confusion_matrix':confusion_matrix(yte,pred,labels=[0,1]).tolist(),'accuracy':accuracy_score(yte,pred),'precision_lapsed':precision_score(yte,pred,zero_division=0),'recall_lapsed':recall_score(yte,pred,zero_division=0),'f1_lapsed':f1_score(yte,pred,zero_division=0)}
    lrm=lr_grid.best_estimator_; rfm_model=rf_grid.best_estimator_
    chosen=rfm_model if metrics(rfm_model)['f1_lapsed']>=metrics(lrm)['f1_lapsed'] else lrm
    # Training-only random oversampling for logistic regression class-balance experiment.
    trdf=Xtr.copy(); trdf['_y']=ytr.to_numpy(); counts=trdf['_y'].value_counts(); maj=int(counts.idxmax()); minc=int(counts.idxmin())
    minority=trdf[trdf._y==minc]
    need=int(counts[maj]-counts[minc])
    extra=minority.sample(n=need,replace=True,random_state=42)
    balanced=pd.concat([trdf,extra]).sample(frac=1,random_state=42)
    lr_bal=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,C=lr_grid.best_params_['model__C'],random_state=42))])
    lr_bal.fit(balanced.drop(columns='_y'),balanced['_y'])
    # Save selected fitted pipeline and verify exact test prediction replay.
    joblib.dump(chosen,OUT/'lapse_model.joblib'); reloaded=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(chosen.predict(Xte),reloaded.predict(Xte))
    # Retention list: Premium is intentionally whole-window/post-hoc, per brief.
    premium_ids=set(rfm.loc[rfm.is_premium,'customer_id'])
    test_rows=feat.loc[feat.customer_id.isin(ids_te)].copy()
    test_rows=test_rows[test_rows.customer_id.isin(premium_ids)]
    if len(test_rows):
        probs=chosen.predict_proba(test_rows[numeric+categorical])[:,list(chosen.named_steps['model'].classes_).index(1)]
        retention=pd.DataFrame({'customer_id':test_rows.customer_id,'lapse_probability':probs,'actually_lapsed':test_rows.lapsed})
        retention=retention.sort_values('lapse_probability',ascending=False).head(10)
    else: retention=pd.DataFrame(columns=['customer_id','lapse_probability','actually_lapsed'])
    retention.to_csv(OUT/'retention_list.csv',index=False)
    # Explainable feature importances and logistic coefficient signs.
    rfbest=rfm_model.named_steps['model']; rfprep=rfm_model.named_steps['prep']; feature_names=rfprep.get_feature_names_out()
    importances=sorted(zip(feature_names,rfbest.feature_importances_),key=lambda z:z[1],reverse=True)[:5]
    lrbest=lrm.named_steps['model']; lrnames=lrm.named_steps['prep'].get_feature_names_out(); coefs=lrbest.coef_[0]
    coef_examples=sorted(zip(lrnames,coefs),key=lambda z:abs(z[1]),reverse=True)[:3]
    report={
      'rfm_summary':{'customers':len(rfm),'recency_median':float(rfm.recency.median()),'recency_min':int(rfm.recency.min()),'recency_max':int(rfm.recency.max()),'frequency_mean':float(rfm.frequency.mean()),'frequency_max':int(rfm.frequency.max()),'monetary_mean':float(rfm.monetary.mean()),'monetary_max':float(rfm.monetary.max())},
      'kmeans':krows,'chosen_k':4,'segment_profiles':profile.to_dict(orient='records'),'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
      'cohort_size':len(feat),'lapsed_count':int(y.sum()),'lapsed_pct':float(y.mean()*100),'train_rows':len(Xtr),'test_rows':len(Xte),
      'baseline':{'accuracy':accuracy_score(yte,np.zeros(len(yte),dtype=int)),'f1_lapsed':f1_score(yte,np.zeros(len(yte),dtype=int),zero_division=0)},
      'logistic':{'best_params':lr_grid.best_params_,'best_cv_f1':lr_grid.best_score_,'test':metrics(lrm)},
      'random_forest':{'best_params':rf_grid.best_params_,'best_cv_f1':rf_grid.best_score_,'test':metrics(rfm_model)},
      'selected_model':'random_forest' if chosen is rfm_model else 'logistic_regression','balanced_logistic_test':metrics(lr_bal),
      'scaler_frequency_train_mean':float(lrm.named_steps['prep'].named_transformers_['num'].named_steps['scale'].mean_[numeric.index('frequency')]),'full_cohort_frequency_mean':float(feat.frequency.mean()),
      'rf_top_importances':[{'feature':str(a),'importance':float(b)} for a,b in importances], 'lr_coefficient_examples':[{'feature':str(a),'coefficient':float(b),'sign':'positive' if b>0 else 'negative' if b<0 else 'zero'} for a,b in coef_examples],
      'reloaded_predictions_exact':True,'retention_rows':len(retention),'retention_true_lapsed':int(retention.actually_lapsed.sum()) if len(retention) else 0,
      'limitations':'The retention list is illustrative and not a leakage-free evaluation because Premium uses whole-window RFM, including the September label window.'}
    (OUT/'part_b_report.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf-8')
    print('PART B COMPLETE',json.dumps({'cohort':len(feat),'lapsed':int(y.sum()),'test_f1':metrics(chosen)['f1_lapsed'],'chosen_model':report['selected_model']}))
if __name__=='__main__': main()

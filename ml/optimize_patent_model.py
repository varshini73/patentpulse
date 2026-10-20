"""Reproducible PatentPulse 3.0 model audit and focused optimization runner.

Outputs are intentionally isolated under outputs/model_optimization; Review 2
artifacts and the current production model are never overwritten.
"""
from __future__ import annotations
import json, platform, shutil, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import estimator_html_repr
try:
    from ml.run_review2 import ROOT, RAW, SEED, warehouse_data
except ModuleNotFoundError:  # supports `python ml/optimize_patent_model.py`
    from run_review2 import ROOT, RAW, SEED, warehouse_data

OUT=ROOT/'outputs'/'model_optimization'; MODELS=ROOT/'models'; OUT.mkdir(parents=True,exist_ok=True)
NUM=['pages','claims','inventors','npl_count','ipc_count','pub_year']; CAT=['country_code','kind_code','ipc_subclass']
FEATURES=NUM+CAT; CURRENT=MODELS/'classification_benchmark_model.joblib'

def data():
    df=warehouse_data().dropna(subset=['is_highly_cited']).copy()
    raw=pd.read_excel(RAW,usecols=['PN','TITLE','ABSTRACT','APP YEAR','CITED TIME IN 3 YEARS','FAMILY PATENTS NUMBER','NUMBER OF CITATION'])
    raw=raw.drop_duplicates('PN').rename(columns={'PN':'pn','TITLE':'title','ABSTRACT':'abstract'})
    df=df.merge(raw[['pn','title','abstract','APP YEAR','CITED TIME IN 3 YEARS','FAMILY PATENTS NUMBER','NUMBER OF CITATION']],on='pn',how='left',validate='one_to_one')
    df['text']=df.title.fillna('')+' '+df.abstract.fillna('')
    return df
def prep_hgb():
    return ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=True),NUM),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan))]),CAT)])
def prep_onehot():
    return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('log',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=5))]),CAT)])
def text_prep():
    return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('log',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=5))]),CAT),('text',TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=5000,sublinear_tf=True),'text')])
def metrics(y,pred,score):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),macro_f1=f1_score(y,pred,average='macro'),roc_auc=roc_auc_score(y,score),pr_auc=average_precision_score(y,score),mcc=matthews_corrcoef(y,pred),brier=brier_score_loss(y,score),log_loss=log_loss(y,score),tn=tn,fp=fp,fn=fn,tp=tp)
def main():
    df=data(); y=df.is_highly_cited.astype(int); xtrain,xtest,ytrain,ytest,idtrain,idtest=train_test_split(df,y,df.pn,test_size=.2,stratify=y,random_state=SEED)
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    # Track B only: all input variables can be PDF-extracted or explicitly confirmed.
    candidates={
      'majority_baseline':(Pipeline([('prep',prep_onehot()),('model',DummyClassifier(strategy='most_frequent'))]),FEATURES,'baseline'),
      'deployable_hgb_v2':(Pipeline([('prep',prep_hgb()),('model',HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=7,l2_regularization=1.0,random_state=SEED))]),FEATURES,'deployable structured'),
      'deployable_extra_trees':(Pipeline([('prep',prep_onehot()),('model',ExtraTreesClassifier(n_estimators=350,min_samples_leaf=2,class_weight='balanced',max_features=.7,random_state=SEED,n_jobs=1))]),FEATURES,'deployable structured'),
      'deployable_text_logistic':(Pipeline([('prep',text_prep()),('model',LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced',random_state=SEED))]),FEATURES+['text'],'deployable title+abstract+structured'),
    }
    rows=[]; fitted={}
    for name,(model,cols,track) in candidates.items():
        start=time.perf_counter(); oof=cross_val_predict(model,xtrain[cols],ytrain,cv=cv,method='predict_proba',n_jobs=1)[:,1]; cvpred=(oof>=.5).astype(int); model.fit(xtrain[cols],ytrain); score=model.predict_proba(xtest[cols])[:,1]; pred=(score>=.5).astype(int); row={'model':name,'track':'B - deployable PDF/document','feature_set':', '.join(cols),'validation':'fixed 80/20 stratified split; 3-fold train-only CV','cv_accuracy':accuracy_score(ytrain,cvpred),'cv_pr_auc':average_precision_score(ytrain,oof),'test_records':len(xtest),'positive_test':int(ytest.sum()),'training_seconds':time.perf_counter()-start,'threshold':.5,'leakage_status':'passed: excludes target/cited_3yr/forward citations/family size/backward citations','pdf_deployable':True,**metrics(ytest,pred,score)}; rows.append(row); fitted[name]=(model,cols,score,pred); print(name,round(row['cv_accuracy'],4),round(row['accuracy'],4),flush=True)
    # Selection relies only on CV: accuracy primary, PR-AUC as tie-breaker.
    selected=max(rows,key=lambda r:(r['cv_accuracy'],r['cv_pr_auc']))['model']; model,cols,score,pred=fitted[selected]
    for r in rows:r['selected_by_cv']=r['model']==selected
    if CURRENT.exists(): shutil.copy2(CURRENT,MODELS/'classification_benchmark_model_pre_optimization.joblib'); meta=MODELS/'classification_benchmark_metadata.json'; shutil.copy2(meta,MODELS/'classification_benchmark_metadata_pre_optimization.json') if meta.exists() else None
    version='document_model_v2'; artifact={'model':model,'features':cols,'numeric_features':NUM,'categorical_features':CAT,'selected_model':selected,'model_version':version,'split':{'test_size':.2,'random_state':SEED,'stratified':True},'threshold':.5,'calibration':'raw estimator probability; Brier and log loss reported, no calibration selected without CV benefit','timing_note':'family_size and backward_citations excluded because prediction-time availability is unresolved','track':'B - deployable document model'}
    joblib.dump(artifact,MODELS/f'{version}.joblib'); (MODELS/f'{version}.json').write_text(json.dumps({k:v for k,v in artifact.items() if k!='model'},indent=2),encoding='utf8')
    pd.DataFrame(rows).sort_values('model').to_csv(OUT/'model_leaderboard.csv',index=False); pd.DataFrame({'pn':idtest.values,'actual_label':ytest.values,'predicted_probability':score,'predicted_label':pred,'selected_model':selected}).to_csv(OUT/'selected_predictions.csv',index=False)
    audit=pd.DataFrame([{'feature':c,'source':'warehouse/raw workbook','availability':'A - document/external confirmation','leakage_risk':'low','production_permitted':True} for c in FEATURES]+[{'feature':'title/abstract text','source':'raw workbook / PDF extraction','availability':'A - document','leakage_risk':'low','production_permitted':True},{'feature':'backward_citations','source':'workbook','availability':'D - timing uncertain','leakage_risk':'possible later updates','production_permitted':False},{'feature':'family_size','source':'workbook','availability':'D - timing uncertain','leakage_risk':'possible later updates','production_permitted':False},{'feature':'cited_3yr / is_highly_cited','source':'workbook / warehouse','availability':'C - outcome only','leakage_risk':'direct target','production_permitted':False}]); audit.to_csv(OUT/'feature_registry.csv',index=False)
    report={'rows_total':13732,'labelled_rows':len(df),'positive':int(y.sum()),'raw_columns':17,'duplicate_pn':int(df.pn.duplicated().sum()),'target_definition':'pub_year 2000-2016 and cited_3yr >= 10','missingness':df[FEATURES].isna().mean().round(4).to_dict(),'cardinality':df[CAT].nunique().to_dict(),'selected_model':selected,'selection_rule':'highest 3-fold training-CV accuracy, PR-AUC tie break','versions':{'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__}}; (OUT/'data_quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'selected={selected}; test_accuracy={next(r for r in rows if r["model"]==selected)["accuracy"]:.4f}')
if __name__=='__main__':main()

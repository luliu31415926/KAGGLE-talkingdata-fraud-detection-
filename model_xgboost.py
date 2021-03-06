import pandas as pd 
import numpy as np
import time 
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import  train_test_split
import gc 
import _pickle as cPickle
import features


def go(data_dict,feats_to_use, params):
  
    X_train=data_dict['X_train'][feats_to_use].copy()
    y_train=data_dict['y_train'].copy()
    X_test=data_dict['X_test'][feats_to_use].copy()
    X_val=data_dict['X_val'][feats_to_use].copy()
    y_val=data_dict['y_val'].copy()

    
    xgb=XGBClassifier(**params)
    print (xgb)

    print ('start xgboost training')
    start=time.time()
    eval_set=[(X_val,y_val)]
   # xgb.fit(X_train,y_train, eval_set=eval_set,eval_metric='auc',early_stopping_rounds=20)
    xgb.fit(X_train,y_train)
    elapsed=time.time()-start
    print (elapsed)
    get_feature_importances(feats_to_use,xgb)
    submission=pd.read_csv('../talkingdata_data/sample_submission.csv')
    submission['is_attributed']=xgb.predict_proba(X_test)[:,1]
    return submission, xgb
def get_feature_importances(feats_to_use, xgb):
    feature_importances=list(zip(feats_to_use,xgb.feature_importances_))
    feature_importances.sort(key=lambda x:x[1], reverse=True)
    print (feature_importances)
    with open('../talkingdata_data/xgb_feature_importances.pkl','wb')as handle:
        cPickle.dump(feature_importances,handle) 
def resplit_data(data_dict):
    X_train=data_dict['X_train']
    y_train=data_dict['y_train']
    X_val=data_dict['X_val']
    y_val=data_dict['y_val']
    full=pd.concat([pd.concat([X_train,X_val]),pd.concat([y_train,y_val])],axis=1)
    full.sort_values(['click_time_day','click_time_hour'],inplace=True)
    test_size=data_dict['X_test'].shape[0]
    train=full.iloc[:-test_size]
    val=full.iloc[-test_size:]

    data_dict['X_train']=train.drop('is_attributed',axis=1)
    data_dict['y_train']=train.is_attributed
    data_dict['X_val']=val.drop('is_attributed',axis=1)
    data_dict['y_val']=val.is_attributed
    for key in data_dict.keys():
        print (key,data_dict[key].shape)
    print ('dumping dataframe to disk')
    cPickle.dump(data_dict,open("../talkingdata_data/data_dict_resplit_time_val.pkl","wb"),protocol=-1)
    gc.collect();

    return data_dict 
def combine_data(data_dict):
    X_train=data_dict['X_train']
    y_train=data_dict['y_train']
    X_val=data_dict['X_val']
    y_val=data_dict['y_val']
    data_dict['X_train']=pd.concat([X_train,X_val])
    data_dict['y_train']=pd.concat([y_train,y_val])


    for key in data_dict.keys():
        print (key,data_dict[key].shape)
   
    gc.collect();
    return data_dict
        
if __name__ == "__main__":
    '''
    with open('../talkingdata_data/test_data_dict.pkl','rb') as handle:
        data_dict=cPickle.load(handle)
    '''
    with open('../talkingdata_data/data_dict_resplit.pkl','rb') as handle:
        data_dict=cPickle.load(handle)
    data_dict=combine_data(data_dict)

    
    feats_to_use=features.xgb_features_selected
    # to deal with imalanced data set, calculate scale_pos_weight parameter
    scale_pos_weight = 100 - ( data_dict['y_train'].sum() / data_dict['y_train'].size  * 100 )
    print ('scale_pos_weight: ',scale_pos_weight)
    params={"n_jobs":-1,
    "silent":True,
    "learning_rate":0.1,
    "n_estimators":100,
    "max_depth":3,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "scale_pos_weight": scale_pos_weight,
    }
    submission,xgb=go(data_dict, feats_to_use,params)
    submission.to_csv('submission_xgb_%s.csv'%time.strftime("%Y%m%d-%H%M%S"),index=False)
    with open('xgb.pkl','wb') as handle:
        cPickle.dump(xgb,handle,protocol=-1)
    
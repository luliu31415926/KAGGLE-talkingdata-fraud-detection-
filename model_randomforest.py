import pandas as pd 
import numpy as np
import time 
from sklearn.ensemble import RandomForestClassifier
import _pickle as cPickle
import features
import gc
def go(data_dict,feats_to_use,params):

    X_train=data_dict['X_train'][feats_to_use].copy()
    y_train=data_dict['y_train'].copy()
    X_test=data_dict['X_test'][feats_to_use].copy()
    #X_val=data_dict['X_val'][feats_to_use].copy()
    #y_val=data_dict['y_val'].copy()

    start_time=time.time()
    rfr=RandomForestClassifier(**params)
    print (rfr) 
    rfr.fit(X_train,y_train)
    elapsed=time.time()-start_time
    print ('elapsed: ',elapsed)
    
    #generate submission

    get_feature_importances(feats_to_use,rfr)
    submission=pd.read_csv('../talkingdata_data/sample_submission.csv')
    submission['is_attributed']=rfr.predict_proba(X_test)[:,1]

    return submission, rfr

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

def get_feature_importances(feats_to_use, rfr):
    feature_importances=list(zip(feats_to_use,rfr.feature_importances_))
    feature_importances.sort(key=lambda x:x[1], reverse=True)
    print (feature_importances)
    with open('../talkingdata_data/rfr_feature_importances.pkl','wb')as handle:
        cPickle.dump(feature_importances,handle) 
if __name__ == "__main__":
    '''
    with open('../talkingdata_data/test_data_dict.pkl','rb') as handle:
        data_dict=cPickle.load(handle)
    '''
    with open('../talkingdata_data/data_dict.pkl','rb') as handle:
        data_dict=cPickle.load(handle)
    
    data_dict=combine_data(data_dict)
    feats_to_use=features.rfr_features
    
    params={"n_jobs":-1,
    "n_estimators":100,
    "max_depth":3,
    "oob_score":True,
    "class_weight":'balanced',
    "verbose":3
    }
    submission,rfr=go(data_dict, feats_to_use,params)
    submission.to_csv('submission_rfr%s.csv'%time.strftime("%Y%m%d-%H%M%S"),index=False)
    with open('rfr.pkl','wb') as handle:
        cPickle.dump(rfr,handle,protocol=-1)
    
import pandas as pd 
import numpy as np
import time 
from sklearn.ensemble import RandomForestClassifier
import _pickle as cPickle
import features
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
    data_dict['X_test']['is_attributed']=rfr.predict_proba(X_test)[:,1]
    submission=data_dict['X_test'][['click_id','is_attributed']]

    return submission, rfr

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
    
    feats_to_use=features.rfr_features
    
    params={"n_jobs":-1,
    "n_estimators":100,
    "max_depth":3,
    "oob_score":True,
    "class_weight":'balanced'
    }
    submission,rfr=go(data_dict, feats_to_use,params)
    submission.to_csv('submission_rfr.csv'%time.strftime("%Y%m%d-%H%M%S"),index=False)
    with open('rfr.pkl','wb') as handle:
        cPickle.dump(rfr,handle,protocol=-1)
    
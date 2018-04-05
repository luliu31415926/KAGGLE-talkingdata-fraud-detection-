import pandas as pd 
import numpy as np
import time 
from sklearn.svm import SVC
from sklearn.model_selection import  train_test_split
import gc 
import _pickle as cPickle
import features


def go(data_dict,feats_to_use, params):
  
    X_train=data_dict['X_train'][feats_to_use].copy()
    y_train=data_dict['y_train'].copy()
    X_test=data_dict['X_test'][feats_to_use].copy()
   

    
    clf=SVC(**params)
    print (clf)

    print ('start SVM training')
    start=time.time()
    
    clf.fit(X_train,y_train)
    elapsed=time.time()-start
    print (elapsed)
    submission=pd.read_csv('../talkingdata_data/sample_submission.csv')
    submission.is_attributed=clf.predict_proba(X_test)[:,1]
    return submission, xgb

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

    
    feats_to_use=features.lr_features
    
    params={'verbose':True,
            'probability':True,
            'class_weight':'balanced',
    }
    submission,clf=go(data_dict, feats_to_use,params)
    submission.to_csv('submission_svm_%s.csv'%time.strftime("%Y%m%d-%H%M%S"),index=False)
   
    
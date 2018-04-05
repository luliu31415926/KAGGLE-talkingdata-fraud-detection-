
import numpy as np
import pandas as pd
import _pickle as cPickle
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Merge
from keras.layers.core import Dense, Dropout,Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import features
from keras import backend as K
import time
# 'ip', 'app', 'device', 'os', 'channel', 'click_time_day', 'click_time_hour'
'''
ip 68740
app 332
device 940
os 292
channel 170
'''
def _build_keras_model(ip_max,app_max,device_max,os_max,channel_max):
   
    print ('creating embedding nn model')
    model_lst=[]

    model_shop = Sequential()
    model_shop.add(Embedding(60, 10, input_length=1)) 
    model_shop.add(Reshape((10,)))
    model_lst.append(model_shop)

    model_month = Sequential()
    model_month.add(Embedding(12, 4, input_length=1)) 
    model_month.add(Reshape((4,)))
    model_lst.append(model_month)

    model_category = Sequential()
    model_category.add(Embedding(84, 12, input_length=1)) 
    model_category.add(Reshape((12,)))
    model_lst.append(model_category)
    
    model_category_name = Sequential()
    model_category_name.add(Dense(5, input_dim=5))
    model_lst.append(model_category_name)

    model_item_name = Sequential()
    model_item_name.add(Dense(5, input_dim=5))
    model_lst.append(model_item_name)
    
    model_month_mean_enc = Sequential()
    model_month_mean_enc.add(Dense(4, input_dim=4))
    model_lst.append(model_month_mean_enc)

    model_year_mean_enc = Sequential()
    model_year_mean_enc.add(Dense(4, input_dim=4))
    model_lst.append(model_year_mean_enc)

    model_item_mean_enc = Sequential()
    model_item_mean_enc.add(Dense(4, input_dim=4))
    model_lst.append(model_item_mean_enc)

    model_shop_mean_enc = Sequential()
    model_shop_mean_enc.add(Dense(4, input_dim=4))
    model_lst.append(model_shop_mean_enc)

    model_category_mean_enc = Sequential()
    model_category_mean_enc.add(Dense(4, input_dim=4))
    model_lst.append(model_category_mean_enc)
    
    model_shop_item_lag = Sequential()
    model_shop_item_lag.add(Dense(16, input_dim=16))
    model_lst.append(model_shop_item_lag)
    
    model_category_lag = Sequential()
    model_category_lag.add(Dense(17, input_dim=17))
    model_lst.append(model_category_lag)
    
    model_item_lag = Sequential()
    model_item_lag.add(Dense(14, input_dim=14))
    model_lst.append(model_item_lag)

    model_shop_lag = Sequential()
    model_shop_lag.add(Dense(15, input_dim=15))
    model_lst.append(model_shop_lag)


    model = Sequential()
    model.add(Merge(model_lst, mode='concat'))
    model.add(Dropout(0.02))
    model.add(Dense(1000, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(500, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model



def preprocess_nn_features_for_embedding(data_dict, nn_feats):

    nn_data_dict=dict()
    for key in data_dict.keys():
        nn_data_dict[key]=data_dict[key][nn_feats] if 'X' in key else data_dict[key]
        if 'X' in key:
            nn_data_dict[key].month-=1 
  
    print ('standardizing features')
    feats_to_scale=[f for f in nn_feats if f not in ['shop_id','month','item_category_id']]
    scaler=StandardScaler(copy=True,with_mean=True,with_std=True)
    nn_data_dict['X_train'][feats_to_scale]=scaler.fit_transform(nn_data_dict['X_train'][feats_to_scale])
    nn_data_dict['X_val'][feats_to_scale]=scaler.transform(nn_data_dict['X_val'][feats_to_scale])
    nn_data_dict['X_test'][feats_to_scale]=scaler.transform(nn_data_dict['X_test'][feats_to_scale])

    #divide the features into individual dataframes 
    print ('divide the features into individual dataframes')
    for key in ['X_train','X_val','X_test']:
        feat_len=  [1,1,1,5,5,4,4,4,4,4,16,17,14,15]
        X=np.array(nn_data_dict[key])
        X_lst=[]
        cum=0
        for n in feat_len:
            X_lst.append(X[:,range(cum,cum+n)])
            cum+=n 
        nn_data_dict[key]=X_lst
    for key in ['y_train','y_val']:
        nn_data_dict[key]=nn_data_dict[key].as_matrix()
    with open('../1c_data/embedding_nn_data_dict.pkl','wb')as handle:
        cPickle.dump(nn_data_dict,handle,protocol=-1)
    with open('../1c_data/embedding_nn_X_test.pkl','wb') as handle:
        cPickle.dump(nn_data_dict['X_test'],handle,protocol=-1)
                  
    return nn_data_dict

def run(data_dict):
    model=_build_keras_model()
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    print (model.summary())
    
    early_stopping=EarlyStopping(monitor='val_loss', patience=5)
    check_point = ModelCheckpoint('../1c_data/embedding_weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    fit_params={
        'epochs':100,
        'batch_size':2048,
        'validation_data':(data_dict['X_val'],data_dict['y_val']),
        'callbacks':[early_stopping, check_point] ,
        'shuffle':True
    }

    model.fit(data_dict['X_train'],data_dict['y_train'],**fit_params )
    

def make_prediction(test_file_path):
    with open(test_file_path,'rb') as handle:
        X_test=cPickle.load(handle)
    model=_build_keras_model()
    model.load_weights('../1c_data/embedding_weights.best.hdf5')
    model.compile(loss='mean_squared_error', optimizer='adam')
    pred= np.squeeze(model.predict(X_test))
    print ('shape of pred:',pred.shape)
    return pred
    
    
if __name__ == "__main__":
    
    file_path='../1c_data/data_dict.pkl'
    print ('getting data from file',file_path)
    with open(file_path,'rb') as handle:
        data_dict = cPickle.load(handle)
    
    
    nn_feats=features.embedding_feats
    print ('using %d features'%len(nn_feats))
    nn_data_dict=preprocess_nn_features_for_embedding(data_dict, nn_feats)
    
    '''
    file_path='../1c_data/embedding_nn_data_dict.pkl'
    with open(file_path,'rb') as handle:
        nn_data_dict = cPickle.load(handle)
    '''
    run(nn_data_dict)
    
    
    test_file_path='../1c_data/embedding_nn_X_test.pkl'
    pred=make_prediction(test_file_path)
    data_dict['X_test']['item_cnt_month']=pred
    test=pd.read_csv('test.csv')
    submission=pd.merge(test,data_dict['X_test'], 
    on=['shop_id','item_id'],how='left')[['ID','item_cnt_month']]
    submission.to_csv('submission_nn_embedding_%s.csv'%time.strftime("%Y%m%d-%H%M%S"), index=False)
    print ('saved submission to local')

   
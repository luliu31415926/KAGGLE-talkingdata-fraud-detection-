import pandas as pd 
import numpy as np
import gc 
import pytz
import itertools
import _pickle as cPickle 

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train=pd.read_csv('../talkingdata_data/train.csv',dtype=dtypes,skiprows=104903890, nrows=80000000, 
               names =["ip", "app", "device", "os", "channel", "click_time", 
                            "attributed_time", "is_attributed"])
test=pd.read_csv('../talkingdata_data/test.csv',dtype=dtypes)
'''
train=pd.read_csv('../talkingdata_data/train.csv',dtype=dtypes,nrows=10000)
test=pd.read_csv('../talkingdata_data/test.csv',dtype=dtypes,nrows=1000)
'''
test_size=test.shape[0]
train.drop('attributed_time',axis=1,inplace=True)

val= train.iloc[-test_size:]
train=train.iloc[:-test_size]

def add_time_features(df):
    cst = pytz.timezone('Asia/Shanghai')
    df['click_time_datetime']=pd.to_datetime(df.click_time).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    df['click_time_day']=df.click_time_datetime.dt.day.astype('uint8')
    df['click_time_hour']=df.click_time_datetime.dt.hour.astype('uint8')
    df.drop(['click_time_datetime','click_time'],axis=1,inplace=True)
    return df
print ('adding time features')
train=add_time_features(train)
val=add_time_features(val)
test=add_time_features(test)

print ('start calculating stats')
'''
for each feature set , calculate 6 features 
1. total conversion rate 
2. max(daily click rate) 
4. max(hourly click rate)
5. ave(hourly click rate)
6. std(hourly click rate) 
'''
def get_feats_df(feats):
    name_prefix='_'.join(feats)
    conversion_df=train.groupby(feats).is_attributed.agg([(name_prefix+'_conversion_rate',lambda x:100*x.sum()/x.count())]).astype('float32')
    daily_df=train.groupby(feats+['click_time_day']).is_attributed.count().reset_index()
    #daily_df=daily_df.groupby(feats).is_attributed.agg([(name_prefix+'_daily_max','max'),(name_prefix+'_daily_mean','mean')]).astype('float32')
    daily_df=daily_df.groupby(feats).is_attributed.agg([(name_prefix+'_daily_max','max')]).astype('float32')
    hourly_df=train.groupby(feats+['click_time_day','click_time_hour']).is_attributed.count().reset_index()
    #hourly_df=hourly_df.groupby(feats).is_attributed.agg([(name_prefix+'_hourly_max','max'),(name_prefix+'_hourly_mean','mean'),(name_prefix+'_hourly_std','std')]).astype('float32')
    hourly_df=hourly_df.groupby(feats).is_attributed.agg([(name_prefix+'_hourly_std','std')]).astype('float32')
    feats_df=pd.concat([conversion_df,daily_df,hourly_df],axis=1).fillna(0)
    return feats_df  

columns=['app', 'channel', 'device', "os", "ip"]
encoding_feats=sum([list(itertools.combinations(columns,i)) for i in range(1,3)],[])
for feats in encoding_feats:
    feats=list(feats)
    feats_df=get_feats_df(feats)
    print ('calculating for :', feats)
    #print (feats_df.index.names)
    train=train.merge(feats_df,how='left',left_on=feats,right_index=True)
    val=val.merge(feats_df,how='left',left_on=feats,right_index=True)
    val.fillna(val.median(),inplace=True)
    test=test.merge(feats_df,how='left',left_on=feats,right_index=True)
    test.fillna(test.median(),inplace=True)
    gc.collect();
print ('train shape',train.shape)
print ('val shape', val.shape)
print ('test shape', test.shape)
print (train.columns)
data_dict=dict()
test_data_dict=dict()
data_dict['X_test']=test
data_dict['X_val']=val.drop('is_attributed',axis=1)
data_dict['y_val']=val.is_attributed
data_dict['X_train']=train.drop('is_attributed',axis=1)
data_dict['y_train']=train.is_attributed
for key in data_dict.keys():
	test_data_dict[key]=data_dict[key].iloc[:1000]

del train,val,test 
gc.collect();

print ('dumping dataframe to disk')
cPickle.dump(test_data_dict,open("../talkingdata_data/test_data_dict.pkl","wb"),protocol=-1)
cPickle.dump(data_dict,open("../talkingdata_data/data_dict.pkl","wb"),protocol=-1)
gc.collect();
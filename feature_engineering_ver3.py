# hourly count features and yesterday's conversion rate 
import pandas as pd 
import numpy as np
import gc 
import pytz
import itertools
import _pickle as cPickle 
from sklearn.model_selection import  train_test_split

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train=pd.read_csv('../talkingdata_data/train.csv',dtype=dtypes)
train=train.sample(n=60000000)
gc.collect();
test=pd.read_csv('../talkingdata_data/test.csv',dtype=dtypes)
'''
train=pd.read_csv('../talkingdata_data/train.csv',dtype=dtypes,nrows=1000)
test=pd.read_csv('../talkingdata_data/test.csv',dtype=dtypes,nrows=100)
'''
test_size=test.shape[0]
train.drop('attributed_time',axis=1,inplace=True)

def add_time_features(df):
    cst = pytz.timezone('Asia/Shanghai')
    df['click_time_datetime']=pd.to_datetime(df.click_time).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    df['click_time_day']=df.click_time_datetime.dt.day.astype('uint8')
    df['click_time_hour']=df.click_time_datetime.dt.hour.astype('uint8')
    df.drop(['click_time_datetime','click_time'],axis=1,inplace=True)
    return df
print ('adding time features')
train=add_time_features(train)
test=add_time_features(test)
all_data=pd.concat([train,test.drop('click_id',axis=1)])
print ('all data',all_data.shape)
gc.collect();
print ('start calculating stats')

def get_feats_df(feats,all_data):
    name_prefix='_'.join(feats)
    # hourly click rate
    feats_df=all_data.groupby(feats+['click_time_day','click_time_hour']).is_attributed.agg([(name_prefix+'_hourly_count','count')]).astype('uint32').reset_index()
    conversion_df=all_data.groupby(feats+['click_time_day']).is_attributed.agg([(name_prefix+'_yes_conversion_rate',lambda x:x.sum()/x.count()*100)]).astype('float32').reset_index()
    conversion_df['click_time_day']=conversion_df.click_time_day+1
    feats_df=feats_df.merge(conversion_df,how='left',on=feats+['click_time_day'])
    feats_df[name_prefix+'_is_new']=feats_df[name_prefix+'_yes_conversion_rate'].isnull().astype('uint8')
    feats_df.fillna(0,inplace=True)
    return feats_df  

columns=['app', 'channel', 'device', "os", "ip"]
encoding_feats=sum([list(itertools.combinations(columns,i)) for i in range(1,3)],[])
for feats in encoding_feats:
    if feats==('ip',): continue 
    feats=list(feats)
    feats_df=get_feats_df(feats,all_data)
    print ('calculating for :', feats)
    all_data=all_data.merge(feats_df,how='left',on=feats+['click_time_day','click_time_hour'])
    all_data.fillna(all_data.median(),inplace=True)
    gc.collect();

print (all_data.columns)
test_df=all_data.iloc[-test_size:]
train_df=all_data.iloc[:-test_size]
train_df=train_df[(all_data.click_time_day>7)&(all_data.click_time_day<10)]
data_dict=dict()
test_data_dict=dict()

data_dict['X_test']=test_df.drop('is_attributed',axis=1)
data_dict['X_test']['click_id']=test.click_id
data_dict['X_train'],data_dict['X_val'],data_dict['y_train'],data_dict['y_val']=train_test_split(train_df.drop('is_attributed',axis=1), train_df.is_attributed, test_size=0.05, random_state=42, shuffle=True)
for key in data_dict.keys():
    test_data_dict[key]=data_dict[key].iloc[:1000]
    print (key,data_dict[key].shape)

gc.collect();

print ('dumping dataframe to disk')
cPickle.dump(test_data_dict,open("../talkingdata_data/test_data_dict.pkl","wb"),protocol=-1)
cPickle.dump(data_dict,open("../talkingdata_data/data_dict.pkl","wb"),protocol=-1)
gc.collect();
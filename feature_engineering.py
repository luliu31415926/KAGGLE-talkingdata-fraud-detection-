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

train=pd.read_csv('../talkingdata_data/train.csv',dtype=dtypes,skiprows=124903890, nrows=100000,names=['ip','app','device','os','channel','click_time','attributed_time','is_attributed'])
test=pd.read_csv('../talkingdata_data/test.csv',dtype=dtypes,nrows=1000)
train.drop('attributed_time',axis=1,inplace=True)
all_df=pd.concat([train,test.drop('click_id',axis=1)])
all_df.fillna(0,inplace=True)
all_df['is_attributed']=all_df.is_attributed.astype('uint8')

def add_time_features(df):
    cst = pytz.timezone('Asia/Shanghai')
    df['click_time_datetime']=pd.to_datetime(df.click_time).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    df['click_time_day']=df.click_time_datetime.dt.day.astype('uint8')
    df['click_time_hour']=df.click_time_datetime.dt.hour.astype('uint8')
    return df
print ('adding time features')
all_df=add_time_features(all_df)


def add_expanding_freq_mean(df,features):
    print ('adding frequency encoding for ',' '.join(features))
    cumcnt=df.groupby(features).cumcount()
    freq_col='_'.join(features)+'_freq_enc'
    freq_corr=np.corrcoef(df['is_attributed'].values, 
                       cumcnt.values)[0][1]
    print ('frequency encoding correlation with target is :', freq_corr)
    if abs(freq_corr)>0.005:
        df[freq_col]=cumcnt
    else: print ('correlation too low')
        
    print ('adding mean encoding for ',' '.join(features))
    cumsum=df.groupby(features).is_attributed.cumsum()-df.is_attributed
    col_name='_'.join(features)+'_mean_enc'
    df[col_name]=cumsum/(cumcnt+1)
    mean_corr=np.corrcoef(df['is_attributed'].values, 
                       df[col_name].values)[0][1]
    print ('mean encoding correlation with target is :', mean_corr)

    if abs(mean_corr)<0.005: 
        print ('correlation with target too low, dropped feature')
        df.drop(col_name, inplace=True)  
    return df,freq_corr,mean_corr 
correlations_dict=dict()
columns=['app', 'channel', 'device', 'ip','os', 'click_time_day', 'click_time_hour']
encoding_feats=sum([list(itertools.combinations(columns,i)) for i in range(1,4)],[])
print ('start calculating freqency encoding and mean encodings')
for features in encoding_feats:
    all_df,freq_corr,mean_corr=add_expanding_freq_mean(all_df,list(features))
    correlations_dict[features]={'freq_corr':freq_corr,'mean_corr':mean_corr}
cPickle.dump(correlations_dict,open("../talkingdata_data/correlations_dict.pkl","wb"))
cPickle.dump(all_df,open("../talkingdata_data/all_df.pkl","wb"))

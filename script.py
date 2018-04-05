import pandas as pd
from sklearn.model_selection import  train_test_split
import _pickle as cPickle

'''
with open('../talkingdata_data/test_data_dict.pkl','rb') as handle:
    data_dict=cPickle.load(handle)
'''
with open('../talkingdata_data/data_dict.pkl','rb') as handle:
    data_dict=cPickle.load(handle)


# resplit train and validation 
X_train=data_dict['X_train']
y_train=data_dict['y_train']
X_val=data_dict['X_val']
y_val=data_dict['y_val']
X=pd.concat([X_train,X_val])
y=pd.concat([y_train,y_val])


data_dict['X_train'],data_dict['X_val'],data_dict['y_train'],data_dict['y_val']=train_test_split(X, y, test_size=data_dict['X_test'].shape[0], random_state=42, shuffle=True)

for key in data_dict.keys():
    print (key,data_dict[key].shape)

print ('dumping dataframe to disk')
cPickle.dump(data_dict,open("../talkingdata_data/data_dict.pkl","wb"),protocol=-1)
gc.collect();

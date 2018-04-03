import _pickle as cPickle
import pandas as pd
import numpy as np
def go(test=False):
	data_dict=dict()

	if test:
		with open('../talkingdata_data/test_all_df.pkl','rb') as handle:
		    all_df=cPickle.load(handle)
		    test_size=1000
	else:
		with open('../talkingdata_data/all_df.pkl','rb') as handle:
		    all_df=cPickle.load(handle)
		    test_size=18790469
    data_dict['X_test']=all_df.iloc[-test_size:].drop('is_attributed',axis=1)
    data_dict['X_val']=all_df.iloc[-test_size*2:-test_size].drop('is_attributed',axis=1)
    data_dict['y_val']=all_df.iloc[-test_size*2:-test_size].is_attributed
    data_dict['X_train']=all_df.iloc[:-test_size*2].drop('is_attributed',axis=1)
    data_dict['y_train']=ll_df.iloc[:-test_size*2].is_attributed
	
	return data_dict
# -*- coding: utf-8 -*-
"""
https://leaderboard.corp.amazon.com/tasks/378/leaderboard
https://w2.weather.gov/climate/xmacis.php?wfo=sew

"""
###############################################################################
## Imports
###############################################################################
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import urllib
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
###############################################################################
## settings, functions and globals
###############################################################################
pd.set_option('display.max_columns', None)

PATH          = 'C:/Work/kaggle/wfs/'

target = 'nhenoshow_flag'


def fix_shift_days_week(df):
    df.loc[df.shift_days_of_week.isnull(),'shift_days_of_week'] = ''
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace(' ', '')
    df['shift_days_of_week'] = df.shift_days_of_week.str.lower()
    
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('thursday' , 'thu')
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('friday'   , 'fri')
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('saturday' , 'sat')
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('sunday'   , 'sun')
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('monday'   , 'mon')
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('tuesday'  , 'tue')
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('wednesday', 'wed')
    
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('tues', 'tue')
    
    ## remove slashes
    df.loc[df.shift_days_of_week=='m/tu/th/f', 'shift_days_of_week'] = 'mon,tue,thu,fri'
    df.loc[df.shift_days_of_week=='w/th/f/su', 'shift_days_of_week'] = 'wed,thu,fri,sun'
    df.loc[df.shift_days_of_week=='m/t/th/f', 'shift_days_of_week'] = 'mon,tue,thu,fri'
    df.loc[df.shift_days_of_week=='m/tu/th/f/sa', 'shift_days_of_week'] = 'mon,tue,thu,fri,sat'
    df.loc[df.shift_days_of_week=='mon-wed/f/sa', 'shift_days_of_week'] = 'mon,tue,wed,fri,sat'
    df['shift_days_of_week'] = df.shift_days_of_week.str.replace('/', ',')
    
    ## remove dashes
    booll = df.shift_days_of_week.str.contains('mon-wed')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('mon-wed','mon,tue,wed')
    booll = df.shift_days_of_week.str.contains('mon-fri')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('mon-fri','mon,tue,wed,thu,fri')
    booll = df.shift_days_of_week.str.contains('mon-thu')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('mon-thu','mon,tue,wed,thu')
    booll = df.shift_days_of_week.str.contains('mon-wed')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('mon-wed','mon,tue,wed')
    booll = df.shift_days_of_week.str.contains('tue-sat')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('tue-sat','tue,wed,thu,fri,sat')
    booll = df.shift_days_of_week.str.contains('tue-fri')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('tue-fri','tue,wed,thu,fri')
    booll = df.shift_days_of_week.str.contains('tue-thu')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('tue-thu','tue,wed,thu')
    booll = df.shift_days_of_week.str.contains('tue-wed')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('tue-wed','tue,wed')
    booll = df.shift_days_of_week.str.contains('wed-sun')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('wed-sun','wed,thu,fri,sat,sun')
    booll = df.shift_days_of_week.str.contains('wed-sat')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('wed-sat','wed,thu,fri,sat')
    booll = df.shift_days_of_week.str.contains('thu-mon')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('thu-mon','thu,fri,sat,sun,mon')
    booll = df.shift_days_of_week.str.contains('thu-sun')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('thu-sun','thu,fri,sat,sun')
    booll = df.shift_days_of_week.str.contains('thu-sat')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('thu-sat','thu,fri,sat')
    booll = df.shift_days_of_week.str.contains('fri-tue')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('fri-tue','fri,sat,sun,mon,tue')
    booll = df.shift_days_of_week.str.contains('fri-mon')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('fri-mon','fri,sat,sun,mon')
    booll = df.shift_days_of_week.str.contains('fri-sun')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('fri-sun','fri,sat,sun')
    booll = df.shift_days_of_week.str.contains('sat-wed')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('sat-wed','sat,sun,mon,tue,wed')
    booll = df.shift_days_of_week.str.contains('sat-tue')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('sat-tue','sat,sun,mon,tue')
    booll = df.shift_days_of_week.str.contains('sat-mon')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('sat-mon','sat,sun,mon')
    booll = df.shift_days_of_week.str.contains('sun-thu')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('sun-thu','sun,mon,tue,wed,thu')
    booll = df.shift_days_of_week.str.contains('sun-wed')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('sun-wed','sun,mon,tue,wed')
    booll = df.shift_days_of_week.str.contains('sun-tue')
    df.loc[booll, 'shift_days_of_week'] =  df[booll].shift_days_of_week.str.replace('sun-tue','sun,mon,tue')

###############################################################################
## Read Data
###############################################################################
print('Reading Data...')
df_train = pd.read_csv(PATH + 'WFS_Training.csv')
df_test  = pd.read_csv(PATH + 'WFS_TestFeatures.csv', encoding = "ISO-8859-1")

## Shuffle
df_train = df_train.sample(frac=1, random_state=2019)
df_train.reset_index(drop=True, inplace=True)

orig_cols = df_train.columns

print(np.mean(df_train[target]))

###############################################################################
## Combine Data & FE
###############################################################################
print('FE...')

print(' Add cand_id expanding mean...')
s=df_train.groupby(['cand_id','appt_1_date'],as_index=False)[['nhenoshow_flag']].agg(['count','sum'])
s.columns = [' '.join(col) for col in s.columns]
s.reset_index(inplace=True)
s['nhenoshow_flag_CumSum']=s.groupby(['cand_id'])[['nhenoshow_flag sum']].cumsum()
s['nhenoshow_flag_CumCount']=s.groupby(['cand_id'])[['nhenoshow_flag count']].cumsum()
s['nhenoshow_flag_CumMean']=s['nhenoshow_flag_CumSum']/ s['nhenoshow_flag_CumCount']
s['nhenoshow_flag_PrevMean']=s.groupby(['cand_id'])['nhenoshow_flag_CumMean'].shift(1)
df_train=df_train.merge(s[['cand_id','appt_1_date','nhenoshow_flag_PrevMean','nhenoshow_flag_CumCount']],how='left',on=['cand_id','appt_1_date'])
df_train.reset_index(drop=True, inplace=True)

df_test=df_test.merge(s[['cand_id','appt_1_date','nhenoshow_flag_PrevMean','nhenoshow_flag_CumCount']],how='left',on=['cand_id','appt_1_date'])
df_test.reset_index(drop=True, inplace=True)

df_test[target]  = -1
df_test['train'] = -1
n = int(df_train.shape[0] * .8)
df_train['train'] = 0
df_train.loc[0:n, 'train'] = 1

df_test = df_test[df_train.columns]
df_test.reset_index(drop=True, inplace=True)

df = pd.concat([df_train, df_test], axis=0)
df.reset_index(drop=True, inplace=True)



weather = pd.read_csv(PATH +'weather.csv')
weather.loc[weather.prec=='T','prec'] = 0
weather['prec'] = weather.prec.astype(float)

weather.loc[weather.snow=='T','prec'] = 0
weather['snow'] = weather.prec.astype(float)

df['date'] = df.appt_1_date.str[0:10]
df = pd.merge( df, weather, how='left', on='date')
df.reset_index(drop=True, inplace=True)

df.drop('date', inplace=True, axis=1)





#SST
df['sst'] = 0
booll = (~df.shift_start_time.isnull()) & (df.shift_start_time.str.contains('PM'))
df.loc[booll, 'sst'] = 12
for i in range(0,13):
    booll = (~df.shift_start_time.isnull()) & (df.shift_start_time.str.contains(str(i)+':'))
    df.loc[booll, 'sst'] = df.loc[booll, 'sst'] + i
booll = ~df.shift_start_time.isnull()
df.loc[booll, 'sst']  =  df[booll].sst + df[booll].shift_start_time.str[-5:-2].astype(int)/60
df.loc[df.shift_start_time.isnull(), 'sst'] = -1 #np.mean(df.loc[booll, 'sst'] )

## SET
df['set'] = 0
booll = (~df.shift_end_time.isnull()) & (df.shift_end_time.str.contains('PM'))
df.loc[booll, 'set'] = 12
for i in range(0,13):
    booll = (~df.shift_end_time.isnull()) & (df.shift_end_time.str.contains(str(i)+':'))
    df.loc[booll, 'set'] = df.loc[booll, 'set'] + i
booll = ~df.shift_end_time.isnull()
df.loc[booll, 'set']  =  df[booll].set + df[booll].shift_end_time.str[-5:-2].astype(int)/60
df.loc[df.shift_end_time.isnull(), 'set'] = -1 #np.mean(df.loc[booll, 'sst'] )

## STT
df['stt']  = df['set'] + 24  - df['sst'] 
df.loc[df.stt>24,'stt'] = df[df.stt>24].stt - 24

## Date Fields
df['appt_1_date'] = df.appt_1_date.str.replace('Z', '')
df['appt_1_date'] = df.appt_1_date.str.replace('T', ' ')
df['appt_1_date'] = pd.to_datetime(df.appt_1_date,infer_datetime_format=True)
df['app_created_date'] = df.app_created_date.str.replace('Z', '')
df['app_created_date'] = df.app_created_date.str.replace('T', ' ')
df['app_created_date'] = pd.to_datetime(df.app_created_date,infer_datetime_format=True)

df['appt_1_day_of_week']    = df['appt_1_date'].dt.day_name()
df['appt_1_hour']           = df['appt_1_date'].dt.hour
df['appt_1_week']           = df['appt_1_date'].dt.week


df['app_create_to_app_1']   = (df['app_created_date']-df['appt_1_date'])/ pd.offsets.Day(-1)

print(' Removing shift codes...')
narf1 = [f for f in df_train.shift_code.unique() if f not in df_test.shift_code.unique()]
narf2 = [f for f in df_test.shift_code.unique() if f not in df_train.shift_code.unique()]
narf3 = df_train.shift_code.value_counts().reset_index()
narf3 = list(narf3[narf3.shift_code<=3]['index'].unique())
df.loc[df.shift_code.isin( list(set( narf1 + narf2 + narf3 )) ), 'shift_code'] = -1

print(' Fixing shift_days_of_week...')
fix_shift_days_week(df)

## Add number of workdays
df['work_days'] = df.apply(lambda x: len(str(x.shift_days_of_week).split(',')),axis=1)

f='cand_education'
df.loc[df.cand_education=='AssociateÄôs / Trade School / Vocational',f] = 'Associate’s / Trade School / Vocational'
df.loc[df.cand_education=='Associateâs / Trade School / Vocational',f] = 'Associate’s / Trade School / Vocational'

print(' Turn cand_assess_overall_score into ordinal...')
f = 'cand_assess_overall_score'
df.loc[df[f]=='Highest'   , f] = 1
df.loc[df[f]=='High'      , f] = .75
df.loc[df[f]=='Moderate'  , f] = .5
df.loc[df[f]=='Low'       , f] = 0
df.loc[df[f]=='Ineligible', f] =-1
df.loc[df[f].isnull()     , f] =-1
df[f] = df[f].astype(int)

df['app_created_day']      = df['app_created_date'].dt.dayofyear
df['appt_1_day'     ]      = df['appt_1_date'     ].dt.dayofyear

narf = pd.DataFrame(df[df.train!=-1].groupby(by=['appt_1_day'])[target].count().reset_index())
narf.rename( {'nhenoshow_flag':'appt_1_day_apts'} , axis=1, inplace=True)
df = pd.merge( df, narf, how='left', on='appt_1_day')
df.reset_index(drop=True, inplace=True)


narf = pd.DataFrame(df[df.train!=-1].groupby(by=['app_created_day'])[target].count().reset_index())
narf.rename( {'nhenoshow_flag':'app_created_day_apts'} , axis=1, inplace=True)
df = pd.merge( df, narf, how='left', on='app_created_day')
df.reset_index(drop=True, inplace=True)


narf = pd.DataFrame(df[df.train!=-1].groupby(by=['appt_1_date'])[target].count().reset_index())
narf.rename( {'nhenoshow_flag':'appt_1_date_apts'} , axis=1, inplace=True)
df = pd.merge( df, narf, how='left', on='appt_1_date')
df.reset_index(drop=True, inplace=True)
df.appt_1_date_apts.fillna(0, inplace=True)


df['appt_1_apts_per'] = df.appt_1_date_apts/df.appt_1_day_apts






print(' One hot encoding app_esl_status...')
df_esl = pd.get_dummies(df.app_esl_status,dummy_na=True)
df_esl.columns= ['app_esl_status_ESL', 'app_esl_status_NonESL', 'app_esl_status_ESLNAN']
df = pd.concat( [df, df_esl], axis=1)
df.drop('app_esl_status', inplace=True, axis=1)
df.reset_index(drop=True, inplace=True)

print(' One hot encoding shift_schedule_type...')
f = 'shift_schedule_type'
df.loc[df[f]=='Flex Time (<19 hours)'     , f] = 'flex'
df.loc[df[f]=='Full-Time'                 , f] = 'full'
df.loc[df[f]=='Part-Time (20-29 hours)'   , f] = 'part'
df.loc[df[f]=='Reduced Time (30-39 hours)', f] = 'reduced'
df.loc[df[f].isnull()     , f] = 'Other'
df_sctype = pd.get_dummies(df.shift_schedule_type,dummy_na=False)
df_sctype.columns= ['shift_schedule_type_' + str(f) for f in list(df_sctype.columns)]
df = pd.concat( [df, df_sctype], axis=1)
df.drop('shift_schedule_type', inplace=True, axis=1)
df.reset_index(drop=True, inplace=True)

print(' Adding App-ID prefix...')
df['app_id'] = df.app_id.str.replace('App-','')
for i in range(1,8):
    df['app_id_' + str(i)] = df.app_id.str[i].astype(int)

print(' One hot encoding shift_startday...')
df_shift_startday = pd.get_dummies(df.shift_startday,dummy_na=False)
df_shift_startday.columns= ['shift_startday_' + str(f) for f in list(df_shift_startday.columns)]
df = pd.concat( [df, df_shift_startday], axis=1)
df.drop('shift_startday', inplace=True, axis=1)
df.reset_index(drop=True, inplace=True)


non_number_columns = df.dtypes[(df.dtypes == object) | (df.dtypes=='datetime64[ns]') ].index.values

for f in non_number_columns:
    print(f, df[f].value_counts().shape)

print(' Adding Events...')
dtemp = df.groupby(['cand_id','app_created_date','appt_1_date']).app_id.count().reset_index()
dtemp.rename(  {'app_id':'events'}, axis=1, inplace=True)
dtemp['events'] = 1/dtemp.events
df = pd.merge(df, dtemp, how='left', on=['cand_id','app_created_date','appt_1_date'])
df.reset_index(drop=True, inplace=True)

print(' Label Encoding non number columns...')
for column in non_number_columns:
    print('  ' + column)
    encoder = LabelEncoder().fit(df[column].astype(str))
    df[column] = encoder.transform(df[column].astype(str)).astype(np.int32)

features = [f for f in df.columns if f not in 
            ['ID', 'train',target,'app_id','cand_id','shift_end_time','shift_start_time'
             ,'appt_1_day', 'app_created_day','app_created_day_apts'
             ,'appt_1_apts_per','com_bin','app_created_hour','nhenoshow_flag_CumCount','nhenoshow_flag_PrevMean']]

features_lr =     [f for f in features if f not in non_number_columns if f not in []]
features_lr_pca = [f for f in features if f not in non_number_columns if f not in []]

features_lr_no_w = [f for f in features_lr if f not in weather.columns]

print('FE done.')


###############################################################################
## Parameters
###############################################################################

n_components = 32
pca = PCA(n_components=n_components)
pca.fit(  np.nan_to_num(  df[ features_lr_pca ].values  )  ) 

params_cat = {}
params_cat['loss_function'] = 'MultiClass'
params_cat['random_seed'] =   2019
params_cat['classes_count'] = 2
params_cat['l2_leaf_reg']   = 3
params_cat['depth']         = 8
params_cat['learning_rate'] = 0.05
params_cat['iterations']    = 250
params_cat['verbose'] = False

params_lgb = {}
params_lgb['objective']        = 'multiclass'
params_lgb['max_depth']        = 7
params_lgb['num_leaves']       = 32
params_lgb['feature_fraction'] = 0.95
params_lgb['bagging_fraction'] = 0.8
params_lgb['bagging_freq']     = 1
params_lgb['learning_rate']    = 0.05
params_lgb['verbosity']        = 2
params_lgb['verbose']          = 2
params_lgb['num_class']        = 2
params_lgb['lambda']          = 0.1
params_lgb['alpha']           = 0.1
params_lgb['random_state']    = 2019

X_train = preprocessing.scale(  np.nan_to_num( df[df.train >= 0][ features_lr ].values )  )
#X_train = np.concatenate(  (X_train, np.nan_to_num( df[ (df.train >= 0) & (df[target]==0)][features_lr][0:15000]) ), axis=0)

Y_train = df[df.train >= 0][ target   ].values
#Y_train = np.concatenate(  (Y_train, np.nan_to_num( df[ (df.train >= 0) & (df[target]==0)][target][0:15000]) ), axis=0)


X_test  = preprocessing.scale(  np.nan_to_num( df[df.train ==-1][ features_lr ].values )  )
EPOCHS  = 5



X_train2 = preprocessing.scale(  np.nan_to_num( df[df.train >= 0][ features_lr_no_w ].values )  )
X_test2  = preprocessing.scale(  np.nan_to_num( df[df.train ==-1][ features_lr_no_w ].values )  )














###############################################################################
## NB
###############################################################################
print(' Naive Bayes...')
y_hat_nb = np.zeros(X_test.shape[0])
y_oof_nb = np.zeros(X_train.shape[0])
kf      = KFold(n_splits = EPOCHS, shuffle = True, random_state=2019)
fold      = 1
for tr_idx, val_idx in kf.split(X_train, Y_train):
    X_tr, X_vl = X_train[tr_idx], X_train[val_idx, :]
    y_tr, y_vl = Y_train[tr_idx], Y_train[val_idx]
    model_nb = GaussianNB().fit(X_tr, y_tr)
 
    y_pred_train = model_nb.predict_proba(X_vl)[:,1]
    y_oof_nb[val_idx] = y_pred_train
    
    y_zero = max(np.mean(y_vl), 1-np.mean(y_vl))
    ACC    = accuracy_score(y_vl, (y_pred_train > 0.5  ).astype(int) )
    AUC    = roc_auc_score( y_vl, y_pred_train)
    LIFT   = ( ACC - y_zero )*100
    print('  NB: ', 'AUC:', AUC, 'ACC:', ACC, 'LIFT:', LIFT)

    y_hat_nb+= model_nb.predict_proba(X_test)[:,1] / EPOCHS
    fold+=1

print('  NB AVG: ', 'AUC:', roc_auc_score( Y_train, y_oof_nb), 'ACC:', accuracy_score(Y_train, (y_oof_nb > 0.5  ).astype(int) ))

df_test[target]= y_hat_nb
df_test[['ID',target]].to_csv(PATH + 'sub.nb.' + 'folds' + str(EPOCHS) + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_nb))

model_nb_full = GaussianNB().fit(X_train, Y_train)
y_hat_nb_full = model_nb_full.predict_proba(X_test)[:,1]

df_test[target]= y_hat_nb_full
df_test[['ID',target]].to_csv(PATH + 'sub.nb.' + 'full' + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_nb_full))

###############################################################################
## Logistic Regression
###############################################################################
print(' Logistic Regression...')
y_hat_lr = np.zeros(X_test.shape[0])
y_oof_lr = np.zeros(X_train.shape[0])
fold      = 1
kf      = KFold(n_splits = EPOCHS, shuffle = True, random_state=2019)

for tr_idx, val_idx in kf.split(X_train, Y_train):
    X_tr, X_vl = X_train[tr_idx], X_train[val_idx, :]
    y_tr, y_vl = Y_train[tr_idx], Y_train[val_idx]
    model_lr = LogisticRegression(random_state = 2019, C=1,tol=.0001).fit(X_tr, y_tr)
 
    y_pred_train = model_lr.predict_proba(X_vl)[:,1]
    y_oof_lr[val_idx] = y_pred_train
    
    y_zero = max(np.mean(y_vl), 1-np.mean(y_vl))
    ACC    = accuracy_score(y_vl, (y_pred_train > 0.5  ).astype(int) )
    AUC    = roc_auc_score( y_vl, y_pred_train)
    LIFT   = ( ACC - y_zero )*100
    print('  LR: ', 'AUC:', AUC, 'ACC:', ACC, 'LIFT:', LIFT)

    y_hat_lr+= model_lr.predict_proba(X_test)[:,1] / EPOCHS
    fold+=1

print('  LR AVG: ', 'AUC:', roc_auc_score( Y_train, y_oof_lr), 'ACC:', accuracy_score(Y_train, (y_oof_lr > 0.5  ).astype(int) ))

df_test[target]= y_hat_lr
df_test[['ID',target]].to_csv(PATH + 'sub.lr.' + 'folds' + str(EPOCHS) + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_lr))

model_lr_full = LogisticRegression(random_state = 2019, C=1,tol=.0001).fit(X_train, Y_train)
y_hat_lr_full = model_lr_full.predict_proba(X_test)[:,1]

df_test[target]= y_hat_lr_full
df_test[['ID',target]].to_csv(PATH + 'sub.lr.' + 'full' + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_lr_full))

##lr AVG:  AUC: 0.9225784879178707 ACC: 0.8501810376916124



###############################################################################
## MLP
###############################################################################

print(' MLP...')
y_hat_mlp = np.zeros(X_test2.shape[0])
y_oof_mlp = np.zeros(X_train2.shape[0])
kf      = KFold(n_splits = EPOCHS, shuffle = True, random_state=2019)

fold      = 1
for tr_idx, val_idx in kf.split(X_train2, Y_train):
    X_tr, X_vl = X_train2[tr_idx], X_train2[val_idx, :]
    y_tr, y_vl = Y_train[tr_idx], Y_train[val_idx]
    model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 2), random_state=2019).fit(X_tr, y_tr)
    y_pred_train = model_mlp.predict_proba(X_vl)[:,1]
    y_oof_mlp[val_idx] = y_pred_train
    y_zero = max(np.mean(y_vl), 1-np.mean(y_vl))
    ACC    = accuracy_score(y_vl, (y_pred_train > 0.5  ).astype(int) )
    AUC    = roc_auc_score( y_vl, y_pred_train)
    LIFT   = ( ACC - y_zero )*100
    print('  MLP: ', 'AUC:', AUC, 'ACC:', ACC, 'LIFT:', LIFT)
    y_hat_mlp+= model_mlp.predict_proba(X_test2)[:,1] / EPOCHS
    fold+=1

print('  MLP AVG: ', 'AUC:', roc_auc_score( Y_train, y_oof_mlp), 'ACC:', accuracy_score(Y_train, (y_oof_mlp > 0.5  ).astype(int) ))

df_test[target]= y_hat_mlp
df_test[['ID',target]].to_csv(PATH + 'sub.mlp.' + 'folds' + str(EPOCHS) + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_mlp))

model_mlp_full = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 2), random_state=2019).fit(X_train2, Y_train)
y_hat_mlp_full = model_mlp_full.predict_proba(X_test2)[:,1]

df_test[target]= y_hat_mlp_full
df_test[['ID',target]].to_csv(PATH + 'sub.mlp.' + 'full' + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_mlp_full))

#MLP AVG:  AUC: 0.9220744051631805 ACC: 0.8466376297356222
###############################################################################
## MLP ADAM
###############################################################################

params_mlp = {}
params_mlp['random_state']       = 2019
params_mlp['max_iter']           = 200
params_mlp['hidden_layer_sizes'] = (5,2)
params_mlp['alpha']              = 0.00025
params_mlp['solver']             = 'adam'
params_mlp['epsilon']            = 2e-4
params_mlp['activation']         = 'relu'
params_mlp['beta_1']             = .91
params_mlp['beta_2']             = .99


print(' MLP ADAM...')
y_hat_mlpa = np.zeros(X_test2.shape[0])
y_oof_mlpa = np.zeros(X_train2.shape[0])
kf      = KFold(n_splits = EPOCHS, shuffle = True, random_state=2019)
fold      = 1
for tr_idx, val_idx in kf.split(X_train2, Y_train):
    X_tr, X_vl = X_train2[tr_idx], X_train2[val_idx, :]
    y_tr, y_vl = Y_train[tr_idx], Y_train[val_idx]
    model_mlpa = MLPClassifier(**params_mlp).fit(X_tr, y_tr)
    y_pred_train = model_mlpa.predict_proba(X_vl)[:,1]
    y_oof_mlpa[val_idx] = y_pred_train
    y_zero = max(np.mean(y_vl), 1-np.mean(y_vl))
    ACC    = accuracy_score(y_vl, (y_pred_train > 0.5  ).astype(int) )
    AUC    = roc_auc_score( y_vl, y_pred_train)
    LIFT   = ( ACC - y_zero )*100
    print('  MLP ADAM: ', 'AUC:', AUC, 'ACC:', ACC, 'LIFT:', LIFT)
    y_hat_mlpa+= model_mlpa.predict_proba(X_test2)[:,1] / EPOCHS
    fold+=1

print('  MLP AVG ADAM: ', 'AUC:', roc_auc_score( Y_train, y_oof_mlpa), 'ACC:', accuracy_score(Y_train, (y_oof_mlpa > 0.5  ).astype(int) ))

# MLP AVG ADAM:  AUC: 0.9230259703009012 ACC: 0.8492528871765371
#MLP AVG ADAM:  AUC: 0.9223850033146234 ACC: 0.8482232201988756

df_test[target]= y_hat_mlpa
df_test[['ID',target]].to_csv(PATH + 'sub.mlpa.' + 'folds' + str(EPOCHS) + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_mlpa))

model_mlpa_full = MLPClassifier(**params_mlp).fit(X_train2, Y_train)
y_hat_mlpa_full = model_mlpa_full.predict_proba(X_test2)[:,1]

df_test[target]= y_hat_mlpa_full
df_test[['ID',target]].to_csv(PATH + 'sub.mlpa.' + 'full' + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_mlpa_full))






###############################################################################
## LGB
###############################################################################
params_lgb['lambda']          = 0.10
params_lgb['max_depth']       = 3

print(' lgbboost...')
y_hat_lgb = np.zeros(X_test.shape[0])
y_oof_lgb = np.zeros(X_train.shape[0])
fold      = 1
kf      = KFold(n_splits = EPOCHS, shuffle = True, random_state=2019)
kvs = '.'.join([str(k) + '=' + str(v).replace(':','') for k,v in zip(list(params_lgb), [str(value) for value in params_lgb.values()])])
nbr_lgb = 300

for tr_idx, val_idx in kf.split(X_train, Y_train):
    postfix = 'epochs=' + str(EPOCHS) + 'fold=' + str(fold) + kvs
    filename = PATH + '/pickles/model_lgb.' + postfix + '.pkl'
    X_tr, X_vl = X_train[tr_idx], X_train[val_idx, :]
    y_tr, y_vl = Y_train[tr_idx], Y_train[val_idx]
    if os.path.isfile(filename) and 1==0:
        model_lgb = pickle.load(open(filename, 'rb'))
    else:
        model_lgb = lgb.train(params_lgb, lgb.Dataset(X_tr, label = y_tr), num_boost_round=nbr_lgb)
        s = pickle.dump(model_lgb, open(filename,'wb'))
    y_pred_train = model_lgb.predict(X_vl)[:,1]
    y_oof_lgb[val_idx] = y_pred_train
    
    y_zero = max(np.mean(y_vl), 1-np.mean(y_vl))
    ACC    = accuracy_score(y_vl, (y_pred_train > 0.5  ).astype(int) )
    AUC    = roc_auc_score( y_vl, y_pred_train)
    LIFT   = ( ACC - y_zero )*100
    print('  lgb: ', 'AUC:', AUC, 'ACC:', ACC, 'LIFT:', LIFT)

    y_hat_lgb+= model_lgb.predict(X_test)[:,1] / EPOCHS
    fold+=1

print('  lgb AVG: ', 'AUC:', roc_auc_score( Y_train, y_oof_lgb), 'ACC:', accuracy_score(Y_train, (y_oof_lgb > 0.5  ).astype(int) ))

df_test[target]= y_hat_lgb
df_test[['ID',target]].to_csv(PATH + 'sub.lgb.' + postfix + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_lgb))

model_lgb_full = lgb.train(params_lgb, lgb.Dataset(X_train, label = Y_train), num_boost_round=nbr_lgb)
y_hat_lgb_full = model_lgb_full.predict(X_test)[:,1]


df_test[target]= y_hat_lgb_full
df_test[['ID',target]].to_csv(PATH + 'sub.lgb.' + 'full' + '.csv', index = False, float_format = '%.4f')


print('  ', np.mean(y_hat_lgb_full))

feature_imp = pd.DataFrame(sorted(zip(model_lgb.feature_importance(),features_lr)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

###############################################################################
## CatBoost
###############################################################################
print(' Catboost...')
params_cat = {}
params_cat['loss_function'] = 'MultiClass'
params_cat['random_seed'] =   2019
params_cat['classes_count'] = 2
params_cat['l2_leaf_reg']   = 3
params_cat['depth']         = 7
params_cat['learning_rate'] = 0.043
params_cat['iterations']    = 250
params_cat['verbose'] = False

y_hat_cat = np.zeros(X_test.shape[0])
y_oof_cat = np.zeros(X_train.shape[0])
fold      = 1
kf      = KFold(n_splits = EPOCHS, shuffle = True, random_state=2019)

kvs = '.'.join([str(k) + '=' + str(v).replace(':','') for k,v in zip(list(params_cat), [str(value) for value in params_cat.values()])])

for tr_idx, val_idx in kf.split(X_train, Y_train):
    postfix = 'epochs=' + str(EPOCHS) + 'fold=' + str(fold) + kvs
    filename = PATH + '/pickles/model_cat.' + postfix + '.pkl'
    X_tr, X_vl = X_train[tr_idx], X_train[val_idx, :]
    y_tr, y_vl = Y_train[tr_idx], Y_train[val_idx]
    if os.path.isfile(filename) and 1 ==0:
        model_cat = pickle.load(open(filename, 'rb'))
    else:
        model_cat = CatBoostClassifier(**params_cat).fit(X_tr, y_tr)
        s = pickle.dump(model_cat, open(filename,'wb'))
    y_pred_train = model_cat.predict_proba(X_vl)[:,1]
    y_oof_cat[val_idx] = y_pred_train
    
    y_zero = max(np.mean(y_vl), 1-np.mean(y_vl))
    ACC    = accuracy_score(y_vl, (y_pred_train > 0.5  ).astype(int) )
    AUC    = roc_auc_score( y_vl, y_pred_train)
    LIFT   = ( ACC - y_zero )*100
    print('  CAT: ', 'AUC:', AUC, 'ACC:', ACC, 'LIFT:', LIFT)

    y_hat_cat+= model_cat.predict_proba(X_test)[:,1] / EPOCHS
    fold+=1

print('  CAT AVG: ', 'AUC:', roc_auc_score( Y_train, y_oof_cat), 'ACC:', accuracy_score(Y_train, (y_oof_cat > 0.5  ).astype(int) ))

#CAT AVG:  AUC: 0.9037461050815381 ACC: 0.8122283830361157
#CAT AVG:  AUC: 0.9038458707383525 ACC: 0.8119866771728148  .048
#CAT AVG:  AUC: 0.9038025160978359 ACC: 0.8124314159612884
#CAT AVG:  AUC: 0.9061203503724364 ACC: 0.8170431638330683 .035
#CAT AVG:  AUC: 0.9065850056408526 ACC: 0.8180051531690056 .032
#CAT AVG:  AUC: 0.9071448281569294 ACC: 0.8193877107070864

df_test[target]= y_hat_cat
df_test[['ID',target]].to_csv(PATH + 'sub.cat.' + postfix + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_cat))

model_cat_full = CatBoostClassifier(**params_cat).fit(X_train, Y_train)
y_hat_cat_full = model_cat_full.predict_proba(X_test)[:,1]

df_test[target]= y_hat_cat_full
df_test[['ID',target]].to_csv(PATH + 'sub.cat.' + 'full' + '.csv', index = False, float_format = '%.4f')


print('  ', np.mean(y_hat_cat_full))

feature_imp = pd.DataFrame(sorted(zip(model_cat.get_feature_importance(),features_lr)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

###############################################################################
## XGBOOST
###############################################################################
## Load From EIDER
print(' XGB...')
y_hat_xgb = pd.read_csv(PATH + 'submission.xgb.20190805_173631.csv').nhenoshow_flag.values

df_test[target]= y_hat_xgb
df_test[['ID',target]].to_csv(PATH + 'sub.xgb.' + 'full' + '.csv', index = False, float_format = '%.4f')

print('  ', np.mean(y_hat_xgb))


###############################################################################
## Ensemble
###############################################################################

#y_hat_ens = y_hat_xgb
#y_hat_ens = (y_hat_cat_full*6 + y_hat_xgb*5 + y_hat_mlp_full*4 + y_hat_lgb_full*3 + y_hat_lr_full*2 + y_hat_nb_full*1)/(21)

y_hat_ens = (y_hat_cat_full**2 + y_hat_xgb**4 + y_hat_mlp_full**4  + y_hat_mlpa_full**4 )/(4)
print('ENS: ', np.mean(y_hat_ens))

##CAT 0.923134310788
##XGB 0.922942891179
##MLP 0.923136475212

##LGB 0.922057108293
##LR  0.922026209810
##NB  0.914270975836


narf = df_train.cand_id.value_counts().reset_index()
narf.columns = ['cand_id', 'events']
v1 = np.mean(df_train[df_train.cand_id.isin( narf[narf.events==1].cand_id   )][target])
v2 = np.mean(df_train[df_train.cand_id.isin( narf[narf.events>1].cand_id   )][target])

narf = pd.concat(  [ df_train[['ID', 'cand_id']], df_test[['ID', 'cand_id']] ], axis=0 ).cand_id.value_counts().reset_index()
narf.columns = ['cand_id', 'events']

df_test['narf'] = v1
df_test.loc[ df_test.cand_id.isin( narf[narf.events>1].cand_id)  , 'narf'] = v2

###############################################################################
## Stacking
###############################################################################
df_train['y_hat_lr']    = y_oof_lr**2
df_train['y_hat_nb']    = y_oof_nb**2
df_train['y_hat_mlp']   = y_oof_mlp**2
df_train['y_hat_mlpa']  = y_oof_mlpa**2
df_train['y_hat_lgb']   = y_oof_lgb**2
df_train['y_hat_cat']   = y_oof_cat
df_train['y_hat_xgb']   = pd.read_csv(PATH + 'y_hat_xgb_train').y_oof_xgb.values**2

df_test['y_hat_lr']    = y_hat_lr_full**2
df_test['y_hat_nb']    = y_hat_nb_full**2
df_test['y_hat_mlp']   = y_hat_mlp_full**2
df_test['y_hat_mlpa']  = y_hat_mlpa_full**2
df_test['y_hat_lgb']   = y_hat_lgb_full**2
df_test['y_hat_cat']   = y_hat_cat
df_test['y_hat_xgb']   = y_hat_xgb**2

cols = ['y_hat_lr', 'y_hat_mlp', 'y_hat_mlpa', 'y_hat_cat', 'y_hat_xgb']
x_train = df_train[cols]
x_test  = df_test[cols]

model_rf = RandomForestClassifier(n_estimators=300, max_depth = 4, random_state = 2019).fit( x_train, Y_train)
y_hat_rf = model_rf.predict_proba( x_test )[:,1]


y_hat_ens = (y_hat_cat**2 + y_hat_xgb**4 + y_hat_mlp_full**4  + y_hat_mlpa_full**4 + y_hat_rf**4)/(5)
print('ENS: ', np.mean(y_hat_ens))


###############################################################################
## Write Outputs
###############################################################################
df_test[target]= y_hat_rf
df_test[['ID',target]].to_csv(PATH + 'submission.stack.{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index = False, float_format = '%.4f')


df_test[target]= y_hat_ens
df_test[['ID',target]].to_csv(PATH + 'submission.ens.{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index = False, float_format = '%.4f')

df_test[target]= df_test.narf *0.1 + y_hat_ens *0.9
print(np.mean(df_test[target]))
df_test[['ID',target]].to_csv(PATH + 'submission.narf.{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index = False, float_format = '%.4f')

## Save Python To File
os.system("copy " + PATH.replace('/','\\') + "wfs3.py " + PATH.replace('/','\\') + "wfs3.py.{}".format(datetime.now().strftime('%Y%m%d_%H%M%S')))











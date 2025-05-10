from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import time
from sklearn.metrics import confusion_matrix 
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pyAppDomain import AppDomainFpSimilarity
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import pickle
from pyAppDomain import AppDomainX, AppDomainFpSimilarity
from metAppDomain_ADM import NSG, NSGVisualizer

from metAppDomain_ADM import NSG
def rigidWt(x, sCutoff=0.75):
    y = np.ones(shape=x.shape)
    y[x < sCutoff] = 0
    return y
#
def expWt(x, a=10, eps=1e-6):
    # a = 3, Liu et al. JCIM, 2018
    return np.exp(-a*(1-x)/(x + eps))
    

wtFunc1a = rigidWt
kw1a = {'sCutoff':0.85}
wtFunc2a = rigidWt
kw2a = {'sCutoff':0.85}
wtFunc1b = expWt
kw1b = {'a':5}
wtFunc2b = expWt
kw2b = {'a':5}

#import data
df_train = pd.read_csv('/home/dell/wuchao/ML/1697FS/AD/1697_trainset.csv',index_col = 'CARID')
df_ext = pd.read_csv('/home/dell/wuchao/ML/1697FS/AD/1697_testset.csv',index_col = 'CARID')
print(df_ext.columns)


# NSG
nsg = NSG(df_train,yCol='Carcinogenicity',smiCol='neuSmi')
#nsg.calcPairwiseSimilarityWithFp('Morgan(bit)',radius=2,nBits=1024)
nsg.calcPairwiseSimilarityWithFp('MACCS_keys')
dfQTSM = nsg.genQTSM(df_ext,'neuSmi')
df_train = df_train[['neuSmi','Carcinogenicity']]
df_ext = df_ext[['neuSmi','Carcinogenicity']]
df_ext = df_ext.join(nsg.queryADMetrics(dfQTSM, wtFunc1=wtFunc1a,kw1=kw1a, wtFunc2=wtFunc2a,kw2=kw2a,code='|rigid'))
df_ext = df_ext.join(nsg.queryADMetrics(dfQTSM, wtFunc1=wtFunc1b,kw1=kw1b, wtFunc2=wtFunc2b,kw2=kw2b,code='|exp'))
df_ext.to_csv('dfEx_ADMetrics_Classifier.csv')
print(df_ext.columns)

start = time.time()
save_time = str(time.ctime()).replace(':','-').replace(' ','_')
#set different ρs and IA cutoff values according your requirment
#ρsDict = {
#'rigid':[ 1, 2, 3, 5, 12, 20], 
#'exp':[ 0.01, 0.15,  0.2, 0.4, 0.8, 1]}
#a= 8
ρsDict = {
'rigid':[ 0.01, 0.1, 0.5, 1,1.3, 1.5, 1.8, 2,2.3, 2.5, 2.8, 3,3.3, 3.5, 3.8, 4, 4.5, 5,5.3, 5.5, 5.8, 6, 6.3, 6.5, 6.8, 7, 7.5, 8, 9, 10, 15, 20], 
'exp':[0.0001,0.001, 0.01, 0.1, 0.5, 1,1.3, 1.5, 1.8, 2,2.3, 2.5, 2.8, 3,3.3, 3.5, 3.8, 4, 4.3, 4.5, 4.8, 5, 5.3, 5.5, 5.8, 6, 6.3, 6.5, 6.8, 7, 7.5, 8, 9, 10, 15, 20]}

os.chdir('/home/dell/wuchao/ML/1697FS')
df1782 = pd.read_csv('PubChem_1697_FS.csv')
df_test = pd.read_csv('/home/dell/wuchao/ML/1697FS/AD/1697_testset.csv')
#X_test = df1782.loc[df1782['CARID'].isin(df_test.CARID)]
X_test = pd.merge(df_test, df1782, on='CARID', how='inner')
X_test.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697_XTEST_F.csv', index=False)
#print(X_test)
X_te = X_test.iloc[:,14:-1]
y_tr =  df_test['Carcinogenicity']
with open("/home/dell/wuchao/ML/1697FS/model/RF_PubChem_1697_FS.pkl", 'rb') as mymodel:
    best_model=pickle.load(mymodel)
    y_pred = best_model.predict(X_te)
    #print(y_pred)
    y_proba = best_model.predict_proba(X_te)
    print(y_proba)
    #y_tr = y_tr.astype(int)
    #y_pred = (y_proba > 0.5).astype(int)
    #print(y_pred)
    #y_pred = (y_proba[:, 1] > 0.5).astype(int)
    confusion = metrics.confusion_matrix(y_tr, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    auc_score = roc_auc_score(y_tr, y_proba[:, 1])  # AUC for the positive class
    print(f'AUC Score: {auc_score}')

    df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])

    # 合并 y_test_proba
    df_y_proba = pd.DataFrame(y_proba[:, 1], columns=['y_proba'])  # 选择正类的概率

    # 合并 X_te, y_pred 和 y_test_proba
    df_combined = pd.concat([df_y_pred, df_y_proba], axis=1)
    df_fplot = pd.concat([X_test, df_combined], axis=1)
    df_fplot.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697_XTEST_Y.csv', index=False)


df_plot = pd.read_csv('/home/dell/wuchao/ML/1697FS/AD/1697_XTEST_Y.csv',index_col = 'CARID')
y_tr = df_plot['Carcinogenicity_y']
y_prob = df_plot['y_proba']
y_pred = (y_prob > 0.5).astype(int)
#y_pred = df_ext['pred']
#yt = df_ext['y_true']
#yprob = df_ext['yExt_probA']
# the threshold value 0.5 can also be changed according your actual requirment
#yp = (yprob > 0.5).astype(int)

#IAVal_List = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
IAVal_List = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35,0.4, 0.45, 0.5, 0.6]

for code in ['rigid','exp']:
    dfn = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfRA = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfAUC = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfSE = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfBA = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])
    dfMCC = pd.DataFrame(index=IAVal_List,columns=ρsDict[code])

    for densLB in dfAUC.columns:
        for LdUB in dfAUC.index:
            adi = df_ext.index[(df_ext['simiDensity|'+code] >= densLB)&(df_ext['simiWtLD_w|'+code] <= LdUB)]
            dfn.loc[LdUB,densLB] = adi.shape[0]
            try:
                dfAUC.loc[LdUB,densLB] = metrics.roc_auc_score(y_tr[adi],y_prob[adi])
            except:
                dfAUC.loc[LdUB,densLB] = np.nan
            dfRA.loc[LdUB,densLB] = metrics.accuracy_score(y_tr[adi],y_pred[adi])
            dfSE.loc[LdUB,densLB] = metrics.recall_score(y_tr[adi],y_pred[adi])
            dfBA.loc[LdUB,densLB] = metrics.balanced_accuracy_score(y_tr[adi],y_pred[adi])
            #dfSP.loc[LdUB,densLB] = metrics.precision_score(y_tr[adi],y_pred[adi])
            dfMCC.loc[LdUB,densLB] = metrics.matthews_corrcoef(y_tr[adi],y_pred[adi])

    #print the performance of classifier with within ADSAL on the external validation set
    dfn.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697ADResults/56_85_Classifier_{:s}_AD_n_{}.csv'.format(code,save_time))
    dfAUC.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697ADResults/56_85_Classifier{:s}_AD_AUC_{}.csv'.format(code,save_time))
    dfBA.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697ADResults/56_85_Classifier{:s}_AD_BA_{}.csv'.format(code,save_time))
    dfRA.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697ADResults/56_85_Classifier{:s}_AD_RA_{}.csv'.format(code,save_time))
    #dfSP.to_csv('Classifier{:s}_AD_SP.csv'.format(code))
    dfSE.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697ADResults/56_85_Classifier{:s}_AD_SE_{}.csv'.format(code,save_time))
    dfMCC.to_csv('/home/dell/wuchao/ML/1697FS/AD/1697ADResults/56_85_Classifier{:s}_AD_MCC_{}.csv'.format(code,save_time))

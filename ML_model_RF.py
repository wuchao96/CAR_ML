import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import StratifiedKFold #交叉验证
import os
from sklearn.model_selection import cross_validate,GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix 
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import pickle

os.chdir('F:/W/input data/1697 FS')
df1697 = pd.read_csv('PubChem_1697_FS.csv')
df1697

X = df1697.iloc[:,7:-1]
y =df1697['Carcinogenicity']

splitned = StratifiedShuffleSplit(n_splits= 2, test_size = 0.3, random_state = 42)
for train_index, test_index in splitned.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_te = X.loc[train_index], X.loc[test_index]
    y_tr, y_te = y.loc[train_index], y.loc[test_index]

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=420)
for train, val in rkf.split(X_tr,y_tr):
    #print("TRAIN:", train, "TEST:", val)
    df_Xtr, df_Xval = X_tr.iloc[train], X_tr.iloc[val]
    df_ytr, df_yval= y_tr.iloc[train], y_tr.iloc[val]
X_train = df_Xtr.iloc[:,::]
X_val = df_Xval.iloc[:,::]
y_train = y_tr.iloc[train]
y_train = y_train.values.ravel()
y_val = y_tr.iloc[val]
y_val = y_val.values.ravel()

clf = RandomForestClassifier(random_state=10)

def evaluate_model(clf,X_val, y_val):
    cv = rkf
    scoring = {'accuracy':'accuracy','balanced_accuracy':'balanced_accuracy',
            'recall':'recall','roc_auc':'roc_auc'}
    
    param_grid = {'criterion':['gini', 'entropy'],
          'max_depth':np.arange(10,50,10),
          'n_estimators': [500,1000,2000,3000],
          'max_features': ['sqrt']}
    
    gridCV = GridSearchCV(clf,param_grid,scoring =scoring ,
                      refit= 'roc_auc',cv = cv)
    grid_result = gridCV.fit(X_train,y_train)
    
    with open('model/RF_PubChem_1697_FS.pkl', 'wb') as f:
        pickle.dump(gridCV, f)
    #pd.DataFrame(grid_result.cv_results_).to_csv('gridCV_+RF 6d 805 2.csv')
    return grid_result

score = evaluate_model(clf,X_val, y_val)
results = pd.DataFrame(score.cv_results_)
results.to_csv('1697results/PubChem_10cross_RF.csv')

y_pred = score.predict(X_te)

confusion = metrics.confusion_matrix(y_te, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print ("TP:", TP)
print ("TN:", TN)
print ("FP:", FP)
print ("FN:", FN)

#计算 Sensitivity(Recall) = TPR = TP / (TP + FN)

recall = metrics.recall_score(y_te, y_pred)
print (metrics.recall_score(y_te, y_pred))

specificity_test = TN / float(TN+FP)
specificity_test

y_test_proba = score.predict_proba(X_te)
fpr, tpr, thresholds = metrics.roc_curve(y_te, y_test_proba[:, 1])
auc = metrics.auc(fpr, tpr)
print('AUC:',auc)

from sklearn.metrics import matthews_corrcoef
matthews_corrcoef_test = matthews_corrcoef(y_te, y_pred)
matthews_corrcoef_test

df_test = pd.DataFrame(
  {"Sensitivity":[recall],"specificity":[specificity_test],
  "Aroc":[auc],"matthews_corrcoef":[matthews_corrcoef_test]},
   index=['RF + PubChem fingerprints']
) 

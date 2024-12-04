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

os.chdir('F:/W/input data/1785 FS')
df1785 = pd.read_csv('df1785 MACCS fingerprints FS.csv')
df1785

X = df1785.iloc[:,6:-1]
y =df1785['Activity']

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

clf = GradientBoostingClassifier(random_state=10)

def evaluate_model(clf,X_val, y_val):
    cv = rkf
    scoring = {'accuracy':'accuracy','balanced_accuracy':'balanced_accuracy',
            'recall':'recall','roc_auc':'roc_auc'}
    
    param_grid = {'learning_rate':[0.001,0.01,0.1],
               'max_depth': np.arange(10,50,10),
               'n_estimators': [  500,1000, 2000,3000],
               'max_features': ['sqrt']}
    
    gridCV = GridSearchCV(clf,param_grid,scoring =scoring ,
                      refit= 'roc_auc',cv = cv)
    grid_result = gridCV.fit(X_train,y_train)
    
    with open('GBDT_CAR_1785_MACCS.pickle', 'wb') as f:
        pickle.dump(gridCV, f)
    return grid_result
score = evaluate_model(clf,X_val, y_val)
score
pd.DataFrame(score.cv_results_).to_csv('F:/W/input data/1785 results/gridCV_+GBDT MACCS.csv')

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
   index=['GBDT+MACCS fingerprints']
) 

#RF
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold #交叉验证
import os
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_validate,GridSearchCV,cross_val_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

os.chdir('/home/dell/wuchao/ML/1697FS/1697FS_12')
file_list = [file for file in os.listdir() if file.endswith('.csv')]

if not os.path.exists('model'):
    os.makedirs('model')

# Create '1785results' folder if it doesn't exist
if not os.path.exists('1697results'):
    os.makedirs('1697results')


for file in file_list:
    # Read the fingerprint file
    df1697 = pd.read_csv(file)
    X = df1697.iloc[:,7:-1]
    y =df1697['Carcinogenicity']
    X_smi = df1697.iloc[:,:-1]

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
    #X_train
    X_val = df_Xval.iloc[:,::]
    #X_val
    y_train = y_tr.iloc[train]
    y_train = y_train.values.ravel()
    #y_train
    y_val = y_tr.iloc[val]
    y_val = y_val.values.ravel()
    #y_val


    clf = RandomForestClassifier(random_state=10)

    def evaluate_model(clf,X_val, y_val):
        cv = rkf
        scoring = {'accuracy':'accuracy','balanced_accuracy':'balanced_accuracy',
                'recall':'recall','roc_auc':'roc_auc', 'f1':'f1', 'precision':'precision'}
        
        param_grid = {'criterion':['gini', 'entropy'],
              'max_depth':np.arange(10,50,10),
              'n_estimators': [500,1000,2000,3000],
              'max_features': ['sqrt']}
        
        gridCV = GridSearchCV(clf,param_grid,scoring =scoring ,
                          refit= 'roc_auc',cv = cv)
        grid_result = gridCV.fit(X_train,y_train)
        
        with open('model/RF_' + file.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(gridCV, f)
        #pd.DataFrame(grid_result.cv_results_).to_csv('gridCV_+RF 6d 805 2.csv')
        return grid_result

    score = evaluate_model(clf,X_val, y_val)
    score
    results = pd.DataFrame(score.cv_results_)
    
    results.to_csv('1697results/' + file.split('.')[0] + '_10_RF.csv')
    
    df_rank_roc = results.loc[(results['rank_test_roc_auc'] == 1)]

    df_best=df_rank_roc[['params','mean_test_accuracy', 'std_test_accuracy', 
                    'mean_test_balanced_accuracy', 'std_test_balanced_accuracy',
                    'mean_test_recall','std_test_recall',
                    'mean_test_roc_auc','std_test_roc_auc',
                    'mean_test_precision', 'std_test_precision', 
                    'mean_test_f1', 'std_test_f1'
                   ]]
    df_best

    mean_test_SP = 2*df_rank_roc[['mean_test_balanced_accuracy']].values-df_rank_roc[['mean_test_recall']].values

    std_test_SP = abs(2*df_rank_roc[[ 'std_test_balanced_accuracy']].values-df_rank_roc[['std_test_recall']].values)

    df_best.insert(7,str('mean_test_specificity'),mean_test_SP)
    df_best.insert(8,str('std_test_specificity'),std_test_SP )

    
    df_best.to_csv('1697results/' + file.split('.')[0] + '_10best_RF.csv')        
    y_pred = score.predict(X_te)

    from sklearn.metrics import confusion_matrix 
    import sklearn.metrics as metrics
    from sklearn.metrics import roc_auc_score

    confusion = metrics.confusion_matrix(y_te, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print ("TP:", TP)
    print ("TN:", TN)
    print ("FP:", FP)
    print ("FN:", FN)

    #计算Accuracy 
    print ((TP+TN) / float(TP+TN+FN+FP))
    print (metrics.accuracy_score(y_te, y_pred))
    Accuracy_test = metrics.accuracy_score(y_te, y_pred)
     

    #计算 Sensitivity(Recall) = TPR = TP / (TP + FN)
    print (TP / float(TP+FN))
    recall = metrics.recall_score(y_te, y_pred)
    print (metrics.recall_score(y_te, y_pred))

    print (TN / float(TN+FP))
    specificity_test = TN / float(TN+FP)

    y_test_proba = score.predict_proba(X_te)
    fpr, tpr, thresholds = metrics.roc_curve(y_te, y_test_proba[:, 1])
    auc = metrics.auc(fpr, tpr)
    print('AUC:',auc)
    
    f1 = f1_score(y_te, y_pred)
    print("F1 Score:", f1)

    # 计算 Precision
    precision = precision_score(y_te, y_pred)
    print("Precision:", precision)

    from sklearn.metrics import matthews_corrcoef
    matthews_corrcoef_test = matthews_corrcoef(y_te, y_pred)
    matthews_corrcoef_test

    df_test = pd.DataFrame(
      {"Accuracy":[Accuracy_test], "Sensitivity":[recall],"specificity":[specificity_test],
      "Aroc":[auc],"F1 Score": [f1], "Precision": [precision], "Matthews Corr Coef": [matthews_corrcoef_test],
      "TP": [TP], "TN": [TN], "FP": [FP], "FN": [FN]},
       index=['RF+fingerprints']
    )

    df_test

    df_test.to_csv('1697results/' + file.split('.')[0] + '_test_RF.csv')



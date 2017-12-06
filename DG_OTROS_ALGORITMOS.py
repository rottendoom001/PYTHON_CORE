import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn import svm, neighbors, linear_model, tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# //////////// LEEMOS TODOS LOS ELEMENTOS DESPUES DEL HPS Y FFT /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/data/final_fft_hps_espec_ceps_frq.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
train_x = df.ix[:,0:ancho - 2].values
train_y = df.ix[:, ancho - 1 ].values

# //////////// LEEMOS TODOS LOS ELEMENTOS DESPUES DEL HPS Y FFT TEST /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/data/final_fft_hps_espec_ceps_frq_t.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
test_x = df.ix[:,0:ancho - 2].values
test_y = df.ix[:, ancho - 1 ].values
#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// RANDOM FOREST ////////////////////")
forest_model = RandomForestClassifier(max_depth=10, n_estimators=5, max_features=1)
forest_model.fit(train_x, train_y)

pred_forest_test = forest_model.predict(test_x)
pred_forest_train = forest_model.predict(train_x)

print(pd.crosstab(test_y, pred_forest_test, rownames=["Actual"], colnames=["Predicted"]))

acc_forest_test = accuracy_score(test_y, pred_forest_test)
acc_forest_train = accuracy_score(train_y, pred_forest_train)

if acc_forest_test > 0.79:
    pickle.dump(forest_model, open("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/random_forest.dat", "wb"))

print("Accuracy R. Forest test", acc_forest_test, ";\nAccuracy R. Forest train ", acc_forest_train)
#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// DECISION TREE ////////////////////")

tree_model = tree.DecisionTreeClassifier(max_depth = 15, max_leaf_nodes=5)
tree_model.fit(train_x, train_y)

pred_tree_test = tree_model.predict(test_x)
pred_tree_train = tree_model.predict(train_x)

print (pd.crosstab(test_y, pred_tree_test, rownames=["Actual"], colnames=["Predicted"]))
acc_tree_test = accuracy_score(test_y, pred_tree_test)
acc_tree_train = accuracy_score(train_y, pred_tree_train)

if acc_tree_test > 0.79:
    pickle.dump(tree_model, open("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/tree.dat", "wb"))


print("Accuracy Tree test", acc_tree_test, ";\nAccuracy Tree train ", acc_tree_train)

#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// SVN ////////////////////")
svm_model = svm.SVC(gamma=0.001, C=100.)
svm_model.fit(train_x, train_y)

pred_svm_test = svm_model.predict(test_x)
pred_svm_train = svm_model.predict(train_x)

print(pd.crosstab(test_y, pred_svm_test, rownames=["Actual"], colnames=["Predicted"]))
acc_svm_test = accuracy_score(test_y, pred_svm_test)
acc_svm_train = accuracy_score(train_y, pred_svm_train)

if acc_svm_test > 0.79:
    pickle.dump(svm_model, open("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/svm_model.dat", "wb"))


print("Accuracy SVM test", acc_svm_test, ";\nAccuracy SVM train ", acc_svm_train)

#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// LOGISTIC REGRESION ////////////////////")
lr_model = linear_model.LogisticRegression(C=1e2)
lr_model.fit(train_x, train_y)

pred_lr_test = lr_model.predict(test_x)
pred_lr_train = lr_model.predict(train_x)
print(pd.crosstab(test_y, pred_lr_test, rownames=["Actual"], colnames=["Predicted"]))

acc_lr_test = accuracy_score(test_y, pred_lr_test)
acc_lr_train = accuracy_score(train_y, pred_lr_train)

if acc_lr_test > 0.79:
    pickle.dump(lr_model, open("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/lr_model.dat", "wb"))


print("Accuracy LR test", acc_lr_test, ";\nAccuracy LR train ", acc_lr_train)

#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// XGBOOST ////////////////////")
XGBoost = XGBClassifier()
XGBoost.fit(train_x, train_y)

pred_xgb_test = XGBoost.predict(test_x)
pred_xgb_train = XGBoost.predict(train_x)


test = [
    5.87473684e+02,   2.95846426e+02,   5.87473684e+02,   8.43631579e+02,
    3.31315789e+02,   5.12315789e+02,  -7.26309571e-16,  -1.20000010e+00,
    8.35074086e+00,   2.18079111e-01,   2.53684211e+02,   7.51578947e+01,
    1.09978947e+03,   1.02463158e+03,   2.05465379e-04,   6.87693252e-04,
    1.06891128e+01,  -3.06876164e-01,   2.53684211e+02,   2.52842105e+02,
    1.51578947e+02,   2.57894737e+02,   2.53052632e+02,   2.55789474e+02,
    2.55157895e+02,   2.54315789e+02,   1.51157895e+02,   2.54947368e+02
]

print ("R=", XGBoost.predict(test))
print (train_x[4000])
print (pred_xgb_train[4000])

#np.savetxt("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/pred.txt", pred_xgb_train, fmt='%5s',delimiter=',')

print(pd.crosstab(test_y, pred_xgb_test, rownames=["Actual"], colnames=["Predicted"]))
acc_xgb_test = accuracy_score(test_y, pred_xgb_test)

print(pd.crosstab(train_y, pred_xgb_train, rownames=["Actual"], colnames=["Predicted"]))
acc_xgb_train = accuracy_score(train_y, pred_xgb_train)

if acc_xgb_test > 0.79:
    pickle.dump(XGBoost, open("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/xgboost_model.dat", "wb"))


print("Accuracy XGBoost test", acc_xgb_test, ";\nAccuracy LR train ", acc_xgb_train)

#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// RBF SGDClassifier ////////////////////")

rbf_feature = RBFSampler(gamma=2, random_state=1)
train_x = rbf_feature.fit_transform(train_x)
test_x = rbf_feature.fit_transform(test_x)

clf = SGDClassifier()
clf.fit(train_x, train_y)

pred_clf_test = clf.predict(test_x)
pred_clf_train = clf.predict(train_x)

print(pd.crosstab(test_y, pred_clf_test, rownames=["Actual"], colnames=["Predicted"]))

acc_clf_test = accuracy_score(test_y, pred_clf_test)
acc_clf_train = accuracy_score(train_y, pred_clf_train)

if acc_clf_test > 0.79:
    pickle.dump(clf, open("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/sgd_clf_model.dat", "wb"))


print("Accuracy RBF SGDClassifier test", acc_clf_test, ";\nAccuracy train ", acc_clf_train)

#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// LOGISTIC REGRESION CON RFE////////////////////")

lr = LogisticRegression()
lr = RFE(lr, 20, step=1)
lr.fit(train_x, train_y)

pred_lr_test = lr.predict(test_x)
pred_lr_train = lr.predict(train_x)

print(pd.crosstab(test_y, pred_lr_test, rownames=["Actual"], colnames=["Predicted"]))

acc_lr_test = accuracy_score(test_y, pred_lr_test)
acc_lr_train = accuracy_score(train_y, pred_lr_train)

if acc_lr_test > 0.79:
    pickle.dump(lr, open("/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/model/lr_rfe_model.dat", "wb"))


print("Accuracy LR con RFE test", acc_lr_test, ";\nAccuracy train ", acc_lr_train)

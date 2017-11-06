import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import svm, neighbors, linear_model, tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# //////////// LEEMOS TODOS LOS ELEMENTOS DESPUES DEL HPS Y FFT /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_st_ceps.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
train_x = df.ix[:,0:ancho - 2].values
train_y = df.ix[:, ancho - 1 ].values

# //////////// LEEMOS TODOS LOS ELEMENTOS DESPUES DEL HPS Y FFT TEST /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_t_st_ceps.csv',
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

print("Accuracy R. Forest test", acc_forest_test, "; Accuracy R. Forest train ", acc_forest_train)
#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// DECISION TREE ////////////////////")

tree_model = tree.DecisionTreeClassifier(max_depth = 15, max_leaf_nodes=5)
tree_model.fit(train_x, train_y)

pred_tree_test = tree_model.predict(test_x)
pred_tree_train = tree_model.predict(train_x)

print (pd.crosstab(test_y, pred_tree_test, rownames=["Actual"], colnames=["Predicted"]))
acc_tree_test = accuracy_score(test_y, pred_tree_test)
acc_tree_train = accuracy_score(train_y, pred_tree_train)

print("Accuracy Tree test", acc_tree_test, "; Accuracy Tree train ", acc_tree_train)

#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// SVN ////////////////////")
svm_model = svm.SVC(gamma=0.001, C=100.)
svm_model.fit(train_x, train_y)

pred_svm_test = svm_model.predict(test_x)
pred_svm_train = svm_model.predict(train_x)

print(pd.crosstab(test_y, pred_svm_test, rownames=["Actual"], colnames=["Predicted"]))
acc_svm_test = accuracy_score(test_y, pred_svm_test)
acc_svm_train = accuracy_score(train_y, pred_svm_train)

print("Accuracy SVM test", acc_svm_test, "; Accuracy SVM train ", acc_svm_train)

#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// LOGISTIC REGRESION ////////////////////")
lr_model = linear_model.LogisticRegression(C=1e2)
lr_model.fit(train_x, train_y)

pred_lr_test = lr_model.predict(test_x)
pred_lr_train = lr_model.predict(train_x)
print(pd.crosstab(test_y, pred_lr_test, rownames=["Actual"], colnames=["Predicted"]))

acc_lr_test = accuracy_score(test_y, pred_lr_test)
acc_lr_train = accuracy_score(train_y, pred_lr_train)

print("Accuracy LR test", acc_lr_test, "; Accuracy LR train ", acc_lr_train)

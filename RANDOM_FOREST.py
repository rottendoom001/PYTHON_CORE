import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
from datetime import datetime
start_time = datetime.now()


# CARGAR DATASET DE DROPBOX
#---------------------------------------------------------------------------------------------
data = pd.read_csv('https://dl.dropboxusercontent.com/u/59930995/dataset/titanic2.csv?dl=1')
clase_name = 'survived' # nombre de variable a predecir
headers    = data.columns.values.tolist()
headers.remove(clase_name)


# TRAIN y TEST
#---------------------------------------------------------------------------------------------
m_train     = np.random.rand(len(data)) < 0.7
data_train  = data.loc[m_train,headers].as_matrix()
data_test   = data.loc[~m_train,headers].as_matrix()
clase_train = data.loc[m_train,clase_name].as_matrix()
clase_test  = data.loc[~m_train,clase_name].as_matrix()


# CONVIERTE EN NUMPY.MATRIX. Para mejor performance
# -----------------------------------------------------------------------------------------------
data_train = np.matrix(data_train)
data_test  = np.matrix(data_test)

print (data_train)
print (data_test)

# MODELO
#---------------------------------------------------------------------------------------------
modelo = RandomForestClassifier(
 random_state      = 1,   # semilla inicial de aleatoriedad del algoritmo
 n_estimators      = 666, # cantidad de arboles a crear
 min_samples_split = 2,   # cantidad minima de observaciones para dividir un nodo
 min_samples_leaf  = 1,   # observaciones minimas que puede tener una hoja del arbol
 n_jobs            = 1    # tareas en paralelo. para todos los cores disponibles usar -1
 )
modelo.fit(X = data_train, y = clase_train)


# PREDICCION
#---------------------------------------------------------------------------------------------
prediccion = modelo.predict(data_test)


# METRICAS
#---------------------------------------------------------------------------------------------
print(metrics.classification_report(y_true=clase_test, y_pred=prediccion))
print(pd.crosstab(clase_test, prediccion, rownames=['REAL'], colnames=['PREDICCION']))


# IMPORTANCIA VARIABLES
#---------------------------------------------------------------------------------------------
var_imp = pd.DataFrame({
 'feature':headers,
 'v_importance':modelo.feature_importances_.tolist()
 })
print var_imp.sort_values(by = 'v_importance', ascending=False)


# END
#---------------------------------------------------------------------------------------------
end_time = datetime.now()
print('duracion: ' + format(end_time - start_time))

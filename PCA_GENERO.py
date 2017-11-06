import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
from sklearn.preprocessing import StandardScaler

def format2CSV(arr, g) :
    s = ''
    for v in arr :
        s = s + str(float(v)) + ", "
    s = s + g + '\n'
    return s

def format2CSV_W(arr) :
    s = ''
    for v in arr :
        s = s + str(float(v)) + ", "
    s = s[0:len(s) - 2 ] + '\n'
    return s

def save2CSV(name, value):
    archivo = open(name, 'a')
    archivo.write(value)
    archivo.close()

def sortEigenValues(eig_pairs):
    j=0
    while (j < len(eig_pairs)):
        i = 0
        for v in eig_pairs :
            if( i < len(eig_pairs) - 1) and (eig_pairs[i+1][0] > v[0]):
                aux = v
                v = eig_pairs[i+1]
                eig_pairs[i+1] = v
            i+=1
        j+=1
    return eig_pairs

df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps.csv',
    header=None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end

col = len(df.columns)
rows = len(df.index)

print ("filas : {} - columnas : {}".format(rows,col))

X = df.ix[:,0:col - 2].values
cl = df.ix[:,col - 1].values
print("/////////// SEXO /////////\n{}".format(cl))
print("/////////// X /////////\n{}".format(X))
# Normalizamos
X_std = StandardScaler().fit_transform(X)
#X_std = X
print("/////////// X STANDARD /////////\n{}".format(X_std))
# Sacamos Matriz de covarianza
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

# La forma rápida
# print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

# Sacamos los eigenvectores y eigenvalores
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#for i in eig_pairs:
#    print("/////////\n",i)
# Sort the (eigenvalue, eigenvector) tuples from high to low
#eig_pairs = sortEigenValues(eig_pairs)

eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
j = 0
for i in eig_pairs:
    print("[{}] - {}".format(j,i[0]))
    j+=1


matrix_w = np.hstack((eig_pairs[0][1].reshape(col -1, 1),
                      eig_pairs[1][1].reshape(col -1, 1),
                      eig_pairs[2][1].reshape(col -1, 1),
                      eig_pairs[3][1].reshape(col -1, 1),
                      eig_pairs[4][1].reshape(col -1, 1),
                      eig_pairs[5][1].reshape(col -1, 1),
                      eig_pairs[6][1].reshape(col -1, 1),
                      eig_pairs[7][1].reshape(col -1, 1),
                      eig_pairs[8][1].reshape(col -1, 1),
                      eig_pairs[9][1].reshape(col -1, 1)
                      ))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)

resultFile ='/Users/alancruz/Desktop/PYTHON/CORE/data/result_pca.csv'
wFile ='/Users/alancruz/Desktop/PYTHON/CORE/data/w_pca.csv'
#/////////////// PERSISTIMOS W ////////////////
for v in matrix_w:
    #Formateamos valor para csv
    csv_tupla = format2CSV_W(v)
    #Persistimos valor para csv
    save2CSV(wFile, csv_tupla)
#/////////////// PERSISTIMOS RESULTADO ////////////////
i = 0
for v in Y:
    gender = cl[i].strip()
    #Formateamos valor para csv
    csv_tupla = format2CSV(v, gender)
    #Persistimos valor para csv
    save2CSV(resultFile, csv_tupla)
    i+=1

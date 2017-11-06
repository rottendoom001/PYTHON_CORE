import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

print (df)

X = df.ix[:,0:4].values
y = df.ix[:,4].values

# Normalizamos
X_std = StandardScaler().fit_transform(X)
print (X_std)

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

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i)


matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)

print ('/////////////// RESULTADO ////////////////')
j = 0
for i in Y:
    print("[{}] = {}".format(j,i))
    j+=1

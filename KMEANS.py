
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#fs, signal = scipy.io.wavfile.read(inputFileName)

float_formatter = lambda x: "%.7f" % x

df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps.csv',
    header=None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end

col = len(df.columns)
rows = len(df.index)

print ("filas : {} - columnas : {}".format(rows,col))

X = df.ix[:,0:col - 2].values

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

np.set_printoptions(formatter={'float_kind':float_formatter})

print(centroids)
print(labels)
np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/kmeans.csv", labels, delimiter=",")

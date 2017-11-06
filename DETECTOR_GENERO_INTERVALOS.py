import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

FRECUENCY = 180
IQR = 70

def printNvalues(n, arr) :
    i = 0
    for v in arr :
        print ("[{}] = {}".format(i,v))
        if n == i :
            break
        i+=1

def getIQR(arr):
    q75, q25 = np.percentile(arr, [75 ,25])
    return q75 - q25

# //////////// LEEMOS TODOS LOS ELEMENTOS DESPUES DEL HPS Y FFT /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
Xtr = df.ix[:,0:ancho - 2].values
Ytr = df.ix[:, ancho - 1 ].values

iqr = []
j = 0
#for j in range(20):
print ("\nFRECUENCIA :",FRECUENCY + j)
accuracy = 0.
hombres = 0
mujeres = 0


for i in range(len(Xtr)):
    med = np.mean(Xtr[i])
    #iqr.append([str(getIQR(Xtr[i])),Ytr[i].strip()])
    #iqr = getIQR(Xtr[i])
    '''
    if med > FRECUENCY :
        r = 'F'
    else :
        if iqr > 30:
            r = 'M'
        else :
            r = 'F'
    '''
    r = 'M' if med <= FRECUENCY + j else 'F'
    if r == Ytr[i].strip():
        if Ytr[i].strip() == 'M':
            hombres+=1
        else :
            mujeres+=1
        accuracy += 1./len(Xtr)
#print ("IQR:",iqr)

#np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/IQR.csv", iqr, delimiter=",", fmt='%s')
print("Done!")
print("Accuracy:", accuracy)
print("correct M :", hombres)
print("correct F :", mujeres)

import numpy as np
import wave
import scipy.io.wavfile
import statistics as st
import math

import sys
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange
from scipy.stats import kurtosis, skew, entropy, gmean
from sklearn.cluster import KMeans


TAINING_M = '/Users/alancruz/Documents/audios/all/entrenamiento_m/'
TAINING_F = '/Users/alancruz/Documents/audios/all/entrenamiento_f/'
FORMAT = '.wav'
TAINING_IN = 4000
TEST_IN = 1000

Q75 = 75
Q25 = 25

MIN_FRQ = 75
MAX_FRQ = 1100

CSV_SEPARATOR = ', '

FRQ_NUM = 10

def printNvalues(n, arr) :
    i = 0
    for v in arr :
        print ("[{}] = {}".format(i,v))
        if n == i :
            break
        i+=1


def get_HPSspectrum_and_cepstrum(y, fs):
    n = len(y)
    j = int (n/2)
    y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), 1.0 / fs)
    freq = freq[range(j)]
    y = y[range(j)]
    y = np.abs(y)
    print (" R =", (10 * math.log10(y[0])))
    #y =[ 20 * math.log10(i) for i in y]
    ceps = 0
    # ceps = np.fft.ifft(np.log(y)).real
    # y = hps(y)
    return y, freq, ceps


def hps(arr):
    r = arr
    d2 = []
    d3 = []
    i = 0
    # Diesmar en 2
    for v in arr :
        if  i % 2 == 0 :
            d2.append(v)
        i+=1
    #Diesmar en 3
    i = 0
    for v in arr :
        if  i % 3 == 0 :
            d3.append(v)
        i+=1
    d2 = np.array(d2)
    d3 = np.array(d3)

    #Multiplicar por d2
    i = 0
    for v in d2 :
        r[i] = r[i] * v
        i+=1
    #Multiplicar por d3
    i = 0
    for v in d3 :
        r[i] = r[i] * v
        i+=1
    return r


def getCentroide(frq) :
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(frq)
    centroids = kmeans.cluster_centers_
    print ("centroids:", centroids)
    return centroids

def getModulationIndex(y, mindom, maxdom, dfrange):
    changes = []
    for j in range(len(y) - 1 ):
        change = abs(y[j] - y[j + 1])
        changes.append(change)
    modindx = 0 if(mindom==maxdom) else np.mean(changes)/dfrange
    return modindx

def core (inputFileName):
    # Frecuencia de muestreo (el doble de la frecuencia máxima audible)
    # Por estandar es 44100
    # fs = 44100.0
    #/////////////// CORE /////////////////
    #spf = wave.open('/Users/alancruz/Documents/audios/la.wav','r')
    fs, signal = scipy.io.wavfile.read(inputFileName)
    print ("FS:", fs)
    # Quitamos la basura y el ruido
    print("NUMERO DE MUESTRAS EN EL TIEMPO : ", signal.size)
    mt = signal.size
    # Deminimos numero de cracteristicas a persistir
    Y, frq, ceps = get_HPSspectrum_and_cepstrum(signal, fs)
    print ("/////////// Y ///////////")
    printNvalues(10, Y)
    print ("/////////// FRQ ///////////")
    printNvalues(10,frq)
    print ("len FRQ:",len(frq))
    mean = np.mean(frq)
    print ("mean:", mean)
    sd = np.std(frq)
    print ("sd:", sd)

inputFileName = TAINING_F + '1' + FORMAT
core (inputFileName)

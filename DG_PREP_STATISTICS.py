import numpy as np
import wave
import scipy.io.wavfile
import statistics as st

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
MAX_FRQ = 500

CSV_SEPARATOR = ', '

FRQ_NUM = 10
MONO = 1

def printNvalues(n, arr) :
    i = 0
    for v in arr :
        print ("[{}] = {}".format(i,v))
        if n == i :
            break
        i+=1

def format2CSV(n, arr, g) :
    s = ''
    i = 1
    for v in arr :
        s = s + str(float(v[1])) + ", "
        if n == i :
            break
        i+=1
    s = s + g + '\n'
    return s

def calculate_spectrum_cepstrum(y, fs):
    # Hay algunos audios que son stereo, se toma un lado
    y = y[:,0] if y.ndim > MONO else y
    n = len(y)
    j = int (n/2)
    y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), 1.0 / fs)
    freq = freq[range(j)]
    y = y[range(j)]
    y = abs(y)
    ceps = np.fft.ifft(np.log(y)).real
    y = hps(y)
    return y, freq, ceps

def plotSpectrum(Y,frq) :
    plot(frq,Y,'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y (Db)|')
    show()

def save2CSV(name, value):
    archivo = open(name, 'a')
    archivo.write(value)
    archivo.close()

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

def clearSignal(signal):
    maxValue = np.amax(signal)
    limit = maxValue/8
    n = len (signal) - 1
    i = 0
    f = 0
    # Buscamos desde donde comenzar
    while i <= n:
        if signal[i] >= limit:
            break
        i+=1
    # Buscamos donde terminar
    while f <= n:
        if signal[n - f] >= limit:
            break
        f+=1
    cleanSignal = signal[i:(n - f)]
    # Tiene que ser un número de muestras par para ser procesada por la FFT
    if len(cleanSignal) %2 != 0 :
        cleanSignal = np.append(cleanSignal, [0])
    return cleanSignal

def getCentroide(frq) :
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(frq)
    centroids = kmeans.cluster_centers_
    print ("centroids:", centroids)
    return centroids

def calculate_modulation_index(y, mindom, maxdom, dfrange):
    changes = []
    for j in range(len(y) - 1 ):
        change = abs(y[j] - y[j + 1])
        changes.append(change)
    modindx = 0 if(mindom==maxdom) else np.mean(changes)/dfrange
    return modindx

def core (inputFileName, resultFile, gender):
    # Frecuencia de muestreo (el doble de la frecuencia máxima audible)
    # Por estandar es 44100
    # fs = 44100.0
    #/////////////// CORE /////////////////
    #spf = wave.open('/Users/alancruz/Documents/audios/la.wav','r')
    fs, signal = scipy.io.wavfile.read(inputFileName)
    # Quitamos la basura y el ruido
    print("NUMERO DE MUESTRAS EN EL TIEMPO : ", signal.size)
    mt = signal.size
    # Deminimos numero de cracteristicas a persistir
    Y, frq, ceps = calculate_spectrum_cepstrum(signal, fs)
    print("NUMERO DE MUESTRAS EN EL LA FRECUENCIA : ", frq.size)
    # Hacemos lista de (decibeles(Y), frecuencia(x)) tuples
    esp_frecuencia_pairs = [(Y[i],frq[i]) for i in range(len(Y))]
    #if elem > MIN_FRQ and elem < MAX_FRQ
    # APLICAMOS FILTRO DE FRECUENCIAS
    esp_frecuencia_pairs = [(Y[i],frq[i]) for i in range(len(Y)) if frq[i] > MIN_FRQ and frq[i] < MAX_FRQ]
    print("NUMERO DE MUESTRAS EN EL LA FRECUENCIA DESPUES DE FLITRO : ", len(esp_frecuencia_pairs))
    ### CARACTERISTICAS #####
    # FRECUENCIAS FILTRADAS
    esp_aux = np.array(esp_frecuencia_pairs)
    frq = esp_aux[:,1]
    # ESPECTRO DE POTENCIA
    Y = esp_aux[:,0]

    # Ordenamos
    esp_frecuencia_pairs.sort()
    esp_frecuencia_pairs.reverse()
    #*********************************************
    print('//////////////////')
    csv_tupla = ''

    mean = np.mean(frq)
    csv_tupla+= str(mean) + CSV_SEPARATOR

    sd = np.std(frq)
    csv_tupla+= str(sd) + CSV_SEPARATOR

    median = np.median(frq)
    csv_tupla+= str(median) + CSV_SEPARATOR

    q75, q25 = np.percentile(frq, [Q75 ,Q25])
    csv_tupla+= str(q25) + CSV_SEPARATOR + str(q75) + CSV_SEPARATOR

    iqr = q75 - q25
    csv_tupla+= str(iqr) + CSV_SEPARATOR

    skw = skew(frq)
    csv_tupla+= str(skw) + CSV_SEPARATOR

    kurt = kurtosis(frq)
    csv_tupla+= str(kurt) + CSV_SEPARATOR

    entr = entropy(frq)
    csv_tupla+= str(entr) + CSV_SEPARATOR

    flatness = gmean(Y)/np.mean(Y)
    csv_tupla+= str(flatness) + CSV_SEPARATOR

    #mode = st.mode(frq)
    #csv_tupla+= str(mode) + CSV_SEPARATOR

    #centroide = getCentroide(frq.reshape(1,-1))
    #csv_tupla+= str(centroide) + CSV_SEPARATOR

    peakf = esp_frecuencia_pairs[0][1]
    csv_tupla+= str(peakf) + CSV_SEPARATOR

    mindom = min (frq)
    csv_tupla+= str(mindom) + CSV_SEPARATOR

    maxdom = max (frq)
    csv_tupla+= str(maxdom) + CSV_SEPARATOR

    dfrange = maxdom - mindom
    csv_tupla+= str(dfrange) + CSV_SEPARATOR

    modindx = calculate_modulation_index(frq, mindom, maxdom, dfrange)
    csv_tupla+= str(modindx) + CSV_SEPARATOR

    # ///////// CON EL CEPSTRUM ////////////
    ceps_mean = np.mean(ceps)
    csv_tupla+= str(ceps_mean) + CSV_SEPARATOR

    ceps_max = max (ceps)
    csv_tupla+= str(ceps_max) + CSV_SEPARATOR

    ceps_min = min (ceps)
    csv_tupla+= str(ceps_min) + CSV_SEPARATOR

    fun_frq = format2CSV(FRQ_NUM, esp_frecuencia_pairs, gender)
    csv_tupla+= ' ' + fun_frq

    #csv_tupla+= gender + '\n'

    #print (csv_tupla)
    #*********************************************
    save2CSV(resultFile, csv_tupla)

resultFile = '/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/data/final_fft_hps_espec_ceps_frq.csv'
# MUESTAS MUJERES
#prefix = '/Users/alancruz/Documents/mujeres/m'

for i in (range(TAINING_IN)) :
    #i = i + current
    print ('/////// F >', i+1)
    inputFileName = TAINING_F + str(i+1) + FORMAT
    core (inputFileName, resultFile, 'F')
# MUESTAS HOMBRE
#prefix = '/Users/alancruz/Documents/hombres/h'

for i in (range(TAINING_IN)) :
    print ('/////// M >', i+1)
    inputFileName = TAINING_M + str(i+1) + FORMAT
    core (inputFileName, resultFile,'M')


resultFile = '/Users/alancruz/Desktop/PYTHON/PYTHON_CORE/data/final_fft_hps_espec_ceps_frq_t.csv'
# MUESTAS MUJERES FUERA DE ENTRENAMIENTO
#prefix = '/Users/alancruz/Documents/mujeres/m'

for i in (range(TEST_IN)) :
    print ('/////// F test >', i)
    index = i + TAINING_IN
    inputFileName = TAINING_F + str(index) + FORMAT
    core (inputFileName, resultFile, 'F')

# MUESTAS HOMBRES FUERA DE ENTRENAMIENTO
#prefix = '/Users/alancruz/Documents/hombres/h'
for i in (range(TEST_IN)) :
    print ('/////// M test>', i)
    index = i + TAINING_IN
    inputFileName = TAINING_M + str(index) + FORMAT
    core (inputFileName, resultFile, 'M')

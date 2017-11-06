import numpy as np
import wave
import scipy.io.wavfile
import sys
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange

TAINING_M = '/Users/alancruz/Documents/audios/all/entrenamiento_m/'
TAINING_F = '/Users/alancruz/Documents/audios/all/entrenamiento_f/'
#TAINING_M = '/Users/alancruz/Documents/hombres/'
#TAINING_F = '/Users/alancruz/Documents/mujeres/'
FORMAT = '.wav'
TAINING_IN = 50
TEST_IN = 0
MONO = 1

MIN_FRQ = 75
MAX_FRQ = 250
def printNvalues(n, arr) :
    i = 0
    for v in arr :
        print ("[{}] = {}".format(i,v))
        if n == i :
            break
        i+=1

def format2CSV(n, arr, g) :
    s = ''
    i = 0
    for v in arr :
        s = s + str(float(v[1])) + ", "
        if n == i :
            break
        i+=1
    s = s + g + '\n'
    return s


def calculateSpectrumWithHPS(y, fs):
    y = y[:,0] if y.ndim > MONO else y
    n = len(y)
    j = int (n/2)
    y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), 1.0 / fs)
    freq = freq[range(j)]
    #print ("Y :", y)
    #print("freq :", freq)
    y = y[range(j)]
    y = abs(y)
    y = hps(y)
    return y, freq

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
    '''
    print("r : ", r.size)
    print("d2 : ", d2.size)
    print("d3 : ", d3.size)
    '''
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



def core (inputFileName, resultFile, gender):
    # Frecuencia de muestreo (el doble de la frecuencia máxima audible)
    # Por estandar es 44100
    # fs = 44100.0
    #/////////////// CORE /////////////////
    #spf = wave.open('/Users/alancruz/Documents/audios/la.wav','r')
    fs, signal = scipy.io.wavfile.read(inputFileName)
    # Quitamos la basura y el ruido
    print("NUMERO DE MUESTRAS EN EL TIEMPO : ", signal.size)
    #signal = clearSignal(signal)
    #signal = signal[:int(fs)]
    # Numero total de muestras
    mt = signal.size
    print("NUMERO DE MUESTRAS EN EL TIEMPO LIMPIA: ", mt)
    # Deminimos numero de cracteristicas a persistir
    C = 20
    if mt >= C:
        Y, frq = calculateSpectrumWithHPS(signal, fs)
        print("NUMERO DE MUESTRAS EN EL LA FRECUENCIA : ", frq.size)
        if frq.size >= C:
            # Hacemos lista de (decibeles(Y), frecuencia(x)) tuples
            esp_frecuencia_pairs = [(Y[i],frq[i]) for i in range(len(Y))]
            #if elem > MIN_FRQ and elem < MAX_FRQ
            # APLICAMOS FILTRO DE FRECUENCIAS
            esp_frecuencia_pairs = [(Y[i],frq[i]) for i in range(len(Y)) if frq[i] > MIN_FRQ and frq[i] < MAX_FRQ]
            print("NUMERO DE MUESTRAS EN EL LA FRECUENCIA DESPUES DE FLITRO : ", len(esp_frecuencia_pairs))
            # Ordenamos
            esp_frecuencia_pairs.sort()
            esp_frecuencia_pairs.reverse()
            # printNvalues(C, esp_frecuencia_pairs)
            print('//////////////////')

            #Formateamos valor para csv
            csv_tupla = format2CSV(C ,esp_frecuencia_pairs, gender)
            #Persistimos valor para csv
            save2CSV(resultFile, csv_tupla)
            #plotSpectrum(Y, frq)
        else :
            print ("NO TIENE LAS SUFICIENTES FRECIENCIAS FUNDAMENTALES POR LO QUE SE OMITIRÁ")

resultFile = '/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_audacity.csv'
# MUESTAS MUJERES
#prefix = '/Users/alancruz/Documents/mujeres/m'

for i in (range(TAINING_IN)) :
    #i = i + current
    print ('/////// F >', i)
    #inputFileName = TAINING_F + "m" + str(i+1) + FORMAT
    inputFileName = TAINING_F + str(i+1) + FORMAT
    core (inputFileName, resultFile, 'F')
# MUESTAS HOMBRE
#prefix = '/Users/alancruz/Documents/hombres/h'

for i in (range(TAINING_IN)) :
    print ('/////// M >', i)
    #inputFileName = TAINING_M + "h" + str(i+1) + FORMAT
    inputFileName = TAINING_M + str(i+1) + FORMAT
    core (inputFileName, resultFile,'M')

resultFile = '/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_t.csv'
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

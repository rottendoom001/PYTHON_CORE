import numpy as np
import wave
import sys
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange

TAINING_M = '/Users/alancruz/Documents/audios/all/entrenamiento_m/'
TAINING_F = '/Users/alancruz/Documents/audios/all/entrenamiento_f/'
FORMAT = '.wav'
TAINING_IN = 50
TEST_IN = 10

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


def calculateSpectrumWithHPS(y,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    j = int (n/2)
    frq = frq[range(j)] # one side frequency range
    Y = fft(y)/n # fft computing and normalization
    Y[0] = 0
    #printNvalues(10,abs(Y))
    Y = Y[range(j)]
    Y = abs(Y)
    Y = hps(Y)
    return Y, frq

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
    limit = maxValue/4
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
    fs = 44100.0
    #/////////////// CORE /////////////////
    #spf = wave.open('/Users/alancruz/Documents/audios/la.wav','r')
    spf = wave.open(inputFileName,'r')
    #signal = np.fromstring(spf.readframes(int(fs)), dtype=np.int16)
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'uint8')

    # Quitamos la basura y el ruido
    signal = clearSignal(signal)
    # Numero total de muestras
    mt = signal.size
    print("NUMERO DE MUESTRAS EN EL TIEMPO : ", mt)

    # Deminimos numero de cracteristicas a persistir
    C = 99
    Y, frq = calculateSpectrumWithHPS(signal, fs)

    # Hacemos lista de (decibeles(Y), frecuencia(x)) tuples
    esp_frecuencia_pairs = [(Y[i],frq[i]) for i in range(len(Y))]
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

resultFile = '/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps.csv'
# MUESTAS MUJERES
#prefix = '/Users/alancruz/Documents/mujeres/m'

for i in (range(TAINING_IN)) :
    print ('/////// F >', i)
    inputFileName = TAINING_F + str(i+1) + FORMAT
    core (inputFileName, resultFile, 'F')
# MUESTAS HOMBRE
#prefix = '/Users/alancruz/Documents/hombres/h'

for i in (range(TAINING_IN)) :
    print ('/////// M >', i)
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

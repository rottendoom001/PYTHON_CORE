import numpy as np
import wave
import sys
import scipy.io.wavfile

from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange

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
        s = s + str(v[1]) + ", "
        if n == i :
            break
        i+=1
    s = s + g + '\n'
    return s


def calculateSpectrumWithHPS(y, fs):
    n = len(y)
    j = int (n/2)
    y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), 1.0 / fs)
    freq = freq[range(j)]
    print ("Y :", y)
    print("freq :", freq)
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

    print("r : ", r.size)
    print("d2 : ", d2.size)
    print("d3 : ", d3.size)

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

#/////////////// CORE /////////////////

#fs, signal = scipy.io.wavfile.read('/Users/alancruz/Documents/audios/all/entrenamiento_f/100.wav')
fs, signal = scipy.io.wavfile.read('/Users/alancruz/Documents/audios/all/entrenamiento_f/17.wav')
print ("FS:", fs)
# Deminimos numero de cracteristicas a persistir
C = 10
Y, frq = calculateSpectrumWithHPS(signal, fs)

# Hacemos lista de (decibeles(Y), frecuencia(x)) tuples
esp_frecuencia_pairs = [(Y[i],frq[i]) for i in range(len(Y))]
# Ordenamos
esp_frecuencia_pairs.sort()
esp_frecuencia_pairs.reverse()

printNvalues(C, esp_frecuencia_pairs)
print('//////////////////')
plotSpectrum(Y, frq)

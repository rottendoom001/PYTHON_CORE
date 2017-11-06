import numpy as np
import wave
import sys
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
spf = wave.open('/Users/alancruz/Documents/audios/all/entrenamiento_f/2.wav','r')

print ("fs :", spf.getframerate())
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'uint8')
#signal = signal[:44100]
# Numero total de muestras
mt = signal.size
print("NUMERO DE MUESTRAS EN EL TIEMPO : ", mt)
# Frecuencia de muestreo (el doble de la frecuencia máxima audible)
# Por estandar es 44100
fs = 44100

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

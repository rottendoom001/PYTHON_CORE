import numpy as np
import scipy.io.wavfile
import base64

def save2CSV(name, value):
    archivo = open(name, 'a')
    archivo.write(value)
    archivo.close()

resultFile = '/Users/alancruz/Desktop/PYTHON/CORE/data/wav_base64.txt'
fs, signal = scipy.io.wavfile.read('/Users/alancruz/Documents/audios/all/entrenamiento_f/17.wav')
encoded = base64.b64encode(signal)
print("-->",encoded)
save2CSV(resultFile,str(encoded))

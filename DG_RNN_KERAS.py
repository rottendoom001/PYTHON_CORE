import pandas as pd
import numpy as np
import tensorflow as tf

STANDARD_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 75
BATCH_SIZE = 128

h1 = 300   # Numero de neuronas en la capa oculta
h2 = 300

# //////////// LEEMOS TODOS LOS ELEMENTOS DESPUES DEL HPS Y FFT /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
width_tr = len(df.columns)
train_x = df.ix[:,0:width_tr - 2].values
train_y = df.ix[:, width_tr - 1 ].values

# //////////// LEEMOS TODOS LOS ELEMENTOS DESPUES DEL HPS Y FFT TEST /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_t.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
width_t = len(df.columns)
test_x = df.ix[:,0:width_t - 2].values
test_y = df.ix[:, width_t - 1 ].values
#////////////////////////////////////////////////////////////////////////////////////
print("///////////////// MULTILAYER PERCEPTRON ////////////////////")

model = Sequential()
model.add(Dense(BATCH_SIZE, input_dim=width_tr, init="uniform", activation="relu"))
model.add(Dense(, init="uniform", activation="relu"))

import numpy as np
import os
import shutil
import sys

# Example /Users/alancruz/Documents/audios/all/originales/audio_1/wav

TAINING_M = '/Users/alancruz/Documents/audios/all/entrenamiento_m/'
TAINING_F = '/Users/alancruz/Documents/audios/all/entrenamiento_f/'

M_LIMIT = 5500
F_LIMIT = 5500

def ls(ruta = os.getcwd()):
    return [arch.name for arch in os.scandir(ruta) if arch.is_file()]

def deleteDirs():
    if os.path.exists(TAINING_M):
        shutil.rmtree(TAINING_M, ignore_errors=False)
        print('Árbol de directorio borrado', TAINING_M)
    else :
        print('Árbol de directorio no existe', TAINING_M)
    if os.path.exists(TAINING_F):
        shutil.rmtree(TAINING_F, ignore_errors=False)
        print('Árbol de directorio borrado', TAINING_F)
    else :
        print('Árbol de directorio no existe', TAINING_F)

def createDirs():
    if not os.path.exists(TAINING_M):
        os.makedirs(TAINING_M)
        print('Directorio creado', TAINING_M)
    if not os.path.exists(TAINING_F):
        os.makedirs(TAINING_F)
        print('Directorio creado', TAINING_F)

def getSexType (route):
    archivo = open(route, "r")
    archivo.readline()
    archivo.readline()
    archivo.readline()
    archivo.readline()
    # en la linea 4 viene
    # Gender: Female
    gender = archivo.readline()
    sex_arr = gender.split(':')
    archivo.close()
    sex = sex_arr[1]
    return sex[1].strip()




deleteDirs()
createDirs()

count_f = 0;
wav_route_f = '/Users/alancruz/Documents/audios/all/entrenamiento_f_limpios'
wav_files_f = ls (wav_route_f)

for f in wav_files_f:
    print ("renombrando archivo",f, "en mujeres con ",count_f,".wav")
    shutil.copyfile(wav_route_f + '/' + f, TAINING_F + f)
    os.rename(TAINING_F + f, TAINING_F + str(count_f) + '_c.wav')
    count_f+=1


count_m = 0;
wav_route_m = '/Users/alancruz/Documents/audios/all/entrenamiento_m_limpios'
wav_files_m = ls (wav_route_m)

for f in wav_files_m:
    print ("renombrando archivo",f, "en hombres con ",count_m,".wav")
    shutil.copyfile(wav_route_m + '/' + f, TAINING_M + f)
    os.rename(TAINING_M + f, TAINING_M + str(count_m) + '_c.wav')
    count_m+=1

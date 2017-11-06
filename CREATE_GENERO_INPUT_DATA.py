import numpy as np
import os
import shutil
import sys

# Example /Users/alancruz/Documents/audios/all/originales/audio_1/wav
INPUT_WAV_PREFIX = '/Users/alancruz/Documents/audios/all/originales/'
SEX_TYPE_FILE = 'etc/README'
DIRECTORY_PREFIX = 'audio_'
WAV = 'wav/'
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


f_count = 0;
m_count = 0;
count = 0;
m_end = False
f_end = False

deleteDirs()
createDirs()
while True:
    if m_end == True and f_end == True :
        break
    try:
        wav_route = INPUT_WAV_PREFIX + DIRECTORY_PREFIX + str(count) +'/'+ WAV
        sex = getSexType(INPUT_WAV_PREFIX + DIRECTORY_PREFIX + str(count) +'/'+ SEX_TYPE_FILE)
        wav_files = ls (wav_route)

        for f in wav_files:
            if sex == 'F':
                if f_count < F_LIMIT:
                    print ("copiando archivo",f,"en mujeres")
                    shutil.copyfile(wav_route + '/' + f, TAINING_F + f)
                    os.rename(TAINING_F + f, TAINING_F + str(f_count) + '.wav')
                    f_count+=1
                else :
                    #print("//////////",f_count)
                    f_end = True
            else :
                if m_count < M_LIMIT:
                    print ("copiando archivo",f,"en Hombres")
                    shutil.copyfile(wav_route + '/' + f, TAINING_M + f)
                    os.rename(TAINING_M + f, TAINING_M + str(m_count) + '.wav')
                    m_count+=1
                else :
                    #print("//////////",m_count)
                    m_end = True
        count+=1

    except OSError as err:
        print("Error OS: {0}".format(err))
        count+=1
        continue
    except :
        print("Error Inesperado: {0}".format(sys.exc_info()[0]))
        count+=1
        continue


'''
while True:
    sex = getSexType(INPUT_WAV_PREFIX + DIRECTORY_PREFIX + count +'/'+ SEX_TYPE_FILE)
'''

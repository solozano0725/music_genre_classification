from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import numpy as np
import glob
import librosa

#Initialization of variables
songsFolder = r"C:\Users\Sol.Lozano\Documents\music_genre_classification\genres"
songsFolderTarget = r"C:\Users\Sol.Lozano\Documents\music_genre_classification\target\dont_speak-no_doubt.wav"
fileName = "my.dat"
target = []


def loadSongsFromFolders(songsFolder, folder):
    r = []
    f = f"{songsFolder}\\{folder}"
    for filename in glob.glob(os.path.join(f, '*.wav')):
        (rate , data) = wav.read(filename)
        data = np.array(data)
        r.append(data)
    r = np.array(r)
    return r

def loadSongsFolders():
    i=0
    dic = {}
    res = []
    obj = os.scandir(songsFolder) 
    for entry in obj : 
        if entry.is_dir(): 
            res = loadSongsFromFolders(songsFolder, entry.name)
            dic[entry.name] = res
            i+=1
            target.append(i)
    return dic, target


def loadSongWithLibrosa(songsFolder, folder):
    f = f"{songsFolder}\\{folder}"
    listOfList = []
    for filename in glob.glob(os.path.join(f, '*.wav')):
        y, sr = librosa.load(str(filename))
        listOfList.append((y, sr))
    return listOfList

def loadSongsLibrosa():
    i=0
    dic = {}
    res = []
    obj = os.scandir(songsFolder) 
    for entry in obj : 
        if entry.is_dir(): 
            res = loadSongWithLibrosa(songsFolder, entry.name)
            dic[entry.name]  = res
            i+=1
            target.append(i)
    return dic, target

def validationKeyDic(dict, name):
    return True if (name in dict) else False

def insertationKeyDic(dict, name, d):
    if validationKeyDic(dict, name) is True:
        dict[name].append(d)
    else: 
        dict[name] = d
    return dict

def printDic(dict):
    for key in dict:
        print(key, '->', dict[key], 'len:', len(dict[key]))

def showSpectrums(title, data):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(data, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def showTimevsFrequency(title, data):
    plt.plot(data)
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.title(title)
    plt.show()
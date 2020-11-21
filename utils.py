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

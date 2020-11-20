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
    dic = {}
    f = f"{songsFolder}\\{folder}"
    for filename in glob.glob(os.path.join(f, '*.wav')):
        y, sr = librosa.load(str(filename))
        print(str(filename))
        print(y)
        print(sr)
        dic = (insertationKeyDic(dic, f'{folder}-y', y))
        dic = (insertationKeyDic(dic, f'{folder}-sr', sr))
        #if validateKeyDic(dic, folder) is True:
        #    dic[f'{folder}-y'].append(y)
        #    dic[f'{folder}-sr'].append(sr)
        #else:
        #    dic[folder+'-y'] = y
        #    dic[folder+'-sr'] = sr
    return dic

def validationKeyDic(dict, name):
    return lambda x : True if (name in dict) else False

def insertationKeyDic(dict, name, d):
    if validationKeyDic(dict, name) is True:
        dict[name].append(d)
    else: 
        dict[name] = d
    return dict

def loadSongsLibrosa():
    i=0
    dic = {}
    res = {}
    obj = os.scandir(songsFolder) 
    for entry in obj : 
        if entry.is_dir(): 
            res = loadSongWithLibrosa(songsFolder, entry.name)
            dic =(insertationKeyDic(dic, entry.name, res))
            #if validateKeyDic(dic, entry.name) is True:
            #    dic[entry.name].append(res)
            #else: 
            #    dic[entry.name]=res 
            i+=1
            target.append(i)
    return dic, target

print(loadSongsLibrosa())
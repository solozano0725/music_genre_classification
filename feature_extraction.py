import librosa
import librosa.display
import utils as utils
import numpy as np
import matplotlib.pyplot as plt

#for key in dic:
#    print(key, '->', dic[key], 'len:', len(dic[key]))


def calculationEnergySpectrums(dict_songs):
    dic = {}
    for key in dict_songs:
        r=[]
        for lol in range(0,len(dict_songs[key])):
            y = dict_songs[key][lol][0]
            sr = dict_songs[key][lol][1]
            energy_spectrum = librosa.feature.melspectrogram(S = np.abs(librosa.stft(y))**2, sr = sr)
            r.append(energy_spectrum)
        dic[key] = r 
    return dic

def calculationMFCC(dict_songs_enegyspec):
    dic = {}
    for key in dict_songs_enegyspec:
        r=[]
        for l in range(0,len(dict_songs_enegyspec[key])):
            y = dict_songs_enegyspec[key][l]
            mcff = np.mean(librosa.feature.mfcc(np.array(y).flatten() , 44100 , n_mfcc = 13) , axis = 1)
            r.append(mcff)
        dic[key] = r 
    return dic

def calculationMFCCDeltas(dict_songs_MFCC):
    dic = {}
    for key in dict_songs_MFCC:
        r=[]
        for l in range(0,len(dict_songs_MFCC[key])):
            y = dict_songs_MFCC[key][l]
            delta_mcff_1 = librosa.feature.delta(y)
            delta_mcff_2 = librosa.feature.delta(y , order = 2)
            r.append((delta_mcff_1, delta_mcff_2))
        dic[key] = r 
    return dic


songs, target = utils.loadSongsLibrosa()
dict_songs_enegyspec = calculationEnergySpectrums(songs)
#utils.showSpectrums('Mel spectrogram of a rock song', dict_songs_enegyspec['rock'][0])
dict_songs_MFCC = calculationMFCC(dict_songs_enegyspec)
dict_songs_MFCC_deltas = calculationMFCCDeltas(dict_songs_MFCC)
#utils.showSpectrums('Time vs Frequency of a rock song', dict_songs_MFCC['rock'][0])
utils.printDic(dict_songs_MFCC_deltas)


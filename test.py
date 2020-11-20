
from near_neighnbors import NearestNeighbors
from training_model import TrainingModel
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import numpy as np

import glob
import librosa
import librosa.display

#Initialization of variables
songsFolder = r"C:\Users\Sol.Lozano\Documents\music_genre_classification\genres\blues"
songsFolderTarget = r"C:\Users\Sol.Lozano\Documents\music_genre_classification\target\dont_speak-no_doubt.wav"
fileName = "my.dat"
target = []

for filename in glob.glob(os.path.join(songsFolder, '*.wav')):
    y, sr = librosa.load(str(filename))
    print(y,sr)
    print(describe(y))
    d = []
    d = librosa.feature.mfcc(y=y , sr=sr , n_mfcc = 20)
    print(d)
    print(d.shape)


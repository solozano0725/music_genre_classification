import librosa
import librosa.display
import utils as utils
import numpy as np

songs, target = utils.loadSongs()

#for key in songs:
#    print(key, '->', , 'len:', len(songs[key]))
r=[]
#r = {}
#for key in songs:
for i in range(100):
    print(songs['rock'][i])
    print(type(songs['rock'][i]))
    audio = songs['rock'][i].astype(np.float32, order='C') / 32768.0
    print(audio)
    print(type(audio))
    y, sr = librosa.load(audio)
    print(y,sr)
    r.append(librosa.feature.melspectrogram(S = np.abs(librosa.stft(songs['rock'][i]))**2) )
#numpy.ndarray
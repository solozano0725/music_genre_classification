import librosa
import librosa.display
import utils as utils
import numpy as np
import matplotlib.pyplot as plt

songs, target = utils.loadSongsLibrosa()

#for key in dic:
#    print(key, '->', dic[key], 'len:', len(dic[key]))

dic = {}

for key in songs:
    r=[]
    for lol in range(0,len(songs[key])):
        y = songs[key][lol][0]
        sr = songs[key][lol][1]
        #print(array)
        energy_spectrum = librosa.feature.melspectrogram(S = np.abs(librosa.stft(y))**2)
        r.append(energy_spectrum)
    dic[key] = r 


plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(dic['blues'][0], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
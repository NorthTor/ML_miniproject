import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#-------- Import audio file 
audio_file = "../Database_200/Airport_reduced/airport-barcelona-0-6-s5.wav"

signal, sampleRate = librosa.load(audio_file)

#-------- Extract MFCCs

mfccs = librosa.feature.mfcc(signal, n_mfcc=12, sr=sampleRate)
# print(mfccs.shape)

#-------- Calculate MFCCs 1st and 2nd orders derivative

delta_mfcc = librosa.feature.delta(mfccs)
delta2_mfcc = librosa.feature.delta(mfccs, order=2)


# ------- Visualize MFCCs

plt.figure(figsize=(25,10))
librosa.display.specshow(mfccs,
                         x_axis='time',
                         sr=sampleRate)

plt.xlabel("Time")
plt.ylabel("MFC Coefficients")
#plt.colorbar(format="%+2f")
plt.show()
#
### 1st order derivative
# plt.figure(figsize=(25,10))
# librosa.display.specshow(delta_mfcc,
#                          x_axis='time',
#                          sr=sampleRate)
# plt.colorbar(format="%+2f")
# plt.show()
#
### 2nd order derivative
# plt.figure(figsize=(25,10))
# librosa.display.specshow(delta2_mfcc,
#                          x_axis='time',
#                          sr=sampleRate)
# plt.colorbar(format="%+2f")
# plt.show()

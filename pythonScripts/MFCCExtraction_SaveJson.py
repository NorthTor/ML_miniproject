import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import math
import random, os
import shutil 

# Extract and save MFCCs from audiofiles 

def save_mfcc(dataset_path, json_path, n_mfcc=20, n_fft=2048, hop_length=1024, num_segments=1):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    samples_per_segment = int(SAMPLES_PER_FILE / num_segments)
    expected_num_mfcc_vectors = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames ,filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("\\")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            for file in filenames:
                file_path = os.path.join(dirpath, file)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for j in range(num_segments):
                    start_sample = samples_per_segment*j
                    finish_sample = start_sample + samples_per_segment
                    mfccs = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mfcc=n_mfcc, sr=SAMPLE_RATE)
                    mfccs = mfccs.T
                    #print(type(mfccs), np.shape(mfccs))
                    if len(mfccs) == expected_num_mfcc_vectors:
                        data["mfcc"].append(mfccs.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, j+1))

    with open(JSON_PATH, 'w') as fp:
        json.dump(data, fp, indent=4)

    return 0


# for mfcc
DATASET_PATH = "../Database_500"            
JSON_PATH = "../Features/mfcc_500.json"


SAMPLE_RATE = 44100  # Sample rate of the audio signals (frequency)
DURATION = 10        # Duration of each audio data (seconds)

SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
 
# -------------------------------------------------
save_mfcc(DATASET_PATH, JSON_PATH)



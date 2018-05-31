import os
import wave
from random import shuffle

import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, delta, lifter
import librosa

_path = 'training_data/chunks/'
CHUNK = 4096

def speaker(filename):
    # if not "_" in file:
    #   return "Unknown"
    return filename.split("_")[1]

def one_hot_to_item(hot, items):
    i=np.argmax(hot)
    item=items[i]
    return item

def one_hot_from_item(item, items):
    # items=set(items) # assure uniqueness
    x = np.zeros(len(items))
    i = items.index(item)
    x[i] = 1
    return x


def generate_wav_file_chunks(name):
    f = wave.open(name, "rb")
    # print("loading %s"%name)
    chunk = []
    FRAME_COUNT = CHUNK * 2
    data0 = f.readframes(FRAME_COUNT)
    while data0:  # f.getnframes()
        # data=np.fromstring(data0, dtype='float32')
        data = np.fromstring(data0, dtype='uint16')
        # data = np.fromstring(data0, dtype='uint8')
        data = (data + 128) / 255.  # 0-1 for Better convergence

        for i in range(0, len(data), FRAME_COUNT):
            chunk.extend(data[i:i + FRAME_COUNT])

            zero_padding = np.zeros(np.max((FRAME_COUNT - len(chunk), 0)))
            # fill with padding 0's
            chunk.extend(zero_padding)

            yield chunk
            chunk = []
        data0 = f.readframes(FRAME_COUNT)

def load_norm_trunc_wav_file(name):
    f = wave.open(name, "rb")
    # print("loading %s"%name)
    chunk = []
    data0 = f.readframes(CHUNK)
    while data0:  # f.getnframes()
        # data=np.fromstring(data0, dtype='float32')
        # data = np.fromstring(data0, dtype='uint16')
        data = np.fromstring(data0, dtype='uint8')
        data = (data + 128) / 255.  # 0-1 for Better convergence
        # chunks.append(data)
        chunk.extend(data)
        data0 = f.readframes(CHUNK)
    # finally trim:
    chunk = chunk[0:CHUNK * 2]  # should be enough for now -> cut
    chunk.extend(np.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
    # print("%s loaded"%name)
    return chunk

def mfcc_batch_generator_2(speakers, batch_size=10, utterance_len=48, path = _path):
    if not speakers:
        raise Exception('No speaker labels provided')

    batch_features = []
    labels = []
    files = os.listdir(path)
    count = 0
    while count < len(files):
        print("loaded batch of %d files" % len(files))
        shuffle(files)
        for file in files:
            count += 1
            if not file.endswith(".wav"):
                continue
            fs, signal = wavfile.read(path+file)
            mfcc_feats = mfcc(
                signal,
                fs,
                winlen=0.032,
                winstep=0.01,
                numcep=20,
                nfilt=26,
                nfft=2048,
                lowfreq=300,
                highfreq=4000,
                appendEnergy=True,
                winfunc=np.hamming,
            )

            name = file.split('.wav')[0]
            label = one_hot_from_item(speaker(name), speakers)

            for i in range(0, len(mfcc_feats), utterance_len):
                utt = mfcc_feats[i:i + utterance_len]
                zero_padding = np.zeros((np.max((utterance_len -
                    len(utt), 0)), 20))
                utt = np.concatenate((utt, zero_padding))
                data = np.zeros((20, utterance_len))

                for i, block in enumerate(utt):
                    for y, coeff in enumerate(block):
                        data[y][i] = coeff

                batch_features.append(data)
                labels.append(label)

                if len(batch_features) >= batch_size:
                    combined = list(zip(batch_features, labels))
                    shuffle(combined)
                    batch_features[:], labels[:] = zip(*combined)

                    yield batch_features, labels
                    batch_features = []
                    labels = []

# def calculate_delta(array):
#     """Calculate and returns the delta of given feature vector matrix"""

#     rows,cols = array.shape
#     deltas = np.zeros((rows,cols))
#     N = 2
#     for i in range(rows):
#         index = []
#         j = 1
#         while j <= N:
#             if i-j < 0:
#                 first = 0
#             else:
#                 first = i-j
#             if i+j > rows -1:
#                 second = rows -1
#             else:
#                 second = i+j
#             index.append((second,first))
#             j+=1
#         deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
#     return deltas

def calculate_delta(array):
    numerator_sum = 0
    denominator_sum = 0
    N = 2
    deltas = []
    for i, val in enumerate(array):
        for n in range(1, 1 + N):
            first_idx = i + n if i + n < len(array) else len(array) - 1
            second_idx = i - n if i - n >= 0 else 0
            numerator_sum += n*(array[first_idx] - array[second_idx])
            denominator_sum += n**2
        deltas.append(numerator_sum / (2*denominator_sum))

    return np.array(deltas)


def mfcc_batch_generator(speakers, batch_size=10, utterance_len=48, path = _path):
    if not speakers:
        raise Exception('No speaker labels provided')
    batch_features = []
    labels = []
    files = os.listdir(path)
    count = 0

    while count < len(files):
        print("loaded batch of %d files" % len(files))
        shuffle(files)
        for file in files:
            count += 1
            if not file.endswith(".wav"):
                continue
            fs, signal = wavfile.read(path+file)
            num_ceps = 20
            mfcc_feats = mfcc(
                signal,
                fs,
                winlen=0.032,
                winstep=0.01,
                numcep=num_ceps,
                nfilt=26,
                nfft=2048,
                lowfreq=300,
                highfreq=3000,
                appendEnergy=True,
                winfunc=np.hamming,
            )

            num_delta = 2
            use_library_delta = True

            delta1_real = delta(mfcc_feats, 2)
            delta2_real = delta(delta1_real, 2)

            result = []
            for i, sample in enumerate(mfcc_feats):
                if use_library_delta:
                    if num_delta == 1:
                        new_sample = np.concatenate((sample, delta1_real[i]))
                        result.append(new_sample)
                    elif num_delta == 2:
                        new_sample = np.concatenate((sample, delta1_real[i], delta2_real[i]))
                        result.append(new_sample)
                    else:
                        result.append(sample)
                else:
                    if num_delta == 1:
                        delta1 = calculate_delta(sample)
                        new_sample = np.concatenate((sample, delta1))
                        result.append(new_sample)
                    elif num_delta == 2:
                        delta1 = calculate_delta(sample)
                        delta2 = calculate_delta(delta1)
                        new_sample = np.concatenate((sample, delta1, delta2))
                        result.append(new_sample)
                    else:
                        result.append(sample)

            mfcc_feats = np.array(result)

            name = file.split('.wav')[0]
            label = one_hot_from_item(speaker(name), speakers)

            for i in range(0, len(mfcc_feats), utterance_len):
                utt = mfcc_feats[i:i + utterance_len]
                utt = np.pad(utt, ((0, 0), (0, num_ceps*(1+num_delta)-len(mfcc_feats[0]))),
                                mode='constant', constant_values=0)
                zero_padding = np.zeros((np.max((utterance_len - len(utt), 0)), num_ceps*(1+num_delta)))
                utt = np.concatenate((utt, zero_padding))
                batch_features.append(utt)
                labels.append(label)

                # wave, sr = librosa.load(path+file, mono=True)
                # mfcc_feats = librosa.feature.mfcc(wave, sr, n_mfcc=20)
                # print(np.array(mfcc).shape)
                # mfcc_feats = np.pad(mfcc_feats, ((0, 0), (0, 81-len(mfcc_feats[0]))),
                #                     mode='constant', constant_values=0)
                # batch_features.append(np.array(mfcc_feats))
                # batch_features.extend(mfcc_feats)
                if len(batch_features) >= batch_size:
                    # excess = batch_features[batch_size:]
                    # label_excess = labels[batch_size:]
                    # batch_features = batch_features[0:batch_size]
                    # labels = labels[0:batch_size]
                    # if target == Target.word:  labels = sparse_labels(labels)
                    # labels=np.array(labels)
                    # print(np.array(batch_features).shape)
                    # yield np.array(batch_features), labels
                    # print(np.array(labels).shape) # why (64,) instead of (64, 15, 32)? OK IFF dim_1==const (20)

                    combined = list(zip(batch_features, labels))
                    shuffle(combined)
                    batch_features[:], labels[:] = zip(*combined)

                    yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                    # batch_features = excess or []  # Reset for next batch
                    # labels = label_excess or []
                    batch_features = []  # Reset for next batch
                    labels = []


# If you set dynamic_pad=True when calling tf.train.batch the returned batch will be automatically padded with 0s. Handy! A lower-level option is to use tf.PaddingFIFOQueue.
# only apply to a subset of all images at one time
def wave_batch_generator(speakers, batch_size=10, path = _path): #speaker
    if not speakers:
        raise Exception('No speaker labels provided')
    batch_waves = []
    labels = []
    # input_width=CHUNK*6 # wow, big!!
    files = os.listdir(path)
    count = 0
    while count < len(files):
        # print("loaded batch of %d files" % len(files))
        shuffle(files)
        # TODO pull in all files and split into segment files
        for wav in files:
            count += 1
            if not wav.endswith(".wav"):
                continue
            name = wav.split('.wav')[0]

            for chunk in generate_wav_file_chunks(path+wav):
                labels.append(one_hot_from_item(speaker(name), speakers))
                batch_waves.append(chunk)
                # batch_waves.append(chunks[input_width])
                if len(batch_waves) >= batch_size:
                    yield batch_waves, labels
                    batch_waves = []  # Reset for next batch
                    labels = []

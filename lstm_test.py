import tensorflow as tf
import tflearn

from utils import one_hot_to_item, mfcc_batch_generator, mfcc_batch_generator_2

learning_rate = 0.0001
batch_size = 1

width = 60  # mfcc features
height = 80  # (max) length of utterance
speakers = ['adrian', 'zhanet']
number_classes=len(speakers)

path = 'testing_data/chunks/'

# Network building
# net = tflearn.input_data([None, width, height])
net = tflearn.input_data([None, height, width])
net = tflearn.lstm(net, 2400, dropout=0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=3)

# NORMAL MFCC NETS


# model.load('models/delta_2_lstm.model_20_80_2400_200e_freq_300_3000') # 75%
# model.load('models/t2_delta_2_lstm.model_20_80_2400_200e_freq_300_3000') # 93.75%


# model.load('models/lstm.model_80_800_0.5_0.001_200e') # 93.75%
# model.load('models/t2_lstm.model_80_800') # 93.75%
# model.load('models/lstm.model_150_800_200es') # 90%
# model.load('models/lstm.model_150_800') # 90%
# model.load('models/lstm.model_80_800_0.5_0.00001_200e') # 87.5%
# model.load('models/lstm.model_13_80_800_200e_freq_range_300_3000') # 87.5%
# model.load('models/delta_2_lstm.model_20_80_800_200e') # 87.5%
# model.load('models/lstm.model_80_800_0.5_0.001') # 87.5%
# model.load('models/lstm.model_80_800_0.4') # 87.5%
# model.load('models/lstm.model_80_800_0.3') # 87.5
# model.load('models/t2_lstm.model_48_128_4') # 81.81%
# model.load('models/t2_delta_lstm.model_20_80_800_200e') # 81.25%
# model.load('models/delta_2_lstm.model_13_80_800_200e') # 81.25%
# model.load('models/lstm.model_80_800_0.5') # 81.25
# model.load('models/lstm.model_80_800_0.6') # 81.25%
# model.load('models/lstm.model_80_800_0.7') # 81.25%
# model.load('models/lstm.model_80_1600') # 81.25%
# model.load('models/lstm.model_20_150_800_200e_freq_range_100_3000') # 80%
# model.load('models/lstm.model_20_80_800_200e_freq_range_300_3000') # 80%
# model.load('models/lstm.model_80_800_0.5_0.00001') # 75%
# model.load('models/delta_lstm.model_13_80_800_200e') # 75%
# model.load('models/t2_delta_2_lstm.model_13_80_800_200e') # 75%
# model.load('models/t2_delta_lstm.model_13_80_800_200e') # 75%
# model.load('models/lstm.model_13_150_800_200e') # 70% freq range 100-2000
# model.load('models/lstm.model_20_150_800_200e_freq_range_300_3000') # 70%
# model.load('models/lstm.model_80_800_0.5_0.0001_200e') # 68.75%
# model.load('models/t2_delta_2_lstm.model_20_80_800_200e') # 68.75%
# model.load('models/delta_lstm.model_20_80_800_200e') # 68.75%
# model.load('models/t2_lstm.model_13_80_800_200e_freq_range_300_5000') # 68.75%
# model.load('models/t2_lstm.model_48_128') # 63.63%
# model.load('models/lstm.model_80_800_0.1') # 62.5%
# model.load('models/lstm.model_13_80_800_200e_freq_range_300_5000') # 62.5%
# model.load('models/lstm.model_20_150_800_200e') # 60% # freq range 100-2000
# model.load('models/lstm.model_80_800_0.5_0.01_200e') # 56.25%
# model.load('models/lstm.model_80_800_0.5_0.01') # 50%

# model.load('models/tflearn.lstm.model_48_128_4') # 86.32%
# model.load('models/tflearn.lstm.model_48_128') # 81.79%
# model.load('models/tflearn.lstm.model_80_800') # 81.18%

# ZIPPED MFCC NETS

# model.load('models/t2_lstm.model_2_80_128_3') # 31.25
# model.load('models/t2_lstm.model_2_48_128') # 27.27%
# model.load('models/t2_lstm.model_2_48_128_4') # 36.36

# model.load('models/tflearn.lstm.model_2_48_128_4') # 49.95%
# model.load('models/tflearn.lstm.model_2_48_128') # 31.81%
# model.load('models/tflearn.lstm.model_2_80_128_3') # 50.04%


batch = mfcc_batch_generator(speakers, batch_size = batch_size, utterance_len = height, path = path)


count = 0
correct = 0

for feats, labels in batch:
    _y = model.predict(feats)
    for i, val in enumerate(_y):
        real_label = one_hot_to_item(labels[i], speakers)
        label = one_hot_to_item(val, speakers)

        if real_label == label:
            correct += 1
        count += 1

    print('% correct: {}%'.format(correct / count * 100))

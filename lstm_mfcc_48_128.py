import tensorflow as tf
import tflearn

import utils

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 50

width = 20  # mfcc features
height = 48  # (max) length of utterance
# classes = 10  # digits
speakers = ['adrian', 'zhanet']
number_classes=len(speakers)

batch = utils.mfcc_batch_generator(speakers, batch_size = batch_size, utterance_len = height)

# Network building
# net = tflearn.input_data([None, width, height])
net = tflearn.input_data([None, height, width])
net = tflearn.lstm(net, 128, dropout=0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=3)

# model.load("models/tflearn.lstm.model_48_128")

## add this "fix" for tensorflow version errors
for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): tf.add_to_collection(tf.GraphKeys.VARIABLES, x )

# Training

while training_iters > 0:
    trainX, trainY = next(batch)
    # testX, testY = next(batch)  # todo: proper ;)
    # if not trainX or not testX:
    #     break

    # model.fit(trainX, trainY, n_epoch=50, validation_set=(testX, testY),
    #           show_metric=True, batch_size=batch_size)
    model.fit(trainX, trainY, n_epoch=50, show_metric=True, batch_size=batch_size)
    training_iters -= 1

    model.save("models/t2_lstm.model_48_128")
model.save("models/t2_lstm.model_48_128")
_y = model.predict(next(batch)[0])  # << add your own voice here
print(_y)

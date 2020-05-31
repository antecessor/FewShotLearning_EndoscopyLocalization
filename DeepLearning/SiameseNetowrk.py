from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random

from keras.datasets import mnist
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, BatchNormalization, Concatenate, concatenate, AveragePooling2D, AveragePooling1D
from keras.optimizers import RMSprop
from keras import backend as K
from tensorflow_core.python.keras.callbacks import Callback


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.maximum(sum_square, K.epsilon())


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + K.abs(1 - y_true) * margin_square)


def create_pairs(x, digit_indices, num_classes):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    filters = [16, 32, 16]
    input = Input(shape=input_shape)
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input)
    x = AveragePooling2D()(x)
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    # x2 = Flatten()(input)
    # x2 = Dense(16, activation='relu')(x2)

    # x = concatenate([x, x2])
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < .5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] >= self.threshold:
            self.model.stop_training = True


# the data, split between train and test sets
def trainSiamese(x_train, y_train, x_test, y_test):
    # x_train = x_train / 255
    # x_test = x_test / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    epochs = 15
    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, num_classes)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, num_classes)

    # network definition
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    distance = BatchNormalization()(distance)
    model = None
    te_acc = 0
    while te_acc < 0.65:
        model = Model([input_a, input_b], distance)

        # train

        model.compile(loss=contrastive_loss, optimizer=RMSprop(), metrics=[accuracy])
        callback = MyThresholdCallback(threshold=0.75)

        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=16,
                  epochs=epochs,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y), verbose=2, callbacks=[callback])

        # compute final accuracy on training and test sets
        y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        tr_acc = compute_accuracy(tr_y, y_pred)
        y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = compute_accuracy(te_y, y_pred)

        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    return model


def testSiamese(model, xGroups, yGroups, imageTests):
    bestClass = []
    bestDist = []
    for index, imageTest in enumerate(imageTests):
        print("frame {} processed".format(index))
        imageTest = np.expand_dims(imageTest, 0)
        imageRepeated = np.repeat(imageTest, xGroups.shape[0], axis=0)
        dist = model.predict([xGroups, imageRepeated])
        dist = np.abs(dist.ravel())
        indsort = np.argsort(dist)
        selectedGroup = yGroups[indsort]
        selectedDist = dist[indsort]
        bestClass.append(selectedGroup[range(5)])
        bestDist.append(selectedDist[range(5)])
    bestDist = np.array(bestDist)
    bestClass = np.array(bestClass)
    return bestDist, bestClass

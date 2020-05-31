import keras
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization


def createClassificationNet(input_img_size, num_output_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_img_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=input_img_size))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(num_output_classes, activation="softmax"))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer='nadam',
        metrics=["accuracy"],
    )

    return model


def testClassification(model, imageTests):
    bestClass = []
    for index, imageTest in enumerate(imageTests):
        print("frame {} processed".format(index))
        classPredicted = model.predict(np.reshape(imageTest, [1, imageTest.shape[0], imageTest.shape[1], imageTest.shape[2]]))
        bestClass.append(np.argmax(classPredicted))

    bestClass = np.array(bestClass)
    return bestClass

import os

import Augmentor
import cv2
import imageio as imageio
import numpy as np
from cv2.cv2 import imread


class DataUtils:

    def loadVideo(self, path="E:\\Workspaces\EndoscopyLocalizationFewShot\data\\video\\Fundus\\"):
        files = os.listdir(path)
        framesForVideos = []
        fs = 0
        for file in files:
            frames = []
            if str(file).__contains__("vid173.mp4"):
                vid = imageio.get_reader(path + file, 'ffmpeg')
                fs = vid.get_meta_data()['fps']
                for image in vid.iter_data():
                    image = np.array(image)
                    dst = self.preprocessImage(image)
                    frames.append(dst)
            if len(frames) > 0:
                framesForVideos.append(frames)
        frameForVideo1 = np.array(framesForVideos[0])
        return frameForVideo1, fs

    def loadImageAnnotated(self, data_path="E:\Workspaces\EndoscopyLocalizationFewShot\data\Positions\\"):

        X_all = []
        y_all = []
        label_all = []
        cat_dict = {}
        lang_dict = {}
        curr_y = 0
        # we load every alphabet seperately so we can isolate them later
        for ind, alphabet in enumerate(os.listdir(data_path)):
            # if ind > 8:
            #     break
            X = []
            y = []
            print("loading alphabet: " + alphabet)
            lang_dict[alphabet] = [curr_y, None]
            alphabet_path = os.path.join(data_path, alphabet)
            # every letter/category has it's own column in the array, so  load seperately
            for letter in os.listdir(alphabet_path):
                if not letter.__contains__(".jpg") and not letter.__contains__(".png"):
                    continue
                cat_dict[curr_y] = (alphabet, letter)
                category_images = []
                letter_path = os.path.join(alphabet_path, letter)

                image_path = os.path.join(letter_path)
                image = imread(image_path)
                dst = self.preprocessImage(image)
                y.append(curr_y)
                try:
                    X.append(dst)
                # edge case  - last one
                except ValueError as e:
                    print(e)
                    print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
            X_all.append(np.asarray(X))
            y_all.append(np.asarray(y))
            label_all.append(alphabet)
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        X_all = np.vstack(X_all)
        y_all = np.hstack(y_all)
        return X_all, y_all, label_all

    def windowingSig(self, sig1, sig2=None, windowSize=5):
        signalLen = sig1.shape[0]
        if len(sig1.shape) > 1:
            signalsWindow1 = [sig1[int(i):int(i + windowSize), :] for i in range(0, signalLen - windowSize, windowSize)]
        else:
            signalsWindow1 = [sig1[int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize, windowSize)]
        if sig2 is not None:

            if len(sig2.shape) > 1:
                signalsWindow2 = [sig2[int(i):int(i + windowSize), :] for i in range(0, signalLen - windowSize, windowSize)]
            else:
                signalsWindow2 = [sig2[int(i):int(i + windowSize)] for i in range(0, signalLen - windowSize, windowSize)]
        else:
            signalsWindow2 = None
        return signalsWindow1, signalsWindow2

    def augmentImagesForKeras(self, x_train, y_train):
        p = Augmentor.Pipeline()
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
        p.random_distortion(probability=0.4, grid_height=40, grid_width=40, magnitude=2)
        p.random_contrast(probability=0.5, min_factor=1.1, max_factor=1.5)
        g = p.keras_generator_from_array(x_train, y_train, batch_size=32)
        return g

    def preprocessImage(self, image):
        dst = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        # dst = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.resize(dst, (128, 128), interpolation=cv2.INTER_CUBIC)

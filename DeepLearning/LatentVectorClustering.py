from tkinter import Tk, Canvas, Scrollbar, VERTICAL, Label, Button, ACTIVE
from tkinter.ttk import Frame

import cv2
from PIL import Image, ImageTk

from keras.engine.saving import load_model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np
from matplotlib import gridspec
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def CAE(input_shape=(28, 28, 1), filters=None):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[2], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[0], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu'))

    model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model


import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def trainClusterImages(x, y, x_test, y_test):
    from time import time

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 32])
    plot_model(model, to_file=args.save_dir + '/EndoscopyClasses-pretrain-model.png', show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger(args.save_dir + '/Endoscopy-pretrain-log.csv')

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)

    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    feature_model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % ("Endoscopy", args.epochs))
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    clustering(features, y=y)


def testClustering(x, y=None, path=None):
    if path == None:
        model = load_model("E:\Workspaces\EndoscopyAnomalyFrameReview\EndoscopyFrameReview\\test\\results\\temp\Endoscopy-pretrain-model-250.h5")
    else:
        model = load_model(path)
    features = model.predictSegmentation(x)

    pred = clustering(features, y=y)

    ind = np.where(pred == 1)[0]

    root = Tk()
    GUI = DisplayImage(root,x[ind])
    GUI.read_image()
    root.mainloop()
    pass


class DisplayImage:

    def __init__(self, master, images):
        self.master = master
        master.title("GUI")
        self.image_frame = Frame(master, borderwidth=0, height=70, width=90)
        self.image_frame.pack()
        self.image_label = Label(self.image_frame, borderwidth=0)
        self.image_label.pack()
        self.Next_image = Button(master, command=self.read_image, text="Next image", width=17, default=ACTIVE, borderwidth=0)
        self.Next_image.pack()
        self.images = images
        self.number = 0

    def display_image(self, event=None):
        self.cv2image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGBA)
        self.from_array = Image.fromarray(self.cv2image)
        self.imgt = ImageTk.PhotoImage(image=self.from_array)
        self.image_label.configure(image=self.imgt)

    def read_image(self, event=None):
        self.img = self.images[self.number]
        self.master.after(10, self.display_image)
        self.number = self.number + 1


def clustering(features, y=None):
    features = np.reshape(features, newshape=(features.shape[0], -1))
    km = KMeans(n_clusters=len(np.unique(y)))
    pred = km.fit_predict(features)
    return pred

    # clust = OPTICS(min_samples=20, xi=.05, min_cluster_size=.05)
    # clust.fit(features)
    # space = np.arange(features.shape[0])
    # reachability = clust.reachability_[clust.ordering_]
    # labels = clust.labels_[clust.ordering_]
    # plt.figure(figsize=(10, 7))
    # G = gridspec.GridSpec(2, 2)
    # ax1 = plt.subplot(G[0, :])
    # ax2 = plt.subplot(G[1, :])
    #
    # # Reachability plot
    # colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    # for klass, color in zip(range(0, 5), colors):
    #     Xk = space[labels == klass]
    #     Rk = reachability[labels == klass]
    #     ax1.plot(Xk, Rk, color, alpha=0.3)
    # ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    # ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    # ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    # ax1.set_ylabel('Reachability (epsilon distance)')
    # ax1.set_title('Reachability Plot')
    #
    # # OPTICS
    # colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    # for klass, color in zip(range(0, 5), colors):
    #     Xk = features[clust.labels_ == klass]
    #     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    # ax2.plot(features[clust.labels_ == -1, 0], features[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    # ax2.set_title('Automatic Clustering\nOPTICS')
    # plt.show()
    #
    # fcm = FCM(n_clusters=len(np.unique(y)))
    # fcm.fit(features)
    # fcm_centers = fcm.centers
    # fcm_labels = fcm.u.argmax(axis=1)

    pass
    # print('acc=', acc(y, pred), 'nmi=', nmi(y, pred), 'ari=', ari(y, pred))

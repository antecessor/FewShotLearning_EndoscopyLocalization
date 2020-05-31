import glob
import os
import subprocess

import cv2
import keras
import numpy as np
import segmentation_models as sm
import matplotlib.pyplot as plt

def visualize(**images):
    """PLot images in one row."""

    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    # plt.show()



def loadTrainedSegmentationModel():
    BACKBONE = 'resnet34'
    LR = 0.0001
    n_classes = 9  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    # create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, .8, .5, .7, 1, 1, 1, 1]))
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)
    model.load_weights('E:\Workspaces\EndoscopyLocalizationFewShot\EndoscopyFrameReview\\test\segmentationModel.h5')

    return model


def predictSegmentation(model, image):
    out = model.predict(np.reshape(cv2.cvtColor(image,cv2.COLOR_YCR_CB2RGB), [1, image.shape[0], image.shape[1], image.shape[2]]))
    return out

def visualizeSegments(image,out):
    visualize(
        image=cv2.cvtColor(image,cv2.COLOR_YCR_CB2RGB),
        specularity=out[..., 0].squeeze(),
        saturation=out[..., 1].squeeze(),
        artifact=out[..., 2].squeeze(),
        blur=out[..., 3].squeeze(),
        contrast=out[..., 4].squeeze(),
        bubbles=out[..., 5].squeeze(),
        instrument=out[..., 6].squeeze(),
        blood=out[..., 7].squeeze(),
    )
def generate_video(images,outs,fs):
    for i,img in enumerate(images):
        visualizeSegments(img,outs[i])
        plt.savefig("./frameSaved/file%02d.png" % i)
        plt.close()


    os.chdir("./frameSaved/")
    subprocess.call([
        'ffmpeg', '-framerate', '{0}'.format(fs), '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p', '../out.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)



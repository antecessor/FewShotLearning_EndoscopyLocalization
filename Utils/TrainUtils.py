from keras.engine.saving import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from DeepLearning.ImageClassificationCNN import createClassificationNet, testClassification
from DeepLearning.LatentVectorClustering import trainClusterImages, testClustering
from DeepLearning.SegmentationNetwork import loadTrainedSegmentationModel, predictSegmentation, generate_video
from DeepLearning.SiameseNetowrk import create_base_network, trainSiamese, testSiamese, contrastive_loss
from Utils.DataUtils import DataUtils
from scipy.stats import stats
import numpy as np


class TrainUtils:

    def __init__(self) -> None:
        super().__init__()
        self.dataUtils = DataUtils()

    def trainCNNClassification(self, X, y, classNames):
        model = createClassificationNet(X[0].shape, len(classNames))
        y_oneHot = to_categorical(y, num_classes=len(classNames))
        X_train, X_test, y_train, y_test = train_test_split(X, y_oneHot, test_size=0.3, stratify=y_oneHot)
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.3)
        # augmentationGenerator = self.dataUtils.augmentImagesForKeras(X_train, y_train)

        batch_size = 16
        # model.fit_generator(augmentationGenerator, steps_per_epoch=len(X_train) / batch_size, epochs=10, validation_data=(X_valid, y_valid), class_weight={0: 150, 1: 1, 2: 150, 3: 250})
        model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_data=(X_valid, y_valid), class_weight={0: 150, 1: 1, 2: 150, 3: 250})
        y_pred = model.predictSegmentation(X_test)

        y_pred = y_pred.argmax(axis=1)
        y_test = y_test.argmax(axis=1)
        report = classification_report(y_test, y_pred, target_names=classNames)
        print(report)
        return model, report

    def testClassificationMOdel(self, testImages, yGroupNames, fs=25):
        model = load_model('E:\Workspaces\EndoscopyLocalizationFewShot\EndoscopyFrameReview\\test\AnomalyModel.h5')
        bestClass = testClassification(model, testImages)
        windowsize = int(fs * 2)
        classWindows, _ = self.dataUtils.windowingSig(bestClass, windowSize=windowsize)
        labelNames = []
        for index, classWindow in enumerate(classWindows):
            selectedClassNum = stats.mode(classWindow)
            labelNames.append(yGroupNames[selectedClassNum.mode[0]])

        return labelNames, windowsize

    def testSegmenationModel(self, testImages, fs):
        model = loadTrainedSegmentationModel()
        segmented = []
        for ind, image in enumerate(testImages):
            print("Segmenting Frame : {}".format(ind))
            segmented.append(predictSegmentation(model, image))
        generate_video(testImages, segmented, fs)
        return segmented

    def trainLatentClustering(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
        trainClusterImages(X_train, y_train, X_test, y_test)

    def applyClustering(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
        testClustering(X_train, y_train)

    def trainSiameseNetwork(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
        model = trainSiamese(X_train, y_train, X_test, y_test)
        model.save("PositionModel.h5")
        return model

    def testSiamseNetwork(self, xGroups, yGroups, testImages, yGroupNames, fs=25):
        model = load_model('E:\Workspaces\EndoscopyLocalizationFewShot\EndoscopyFrameReview\\test\PositionModel.h5', custom_objects={"contrastive_loss": contrastive_loss})
        bestDist, bestClass = testSiamese(model, xGroups, yGroups, testImages)
        windowsize = int(fs * 2)
        distWindow, classWindow = self.dataUtils.windowingSig(bestDist, bestClass, windowSize=windowsize)
        labels = []
        distances = []
        labelNames = []
        for index, distW in enumerate(distWindow):
            filter = (distWindow[index] < 0.5)
            if any(filter.ravel()):
                res = stats.mode(classWindow[index][filter], axis=None)
                distances.append(np.min(distWindow[index][filter]))
                assignLabel = yGroups[res.mode[0]]
                assignLabelName = yGroupNames[assignLabel]
                # if len(labels) > 0:
                #     lastLabel = np.max(labels)
                #     indLastLabel = np.where(labels == lastLabel)[0]
                #     timePreviosuOccurance = (index - indLastLabel[0]) * windowsize / fs
                #     if np.max(labels) > yGroups[res.mode[0]]:
                #         assignLabel = -1
                #         assignLabelName = "Other"
                #     else:
                #         if np.max(labels) == yGroups[res.mode[0]] or timePreviosuOccurance > 4:
                #             assignLabel = yGroups[res.mode[0]]
                #             assignLabelName = yGroupNames[assignLabel]
                #             if assignLabel - lastLabel > 2 and timePreviosuOccurance < 10:
                #                 assignLabel = -1
                #                 assignLabelName = "Other"
                #         else:
                #             assignLabel = -1
                #             assignLabelName = "Other"
                # else:
                #     assignLabel = yGroups[res.mode[0]]
                #     assignLabelName = yGroupNames[assignLabel]
                #
                labels.append(assignLabel)
                labelNames.append(assignLabelName)

            else:
                labels.append(-1)
                labelNames.append("Other")
                distances.append(np.min(distWindow[index]))
        return labelNames, distances, windowsize

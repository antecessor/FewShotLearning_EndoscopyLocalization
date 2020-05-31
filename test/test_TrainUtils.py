from unittest import TestCase

from keras.engine.saving import load_model

from Utils.DataUtils import DataUtils
from Utils.TrainUtils import TrainUtils
from Utils.VisualizeUtil import CreateGanttChartForPosition, CreateGanttChartForAnomaly, CreateGanttChartForAnomalyAndPosition
import numpy as np


class TestTrainUtils(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.trainUtils = TrainUtils()
        self.dataUtils = DataUtils()

    def test_train_latent_clustering(self):
        X, y, _ = self.dataUtils.loadImageAnnotated()
        self.trainUtils.trainLatentClustering(X, y)
        pass

    def test_apply_clustering(self):
        X, y, _ = self.dataUtils.loadImageAnnotated()
        self.trainUtils.applyClustering(X, y)
        pass

    def test_trainSiameseNet(self):
        X, y, _ = self.dataUtils.loadImageAnnotated()
        self.trainUtils.trainSiameseNetwork(X, y)
        pass

    def test_testSiameseNetByVideo(self):
        X, y, labelNames = self.dataUtils.loadImageAnnotated()
        images, fs = self.dataUtils.loadVideo()
        labels, dist, windowsize = self.trainUtils.testSiamseNetwork(X, y, images, labelNames, fs)
        times = np.array([range(len(labels))]) * windowsize / fs
        CreateGanttChartForPosition(labels, list(times.ravel()), dist)
        pass

    def test_trainClassificationNet(self):
        X, y, labels = self.dataUtils.loadImageAnnotated("E:\Workspaces\EndoscopyLocalizationFewShot\data\Anomaly")
        model, report = self.trainUtils.trainCNNClassification(X, y, labels)
        model.save("AnomalyModel.h5")
        pass

    def test_testClassificationNetByVideo(self):
        labelNames = ["Inflammatory", "Normal", "Polypoid", "Vascular"]
        images, fs = self.dataUtils.loadVideo()
        labels, windowsize = self.trainUtils.testClassificationMOdel(images, labelNames, fs)
        times = np.array([range(len(labels))]) * windowsize / fs
        CreateGanttChartForAnomaly(labels, list(times.ravel()), labelNames)
        pass

    def test_localizationAndClassificationByVideos(self):
        labelNamesAnomaly = ["Inflammatory", "Normal", "Polypoid", "Vascular"]
        images, fs = self.dataUtils.loadVideo()

        self.trainUtils.testSegmenationModel(images,fs)

        labelsAnomaly, windowsizeAnomaly = self.trainUtils.testClassificationMOdel(images, labelNamesAnomaly, fs)
        timesAnomaly = np.array([range(len(labelsAnomaly))]) * windowsizeAnomaly / fs

        X, y, labelNamesPosition = self.dataUtils.loadImageAnnotated()
        labelsPosition, dist, windowsizePosition = self.trainUtils.testSiamseNetwork(X, y, images, labelNamesPosition, fs)
        timesPosition = np.array([range(len(labelsPosition))]) * windowsizePosition / fs

        CreateGanttChartForAnomalyAndPosition(labelAnomaly=labelsAnomaly, timeAnomaly=list(timesAnomaly.ravel()), labelNamesAnomaly=labelNamesAnomaly, labelPosition=labelsPosition,
                                              timePosition=list(timesPosition.ravel()),
                                              dist=dist, labelNamesPosition=labelNamesPosition)

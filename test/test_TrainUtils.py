from unittest import TestCase

from keras.engine.saving import load_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from werkzeug._compat import izip

from Utils.DataUtils import DataUtils
from Utils.TrainUtils import TrainUtils
from Utils.VisualizeUtil import CreateGanttChartForPosition, CreateGanttChartForAnomaly, CreateGanttChartForAnomalyAndPosition
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


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

        self.trainUtils.testSegmenationModel(images, fs)

        labelsAnomaly, windowsizeAnomaly = self.trainUtils.testClassificationMOdel(images, labelNamesAnomaly, fs)
        timesAnomaly = np.array([range(len(labelsAnomaly))]) * windowsizeAnomaly / fs

        X, y, labelNamesPosition = self.dataUtils.loadImageAnnotated()
        labelsPosition, dist, windowsizePosition = self.trainUtils.testSiamseNetwork(X, y, images, labelNamesPosition, fs)
        timesPosition = np.array([range(len(labelsPosition))]) * windowsizePosition / fs

        CreateGanttChartForAnomalyAndPosition(labelAnomaly=labelsAnomaly, timeAnomaly=list(timesAnomaly.ravel()), labelNamesAnomaly=labelNamesAnomaly, labelPosition=labelsPosition,
                                              timePosition=list(timesPosition.ravel()),
                                              dist=dist, labelNamesPosition=labelNamesPosition)

    def getClassificationReport(self, excelName, selectedLabels, type, testSet):
        res = pd.read_excel(excelName)
        Target = res['Target']
        Output = res['Output']
        sampleClassificationReport = classification_report(Target, Output, target_names=selectedLabels, output_dict=True)
        df = pd.DataFrame(sampleClassificationReport)
        df = df.transpose()
        acc = df.loc['accuracy'].values[0]
        df = df.drop('accuracy', axis=0)
        df.reset_index(level=0, inplace=True)
        df['type'] = np.repeat([type], df.shape[0])
        df['Test Set'] = np.repeat([testSet], df.shape[0])
        return df, acc

    def getDFForHeatClassificationResult(self, excelName, selectedLabels):
        res = pd.read_excel(excelName)
        Target = res['Target']
        Output = res['Output']
        sampleClassificationReport = classification_report(Target, Output, target_names=selectedLabels, output_dict=True)
        df = pd.DataFrame(sampleClassificationReport).iloc[:-1, :].T
        return df

    def testPlotBarResults(self):
        labels = np.asarray(["Larynx", "Oesophagus", "Cardia", "Angularis", "Antrum", "Pylorus", "Duodenum", "Jejunum", "Ileum", "Appendix", "Colon", "Rectum", "Anus"])
        numberCETest1 = [4, 3, 3, 3, 4, 2, 7, 2, 5, 2, 5, 2, 2]
        numberWCETest1 = [0, 1, 1, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0]
        numberCETest2 = [50, 3075, 2450, 500, 0, 2500, 2700, 1500, 0, 0, 5475, 5500, 2000]
        numberWCETest2 = [0, 260, 20, 0, 0, 280, 130, 380, 280, 0, 475, 0, 0]

        selectedLabelsCE2 = labels[np.where(np.asarray(numberCETest2) != 0)[0]]
        dfCE, accCE = self.getClassificationReport("CETest2.xlsx", selectedLabelsCE2, "CE", "Without")
        selectedLabelsWCE2 = labels[np.where(np.asarray(numberWCETest2) != 0)[0]]
        dfWCE, accWCE = self.getClassificationReport("WCETest2.xlsx", selectedLabelsWCE2, "WCE", "Without")

        df1 = dfCE.append(dfWCE)

        selectedLabelsCE2 = labels[np.where(np.asarray(numberCETest2) != 0)[0]]
        dfCE, accCE = self.getClassificationReport("CETest2After.xlsx", selectedLabelsCE2, "CE", "With")
        selectedLabelsWCE2 = labels[np.where(np.asarray(numberWCETest2) != 0)[0]]
        dfWCE, accWCE = self.getClassificationReport("WCETest2After.xlsx", selectedLabelsWCE2, "WCE", "With")

        df2 = dfCE.append(dfWCE)

        df = df1.append(df2)

        del df['support']
        df.columns = ['index', 'Precision', 'Recall', 'F1-Score', 'type', 'Post-Processing']

        order = np.append(labels, ["macro avg", 'weighted avg'])
        g = sns.catplot(x="index", y="F1-Score", hue="Post-Processing", data=df, col='type', kind='bar', order=order)
        [plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
        g.savefig('TestResAfter.eps', format='eps')

    def testPlotHeatMapResults(self):
        labels = np.asarray([ "Oesophagus", "Cardia", "Angularis", "Antrum", "Pylorus", "Duodenum", "Jejunum", "Ileum", "Appendix", "Colon", "Rectum", "Anus"])
        numberCETest1 = [ 3, 3, 3, 4, 2, 7, 2, 5, 2, 5, 2, 2]
        numberWCETest1 = [ 1, 1, 0, 0, 1, 2, 1, 2, 0, 2, 0, 0]
        numberCETest2 = [ 3075, 2450, 500, 0, 2500, 2700, 1500, 0, 0, 5475, 5500, 2000]
        numberWCETest2 = [ 260, 20, 0, 0, 280, 130, 380, 280, 0, 475, 0, 0]

        selectedLabelsCE2 = labels[np.where(np.asarray(numberWCETest2) != 0)[0]]
        name = "WCETest2DeepLearningAfter.xlsx"
        dfCE = self.getDFForHeatClassificationResult(name, selectedLabelsCE2)

        g = sns.heatmap(pd.DataFrame(dfCE).iloc[:-1, :].T, annot=True)

        g.figure.savefig('heatMap_{}.eps'.format(name.split('.')[0]), format='eps')

    def testBlandAltman(self):
        Names = ['CETest2.xlsx', 'CETest2After.xlsx', 'WCETest2.xlsx', 'WCETest2After.xlsx']
        titles = ["Before pre-processing on CE", "After pre-processing on CE", "Before pre-processing on WCE", "After pre-processing on WCE"]
        f, ax = plt.subplots(2, 2, figsize=(8, 5))
        k = 0
        for i in range(2):
            for j in range(2):
                data = pd.read_excel(Names[i])
                target = data["Target"].values
                predicted = data["Output"].values

                sm.graphics.mean_diff_plot(predicted, target, ax=ax[i][j])

                ax[i][j].autoscale()
                ax[i][j].set_title(titles[k])
                if k % 2 != 0:
                    ax[i][j].get_yaxis().set_visible(False)
                k = k + 1
        plt.tight_layout()
        plt.show()
        plt.savefig("blandAtlmanFigTest2ProposedMethod.png")

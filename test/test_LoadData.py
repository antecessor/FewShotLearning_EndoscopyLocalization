from unittest import TestCase

from Utils.DataUtils import DataUtils
import Augmentor


class TestLoadData(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.loadData = DataUtils()

    def test_load_video(self):
        frames, fs = self.loadData.loadVideo("E:\Workspaces\EndoscopyAnomalyFrameReview\data\\video")
        self.assertIsNotNone(frames)

    def test_loadDataForTrain(self):
        x, y = self.loadData.loadImageAnnotated()
        self.assertIsNotNone(x)

    def test_imageAugmentation(self):
        p = Augmentor.Pipeline("E:\Workspaces\EndoscopyLocalizationFewShot\data\Anomaly\Polypoid")
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
        p.random_distortion(probability=0.4, grid_height=40, grid_width=40, magnitude=2)
        p.random_contrast(probability=0.5, min_factor=1.1, max_factor=1.5)
        p.sample(1000)

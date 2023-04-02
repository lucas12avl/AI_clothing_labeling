import pickle
import unittest

import Kmeans as km
from Kmeans import *
from utils import *


# unittest.TestLoader.sortTestMethodsUsing = None

class TestCases(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        with open('./test/test_cases_kmeans.pkl', 'rb') as f:
            self.test_cases = pickle.load(f)

    def test_01_NIU(self): #NO HACER CASO, SOLO COMPRUEBA QUE ESTEN LOS NIUS Y EL GRUPO
        # DON'T FORGET TO WRITE YOUR NIU AND GROUPS
        self.assertNotEqual(km.__authors__, "1636290, 1631153, 1636589", msg="CHANGE IT TO YOUR NIU!")
        self.assertNotEqual(km.__group__, "DJ.10", msg="CHANGE YOUR GROUP NAME!")

    def test_02_init_X(self): #HECHA 
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            np.testing.assert_array_equal(km.X, self.test_cases['shape'][ix])

    def test_03_init_centroids(self):
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km._init_centroids()
            np.testing.assert_array_equal(km.centroids, self.test_cases['init_centroid'][ix])

    def test_04_distance(self):
        for ix, input in enumerate(self.test_cases['shape']):
            dist = km.distance(input, self.test_cases['init_centroid'][ix])
            np.testing.assert_array_almost_equal_nulp(dist, self.test_cases['distance'][ix])

    def test_05_get_labels(self):
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km._init_centroids()
            km.get_labels()
            np.testing.assert_array_equal(km.labels, self.test_cases['labels'][ix])

    def test_06_get_centroids(self):
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km._init_centroids()
            km.get_labels()
            km.get_centroids()
            # Compare old centroids
            np.testing.assert_array_equal(km.old_centroids, self.test_cases['get_centroid'][ix][0])
            # Compare new centroids
            np.testing.assert_array_equal(km.centroids, self.test_cases['get_centroid'][ix][1])

    def test_07_converges(self):
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km._init_centroids()
            old_centroid, centroid, bool_value = self.test_cases['converge'][ix]
            km.old_centroids, km.centroids = old_centroid, centroid
            self.assertEqual(km.converges(), bool_value)

    def test_08_Kmeans(self):
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km.fit()
            np.testing.assert_array_equal(km.centroids, self.test_cases['kmeans'][ix])

    def test_09_find_bestK(self):
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km.find_bestK(10)
            self.assertEqual(km.K, self.test_cases['bestK'][ix])

    def test_10_get_color(self):
        for ix, centroid in enumerate(self.test_cases['kmeans']):
            color = get_colors(centroid)
            self.assertCountEqual(color, self.test_cases['color'][ix])


if __name__ == "__main__":
    unittest.main()

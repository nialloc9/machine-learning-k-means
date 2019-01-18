from matplotlib import style, pyplot
import numpy as np
import random
from sklearn.cluster import KMeans
style.use('ggplot')


def get_random_int(min=1, max=20):
    return random.randint(min, max)


def graph_data(features, labels, centroids):
    colors = ["g.", "r.", "c.", "y."]

    for i in range(len(features)):
        pyplot.plot(features[i][0], features[i][1], colors[labels[i]], markersize=10)

    pyplot.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)

    pyplot.show()


features = np.array([[get_random_int(), get_random_int()],
              [get_random_int(), get_random_int()],
              [get_random_int(), get_random_int()],
              [get_random_int(), get_random_int()],
              [get_random_int(), get_random_int()],
              [get_random_int(), get_random_int()]])


classifier = KMeans(n_clusters=2)
classifier.fit(features)

centroids = classifier.cluster_centers_
labels = classifier.labels_

graph_data(features, labels, centroids)
#!/usr/bin/env python

import math
from random import randint, uniform
import matplotlib.pyplot as plt


def distance(p1, p2):
    # euclidean distance
    dims = len(p1)
    d = math.sqrt(sum([(p1[d]-p2[d])**2 for d in range(dims)]))
    return abs(d)


def cluster_loop(centroids, K):
    # distance from each point to each centroid
    distances = [[] for _ in range(K)]
    for idx, c in enumerate(centroids):
        for point in data:
            d = distance(point, c)
            distances[idx].append(d)
    
    # find closest centroid to each point
    nearest = [ds.index(min(ds)) for ds in zip(*distances)]
    clusters = [[] for _ in range(K)]
    for n, p in zip(nearest, data):
        clusters[n].append(p)
    
    # create new centroids (average of the points in each cluster)
    new_centroids = []
    for idx, cluster in enumerate(clusters):
        if not cluster:
            new_centroids.append(centroids[idx])
        else:
            avg_x = sum([x for x, _ in cluster]) / len(cluster)
            avg_y = sum([y for _, y in cluster]) / len(cluster)
            new_centroids.append([avg_x, avg_y])
    
    return new_centroids, clusters


# K MEANS #############################

K = 3

## 50 random points around 3 random centers
a,b,c = [randint(2, 18) for _ in range(K)]
centroids = [[a, b], [b, a], [c, c]]
print(f"Centroids: {centroids}")
data = []
for centroid in centroids:
    for _ in range(50):
        # randomize point coords
        x = centroid[0] + uniform(-2, 2)
        y = centroid[1] + uniform(-2, 2)
        data.append([x, y])

## clustering loop
clusters = []
done = False
while not done:
    new_centroids, clusters = cluster_loop(centroids, K)
    done = centroids == new_centroids
    centroids = new_centroids
print(f"Done: {centroids}")
for idx, cluster in enumerate(clusters):
    print(f"{idx}: {cluster}")


# PLOT IT #############################

colors = ['r', 'g', 'b']
plt.figure(figsize=(8, 6))

## plot clusters in red, green, and blue
for idx, cluster in enumerate(clusters):
    if len(cluster) > 0:  # safely skip empty clusters if that happens
        points = list(zip(*cluster))
        plt.scatter(points[0], points[1], color=colors[idx % len(colors)], label=f'Cluster {idx}')

## plot centroids as black X
centroid_x, centroid_y = zip(*centroids)
plt.scatter(centroid_x, centroid_y, color='black', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()


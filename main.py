#get the cluster makes going here.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score
import umap
import numpy as np

file=np.load("feat_dinov2_VocPascal.npy")
#print(file)

scaler=StandardScaler()

kmeans20=KMeans(n_clusters=20)
kmeans20.fit(file)
kmeans50=KMeans(n_clusters=50)
kmeans50.fit(file)

kmeans_20=kmeans20.labels_
kmeans_50=kmeans50.labels_
print(kmeans_20,kmeans_50)

print(len(file))
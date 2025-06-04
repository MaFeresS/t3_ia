#get the cluster makes going here.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score
import umap
import numpy as np

#loading file and labels, both np.array
file=np.load("feat_dinov2_VocPascal.npy")
with open("VocPascal/val_voc.txt","r") as labelfile:
  labels=[]
  for i in labelfile.readlines():
    labels.append(i.split("\t")[1])
labels=np.array(labels)


#scaler=StandardScaler()

#just kmeans
kmeans20=KMeans(n_clusters=20)
kmeans20.fit(file)
kmeans_20=kmeans20.labels_
kmeans50=KMeans(n_clusters=50)
kmeans50.fit(file)
kmeans_50=kmeans50.labels_
#output from just kmeans
ri20=rand_score(labels,kmeans_20)
ari20=adjusted_rand_score(labels,kmeans_20)
print(f"ri20: {ri20}, ari20: {ari20}")
ri50=rand_score(labels,kmeans_50)
ari50=adjusted_rand_score(labels,kmeans_50)
print(f"ri50: {ri50}, ari50: {ari50}")

#pca settup
pca=PCA(n_components=384)
pca.fit(file)
pcat=pca.transform(file)
#PCA -> kmeans
kmeans20=KMeans(n_clusters=20)
kmeans20.fit(pcat)
kmeans_20=kmeans20.labels_
kmeans50=KMeans(n_clusters=50)
kmeans50.fit(pcat)
kmeans_50=kmeans20.labels_
#output from PCA -> kmeans
ri20=rand_score(labels,kmeans_20)
ari20=adjusted_rand_score(labels,kmeans_20)
print(f"pca ri20: {ri20}, pca ari20: {ari20}")
ri50=rand_score(labels,kmeans_50)
ari50=adjusted_rand_score(labels,kmeans_50)
print(f"pca ri50: {ri50}, pca ari50: {ari50}")


print(len(file))
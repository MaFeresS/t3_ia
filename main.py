#get the cluster makes going here.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt

#loading file and labels, both np.array
file=np.load("feat_dinov2_VocPascal.npy")
with open("VocPascal/val_voc.txt","r") as labelfile:
  labels=[]
  for i in labelfile.readlines():
    labels.append(i.split("\t")[1])
labels=np.array(labels)

kmeans20=KMeans(n_clusters=20)
kmeans50=KMeans(n_clusters=50)

comp_sizes=[16,32,64,128,256,2]

#just kmeans
kmeans20.fit(file)
kmeans_20=kmeans20.labels_
kmeans50.fit(file)
kmeans_50=kmeans50.labels_
#output from just kmeans
ri20=rand_score(labels,kmeans_20)
ari20=adjusted_rand_score(labels,kmeans_20)
print(f"ri20: {ri20}, ari20: {ari20}")
ri50=rand_score(labels,kmeans_50)
ari50=adjusted_rand_score(labels,kmeans_50)
print(f"ri50: {ri50}, ari50: {ari50}")

for i in comp_sizes:
  #PCA settup
  scaler=StandardScaler()
  infile=scaler.fit_transform(file)
  pca=PCA(n_components=i)
  pca_file=pca.fit_transform(infile)
  #PCA -> kmeans

  kmeans20.fit(pca_file)
  kmeans_20=kmeans20.labels_
  kmeans50.fit(pca_file)
  kmeans_50=kmeans20.labels_
  #output from PCA -> kmeans
  if i!=2:
    print(f"comp: {i}")
    ri20=rand_score(labels,kmeans_20)
    ari20=adjusted_rand_score(labels,kmeans_20)
    print(f"pca ri20: {ri20}, pca ari20: {ari20}")
    ri50=rand_score(labels,kmeans_50)
    ari50=adjusted_rand_score(labels,kmeans_50)
    print(f"pca ri50: {ri50}, pca ari50: {ari50}")
  elif i==2:
    plt.scatter(x=pca_file[:,0][:100],y=pca_file[:,1][:100],c=kmeans_20[:100])
    plt.title("PCA 2D usando k = 20")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()
    plt.scatter(x=pca_file[:,0][:100],y=pca_file[:,1][:100],c=kmeans_50[:100])
    plt.title("PCA 2D usando k = 50")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()

for i in comp_sizes:
  #UMAP settup
  #scaler=StandardScaler()
  #infile=scaler.fit_transform(file)
  umap_=UMAP(n_components=i)
  umap_file=umap_.fit_transform(file)
  #UMAP -> kmeans

  kmeans20.fit(umap_file)
  kmeans_20=kmeans20.labels_
  kmeans50.fit(umap_file)
  kmeans_50=kmeans20.labels_
  #output from UMAP -> kmeans
  if i!=2:
    print(f"comp: {i}")
    ri20=rand_score(labels,kmeans_20)
    ari20=adjusted_rand_score(labels,kmeans_20)
    print(f"pca ri20: {ri20}, pca ari20: {ari20}")
    ri50=rand_score(labels,kmeans_50)
    ari50=adjusted_rand_score(labels,kmeans_50)
    print(f"pca ri50: {ri50}, pca ari50: {ari50}")
  elif i==2:
    plt.scatter(x=umap_file[:,0][:100],y=umap_file[:100,1][:100],c=kmeans_20[:100])
    plt.title("UMAP 2D usando k = 20")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()
    plt.scatter(x=umap_file[:,0][:100],y=umap_file[:,1][:100],c=kmeans_50[:100])
    plt.title("UMAP 2D usando k = 50")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()


print(len(file))
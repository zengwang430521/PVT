import torch
# from sklearn.cluster import spectral_clustering
#
# x = torch.zeros(1000, 64)
# A = torch.cdist(x, x)
# clu = spectral_clustering(A.numpy(), n_clusters=250)


import torchcluster
model = torchcluster.zoo.spectrum.SpectrumClustering(n_clusters=250, threshold=1000, k=64)
x = torch.rand(1000, 64)
y = model(x)
t = y[0]
y=y
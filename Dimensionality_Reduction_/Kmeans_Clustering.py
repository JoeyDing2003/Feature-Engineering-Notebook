# Implete some metric calculation functions
from sklearn.cluster import KMeans
import matplotlib.ticker as ticker
def calculate_WSS(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k, random_state=30, max_iter= 1000, n_init=20).fit(points) # points is the 2d array
    # n_init (Number of Initializations: Runs K-Means multiple times with different centroid initializations. The value is 10 (default), 20-50 for better stability.

    centroids = kmeans.cluster_centers_ # an arrary of coordinates of k cluster centers
    pred_clusters = kmeans.predict(points) # Compute cluster centers return a 1d arrary of K labels that each data belongs to
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse


from sklearn.metrics import silhouette_score

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
def sil(points, kmax): 
  sil = []
  for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k, random_state=30, max_iter= 1000, n_init=20).fit(points)
    labels = kmeans.labels_
    sil.append(silhouette_score(points, labels, metric = 'euclidean'))
  return sil
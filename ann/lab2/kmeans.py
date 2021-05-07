import numpy as np

def kmeans(x,k):
    current_clusters = np.random.choice(np.squeeze(x), size=k)
    clusters = current_clusters.copy()
    converged = False

    while not converged:
	dist = np.squeeze(np.abs(x[:,np.newaxis] - current_clusters[np.newaxis,:]))
	closest = np.argmin(dist,axis = 1)
	for i in range(k):
	    points = x[closest == i]
	    if len(points) > 0:
		current_clusters[i] = np.mean(points, axis = 0)
	
	if np.linalg.norm(current_clusters - clusters) < 1e-6:
	    converged = True
	clusters = current_clusters.copy()

    return current_clusters

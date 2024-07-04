#####
## pip install numpy matplotlib scikit-learn scikit-image

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize


# # Randomly select initial centroids from the dataset.
# def initialize_centroids(X, K):
#     ### YOUR CODE HERE
    
#     centroids = 
#     return centroids


# Finds the closest centroid for each sample
# X-> dataset (each row is a data point)
def find_closest_centroids(X, centroids):
    print("in find_closest_centroids")
    # Number of centroids
    K = centroids.shape[0]
    
    # Number of samples
    num_of_samples = len(X)
    
    # Initialize array to store index of closest centroid for each sample
    idx = np.zeros(num_of_samples)
    
    # For each sample
    for i in range(num_of_samples):
        min_dist = float('inf')# starting minimum distance is infinity
        closest_centroid = -1 # starting with null 
        
        # Iterate over each centroid
        for k in range(K):
            # Compute Euclidean distance between sample and centroid
            #|u| = sqrt(sum_j(uj^2))
            # sum_of_squares = np.sum((X[i] - centroids[k]) ** 2)
            # dist = np.sqrt(sum_of_squares) 
            dist = np.linalg.norm(X[i] - centroids[k])             
            
            # If calculated distance < previous min distance  
            # update the new closest centroid and min distance
            if dist < min_dist:
                min_dist = dist
                closest_centroid = k
        
        
        idx[i] = closest_centroid # for this sample assign the closest found centroid
    return idx

# Compute the mean of samples assigned to each centroid
# X-> dataset (each row is a data point)
# idx -> array of the index of the cluster assignments for each data point
# K -> # of clusters
def compute_centroids(X, idx, K):
    print("in compute_centroids")
    # X.shape will return a tuple (number_of_samples, number_of_features)
    num_of_features = X.shape[1]   
    # Initialize centroids with zeros
    centroids = np.zeros((K, num_of_features))
    
    #For each class
    #Implementing the formula for calculating the centroid of each class
    for k in range(K):#for each cluster  
        cluster_points = X[idx == k]
        centroids[k] = np.mean(cluster_points, axis=0)
        
        
        
        
        
        
        
        
        ################################################################
        #cluster_points = np.array([])# list to store data for cluster k
        
        # #Loop through data to get all the points assigned to cluster k
        # for i in range(len(X)):           
        #     if idx[i] == k: # If the data point is assigned to the k-th cluster, add it to the list
        #         cluster_points = np.append(cluster_points,X[i])
        
        # centroids[k] = np.mean(cluster_points, axis=0)  # axis=0 specifies that the mean should be calculated along the first
        #                                                 # axis of points_in_cluster -> calculated for each feature (column) independently 
        #                                                 # across all the data points in points_in_cluster
          ##############################################################################  
    return centroids

# K-means algorithm for a specified number of iterations
def run_kmeans(X, initial_centroids, max_iters):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for i in range(max_iters):
        print(i)
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# Initialize centroids randomly
def kmeans_init_centroids(X, K):
    # Number of samples in the dataset
    num_of_samples = X.shape[0]    
    # Randomly shuffle indices to select K centroids
    random_indices = np.random.choice(num_of_samples, K, replace=False)    
    # Initialize centroids array
    initial_centroids = X[random_indices]
    return initial_centroids

# Load the image
image = io.imread('C:/Users/gfrag/Desktop/Project2_Statistical_Modeling_Pattern_Recognition/exercise2_4/Fruit.png')
# print("Before normalization ")
# print(image)
#image = resize(image, (256, 256))  # Resize for faster processing, if needed
#rows, cols, dims = image.shape

# Size of the image
img_size = image.shape

# Normalize image values in the range 0 - 1
# max_pixel_value = np.max(image)
max_pixel_value = 255.0
image = image / max_pixel_value
# print("After normalization ")
# print(image)

# Size of the image
img_size = image.shape

# Reshape the image to be a Nx3 matrix (N = num of pixels)
#X = image.reshape(img_size[0] * img_size[1], 4)
X = image.reshape(-1, 4)
print("Image")
print(image)
# X = image.reshape(-1, 4)
print("X, after reshaping")
print(X)



# Perform K-means clustering
K = 8
max_iters = 10
##########################################################################
# Initialize the centroids randomly
initial_centroids = kmeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kmeans(X, initial_centroids, max_iters)


# K-Means Image Compression
print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = find_closest_centroids(X, centroids)
print(idx)
print(idx.shape)
# Recover the image from the indices
idx = idx.astype(int)
X_recovered = centroids[idx]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size)
# X_recovered = X_recovered.reshape(img_size[0], img_size[1], 4)
# X_recovered = X_recovered.reshape(-1, 4)
print(X_recovered)


# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title(f'Compressed, with {K} colors.')

plt.show()




#####
## pip install numpy matplotlib scikit-learn scikit-image

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize


# Randomly select initial centroids from the dataset.
def initialize_centroids(X, K):
    ### YOUR CODE HERE
    
    centroids = 
    return centroids


# Finds the closest centroid for each sample
def find_closest_centroids(X, centroids):
    ### YOUR CODE HERE
        
    idx = 
    return idx

# Compute the mean of samples assigned to each centroid
def compute_centroids(X, idx, K):
    ### YOUR CODE HERE

    centroids = 
    return centroids

# K-means algorithm for a specified number of iterations
def run_kmeans(X, initial_centroids, max_iters):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# Initialize centroids randomly
def kmeans_init_centroids(X, K):
    ### YOUR CODE HERE

    initial_centroids = 
    return initial_centroids

# Load the image
image = io.imread('Fruit.png')
#image = resize(image, (256, 256))  # Resize for faster processing, if needed
#rows, cols, dims = image.shape

# Size of the image
img_size = image.shape

# Normalize image values in the range 0 - 1
### YOUR CODE HERE

image = 


# Reshape the image to be a Nx3 matrix (N = num of pixels)
X = image.reshape(img_size[0] * img_size[1], 3)

# Perform K-means clustering
K = 16
max_iters = 10

# Initialize the centroids randomly
initial_centroids = kmeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kmeans(X, initial_centroids, max_iters)


# K-Means Image Compression
print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = find_closest_centroids(X, centroids)

# Recover the image from the indices
X_recovered = centroids[idx]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title(f'Compressed, with {K} colors.')

plt.show()




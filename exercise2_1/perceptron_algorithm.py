import numpy as np
import matplotlib.pyplot as plt

# Define the samples given for the exercise
samples = {
    'ω1': np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8],
                    [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5, 6.4], [3.9, 4]]),
    'ω2': np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0], [6.3, 1.6], [4.2, 1.9],
                    [1.4, -3.2], [2.4, -4], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]),
    'ω3': np.array([[-3, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4, 2.2],
                    [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]),
    'ω4': np.array([[-2, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
                    [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])
}

# Plot the samples for question a)
def plot_samples_boundaries(class1,class2):
    colors = {'ω1': 'b', 'ω2': 'r', 'ω3': 'g', 'ω4': 'y'}
    for label, points in samples.items():
        plt.scatter(points[:, 0], points[:, 1], c=colors[label], label=label)

    # plot the boundaries if there exist
    if(class1 != None and class2 != None):
        classify_and_plot(class1,class2)
        return 

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Samples from 4 classes with different colors')
    plt.grid(True)
    plt.show()


# Perceptron algorithm implementation
def perceptron(X, delta_x, max_iter=10000, r = 0.0005):
    np.random.seed(0)
    w = np.zeros(X.shape[1]) + np.exp(-9)      # initialize the weights to zero
    b = 0                          # bias term or threashold
    t = 0                          # number of iterations

    while t < max_iter:
        Y = []
        for i in range(X.shape[0]):
            #if delta_x[i] * (np.dot(w, X[i]) + b) >= 0:
            if delta_x[i] * (np.dot(w, X[i]) + b) < 0:
                Y.append(i)
        
        # No missclassified samples -> algorithm converged
        if len(Y) == 0: 
            break
        if(t == 1):
            print(f"\nStarting with: {len(Y)} Misclassified Samples")
        
        # Updating the parameter vector
        dw = np.sum([-delta_x[i] * X[i] for i in Y], axis=0)
        db = np.sum([-delta_x[i] for i in Y])
        w = w - r * dw
        b = b - r * db
        
        t += 1

    return w, b, t


# Function to prepare data and apply Perceptron
def classify_and_plot(class1, class2):
    X1 = samples[class1]
    X2 = samples[class2]
    X = np.vstack((X1, X2))
    delta_x = np.hstack((-np.ones(X1.shape[0]), np.ones(X2.shape[0]))) # -1 for class1, 1 for class2
    
    # run batch perceptron
    w, b, iterations = perceptron(X, delta_x)
    print(f"Number Of Iterations Until Convergence: {iterations} for classes: {class1},{class2}")
    
    # plot the decision boundary for the given classes
    x_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k--', label='Decision boundary')
    plt.legend()
    plt.title(f'Classification between {class1} and {class2}')
    plt.show()



# start with just printing the samples of the different classes
plot_samples_boundaries(None, None)

# perform the perceptron algorithm for classification of the samples
plot_samples_boundaries('ω1', 'ω2') # seperate classes ω1, ω2
plot_samples_boundaries('ω2', 'ω3') # seperate classes ω2, ω3
plot_samples_boundaries('ω3', 'ω4') # seperate classes ω3, ω4

import numpy as np
import matplotlib.pyplot as plt

# Define the samples given for the exercise
samples = {
    'ω1': np.array([[0.42, -0.087, 0.58], [-0.2, -3.3, -3.4], [1.3, -0.32, 1.7], [0.39, 0.71, 0.23], [-1.6, -5.3, -0.15],
                    [-0.029, 0.89, -4.7], [-0.23, 1.9, 2.2], [0.27, -0.3, -0.87], [-1.9, 0.76, -2.1], [0.87, -1, -2.6]]),
    'ω2': np.array([[-0.4, 0.58, 0.089], [-0.31, 0.27, -0.04], [0.38, 0.055, -0.035], [-0.15, 0.53, 0.011], [-0.35, 0.47, 0.034],
                    [0.17, 0.69, 0.1], [-0.011, 0.55, -0.18], [-0.27, 0.61, 0.12], [-0.065, 0.49, 0.0012], [-0.12, 0.054, -0.063]])
}

# calculate the mean and the variance from the input data
def MLE_calculation(input_data):
    mean = np.mean(input_data)
    variance = np.var(input_data)
    return mean,variance

# calculate the mean and the covariance  of the input data of 2 Dimensions
def MLE_2D_calculation(input_data):
    input_array = np.array(input_data).T    # convert the matrix into an array for easier covariance calculation

    mean_x1 = np.mean(input_data[0],axis=0) # mean of feature x1
    mean_x2 = np.mean(input_data[1],axis=0) # mean of feature x2
    mean = [mean_x1, mean_x2] 
    
    # we use bias = true because we are calculating the biased covariance since we 
    # are estimating its value with the ML estimation
    # The value of the estimation for the covariance is different from the real one
    covariance = np.cov(input_array,rowvar=False,bias=True) # covariance of x1 - x2

    return mean,covariance

# calculate the mean and the covariance  of the input data of 3 Dimensions
def MLE_3D_calculation(input_data):
    input_array = np.array(input_data).T    # convert the matrix into an array for easier covariance calculation

    mean_x1 = np.mean(input_data[0],axis=0) # mean of feature x1
    mean_x2 = np.mean(input_data[1],axis=0) # mean of feature x2
    mean_x3 = np.mean(input_data[2],axis=0) # mean of feature x3
    mean = [mean_x1, mean_x2, mean_x3]
    covariance = np.cov(input_array,rowvar=False,bias=True) # covariance of x1 - x2 - x3

    return mean,covariance


###### exercise a ######


class_omega_1 = samples['ω1']

feature_x1 = class_omega_1[:,0]
feature_x2 = class_omega_1[:,1]
feature_x3 = class_omega_1[:,2]

# for feature x1
mean_x1,variance_x1 = MLE_calculation(feature_x1)
print(f"Mean: {mean_x1} Variance: {variance_x1}")

# for feature x2
mean_x2,variance_x2 = MLE_calculation(feature_x2)
print(f"Mean: {mean_x2} Variance: {variance_x2}")

# for feature x3
mean_x3,variance_x3 = MLE_calculation(feature_x3)
print(f"Mean: {mean_x3} Variance: {variance_x3}")


###### exercise b ######
feature_x1_x2 = [class_omega_1[:,0],class_omega_1[:,1]]     # features x1 - x2
feature_x2_x3 = [class_omega_1[:,1],class_omega_1[:,2]]     # features x2 - x3
feature_x1_x3 = [class_omega_1[:,0],class_omega_1[:,2]]     # features x1 - x3

mean_x1_x2,variance_x1_x2 = MLE_2D_calculation(feature_x1_x2)  # means and covariance of x1 - x2
print(f"\n2D X1-X2 ->\nMean : {mean_x1_x2}\nCovariance:\n{variance_x1_x2}\n") 

mean_x2_x3,variance_x2_x3 = MLE_2D_calculation(feature_x2_x3)  # means and covariance of x2 - x3
print(f"\n2D X2-X3 ->\nMean : {mean_x2_x3}\nCovariance:\n{variance_x2_x3}\n")

mean_x1_x3,variance_x1_x3 = MLE_2D_calculation(feature_x1_x3)  # means and covariance of x1 - x3
print(f"\n2D X1-X3 ->\nMean : {mean_x1_x3}\nCovariance:\n{variance_x1_x3}\n")


###### exercise c ######
feature_x1_x2_x3 = [class_omega_1[:,0],class_omega_1[:,1],class_omega_1[:,2]] # features x1 - x2 - x3

mean_x1_x2_x3,variance_x1_x2_x3 = MLE_3D_calculation(feature_x1_x2_x3)        # means and covariance of x1 - x2 - x3
print(f"3D X1-X2-X3 ->\nMean : {mean_x1_x2_x3}\nCovariance:\n{variance_x1_x2_x3}")


###### exercise d ######
class_omega_2 = samples['ω2']       # features of class ω2 

feature_x1 = class_omega_2[:,0]     # features x1 - x2
feature_x2 = class_omega_2[:,1]     # features x2 - x3
feature_x3 = class_omega_2[:,2]     # features x1 - x3

# for feature x1
mean_x1,variance_x1 = MLE_calculation(feature_x1)
print(f"Mean: {mean_x1} Variance: {variance_x1}")

# for feature x2
mean_x2,variance_x2 = MLE_calculation(feature_x2)
print(f"Mean: {mean_x2} Variance: {variance_x2}")

# for feature x3
mean_x3,variance_x3 = MLE_calculation(feature_x3)
print(f"Mean: {mean_x3} Variance: {variance_x3}\n")

mean_x1_x2_x3 = [mean_x1, mean_x2, mean_x3]                 # means
variances_matrix = [variance_x1, variance_x2, variance_x3]  # matrix of variances
covariance_omega_2 = np.diag(variances_matrix)              # covariance matrix

print(f"3D - ω2 X1-X2-X3 ->\nMean : {mean_x1_x2_x3}\nCovariance:\n{covariance_omega_2}")

  
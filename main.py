import cv2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

colors = [
    (255, 0, 0),   # red
    (0, 255, 0),  # green
    (0, 0, 255),   # blue
    (255, 255, 0), # yellow
    (255, 0, 255), # magenta
]

# Convert jpg file to matrix of values
def image_to_matrix(image_file):
    img= cv2.imread(image_file)
    return img

# Flatten 3D to 2D matrix, i.e (height, width, depth)=((height*width), depth)
def flatten_matrix(image_matrix):
    if(len(image_matrix.shape)==3):
        height, width, depth= image_matrix.shape
    else:
        height, width= image_matrix.shape
        depth= 1
    flattened_values= np.zeros((height*width, depth), dtype=np.uint8)
    for i,r in enumerate(image_matrix):
        for j,c in enumerate(r):
            flattened_values[i*width+j,:]= c
    return flattened_values

# Perform foreground- background extraction
def foreground_background(mask, image, height, width):
    foreground = np.uint8(np.multiply(mask, image))
    background = np.uint8(image-foreground)

    # Unflatten all the image matrices
    mask = mask.reshape((height, width))
    background = background.reshape((height, width, 3))
    foreground = foreground.reshape((height, width, 3))

    # Convert BGR to RGB
    foreground= cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    background= cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    # Save images
    plt.imsave("_mask.png", mask, cmap=cm.gray)
    plt.imsave("_background.png", background)
    plt.imsave("_foreground.png", foreground)

# Perform image segmentation
def segment(mask, image, height, width):
    # Unflatten all the image matrices
    mask = mask.reshape((height, width))
    image = image.reshape((height, width, 3))

    segmented_map = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            segmented_map[i,j,:] = np.asarray(colors[mask[i][j]]) / 255.0

    # Save images
    plt.imsave("_segmented.png", segmented_map)

class GMM:
    """
    This class defines a Gaussian Mixture Model object.
    Args:
        data(np array): input data, dimension: N*D where N is number of data points and D is the number of features
        ncomp(int): Number of gaussian components

    Attributes:
        mu(np array): means, dimension: ncomp*D
        sigma(np array): covariance matrix, dimension: ncomp*D*D
        weights(np array): mixing coefficients, dimension:ncomp
    """
    def __init__(self, data, ncomp):
        self.data= data
        self.K= ncomp
        self.N= data.shape[0]
        self.D= data.shape[1]

            ## UPGRADE: Can initialize using K clustering for better convergence
        # Initialize mu and sigma
        self.mu= np.array([self.data[0], self.data[1], self.data[self.N//2]])

        self.sigma= np.zeros((self.K, self.D, self.D), dtype=float)
        cov= np.cov(self.data, rowvar=False)
        for i in range(self.K):
            self.sigma[i,:,:]= cov
        
        # Initialize weights with uniform distribution
        self.weights= np.random.rand(self.K)

        # Update normal distributions
        self.__update_distributions()

        self.log_likelihood= 0
        
    def __update_distributions(self):
        self.distributions=[]
        # print(self.sigma.shape, self.mu.shape)
        for i in range(self.K):
            # print(i)
            self.distributions.append(multivariate_normal(self.mu[i], self.sigma[i]))

    def __update_responsibilities(self):
        """
        Assigns every data point with probabilities of belonging to each cluster 

        Attributes:
        resp(np array): responsibilities of each class for each data point, dimension: N*ncomp
        total_resp(np array): total responsibilities of all classes, dimension: ncomp*1
        """
        resp= np.zeros((self.N,self.K), dtype= float)
        for k in range(self.K):
            resp[:,k]= self.weights[k]*self.distributions[k].pdf(self.data)
        l1_norm= np.sum(resp, axis=1)
        l1_norm = l1_norm.reshape(l1_norm.shape[0], 1)
        self.resp= resp/ l1_norm
        # print(self.resp.shape)

        self.totalresp= np.sum(self.resp, axis=0)
        self.totalresp= self.totalresp.reshape(self.totalresp.shape[0], -1)

    def __update_mu(self):
        self.mu= np.divide(np.matmul(np.transpose(self.resp), self.data), self.totalresp)

    def __update_sigma(self):
        for k in range(self.K):
            data_shifted= np.subtract(self.data,self.mu[k,:])
            resp = self.resp[:, k]
            resp = resp.reshape(resp.shape[0], -1)
            cov= np.divide(np.matmul(np.transpose(np.multiply(resp,data_shifted)), data_shifted), self.totalresp[k])
            self.sigma[k,:,:]= cov
            # print(data_shifted.shape)
    
    def __update_weights(self):
        self.weights= np.divide(self.totalresp,self.N)

    def __compute_log_likelihood(self):
        self.old_log_likelihood= self.log_likelihood

        probs= np.zeros((self.N,self.K), dtype= float)
        for k in range(self.K):
            probs[:,k]= self.weights[k]*self.distributions[k].pdf(self.data)
        
        # print(probs.shape)
        self.log_likelihood= np.sum(np.log(np.sum(probs, axis=1)), axis=0)

    def train(self, max_iters):
        for i in range(max_iters):
            # E step
            self.__update_responsibilities()

            # M step
            self.__update_mu()
            self.__update_sigma()
            self.__update_weights()

            # Evaluate log log_likelihood
            self.__update_distributions()
            self.__compute_log_likelihood()
            print('Iteration {}: Log Likelihood = {}'.format(i+1, self.log_likelihood))

            # Check for convergence
                ## UPGRADE: can have a different criteria like check if params are within a threshold
            if abs(self.log_likelihood-self.old_log_likelihood) <= 1e-9:
                return

def main():
    # Load data
    image_file= './input_images/deer.jpg'
    values= image_to_matrix(image_file)
    height, width, depth= values.shape
    values_vector= flatten_matrix(values)
    # print(values.shape, values_vector.shape)

    # Fit GMM model
        ## Tunable params- num_components, iterations
    num_components= 3
    iterations= 500
    gmm= GMM(values_vector, num_components)
    gmm.train(iterations)

    # Generate mask from GMM fit
    mask= np.argmax(gmm.resp, axis=1)
    # print(mask.shape)
    mask= mask.reshape(mask.shape[0], 1)
    # print(mask.shape)

    # Perform foreground-background extraction
    # foreground_background(mask, values_vector, height, width)

    # Perform image segmentation
    segment(mask, values_vector, height, width)

if __name__ == "__main__":
    main()

#################################
# FNNs Classification Paper     #
#                               #
# Example 3 code for JSS paper  #
#                               #
# Barinder Thind, Jiguo Cao     #
#################################

# Libraries
library(FNN)
library(fda)
library(keras)
library(ggplot2)
library(refund)
library(modEvA)

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1)
use_session_with_seed(
  1,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Reading in data
mnist_train = read.csv("fashion-mnist_train.csv", as.is = T, header = T)
mnist_test = read.csv("fashion-mnist_test.csv", as.is = T, header = T)

# obs
obs = 1000

# Making classification bins
mnist_resp = as.factor(mnist_train[1:obs,1])

# define the time points on which the functional predictor is observed. 
pixels = seq(1, 784, 1)

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1, 784), nbasis)

# convert the functional predictor into a fda object
mnist_fd =  Data2fd(pixels, t(mnist_train[1:obs, -1]), spline_basis)
mnist_deriv1 = deriv.fd(mnist_fd)
mnist_deriv2 = deriv.fd(mnist_deriv1)

# Testing with bike data
func_cov_1 = mnist_fd$coefs
func_cov_2 = mnist_deriv1$coefs
func_cov_3 = mnist_deriv2$coefs
mnist_data = array(dim = c(nbasis, obs, 3))
mnist_data[,,1] = func_cov_1
mnist_data[,,2] = func_cov_2
mnist_data[,,3] = func_cov_3

# Indices
ind = sample(1:obs, floor(0.5*obs))

# Splitting response
train_y = mnist_resp[ind]
test_y = mnist_resp[-ind]

# Setting up for FNN
mnist_data_train = array(dim = c(nbasis, length(ind), 3))
mnist_data_test = array(dim = c(nbasis, obs - length(ind), 3))
mnist_data_train = mnist_data[, ind, ]
mnist_data_test = mnist_data[, -ind, ]

# Running FNN for bike
mnist_example <- fnn.fit(resp = train_y, 
                            func_cov = mnist_data_train, 
                            scalar_cov = NULL,
                            basis_choice = c("bspline"), 
                            num_basis = c(5),
                            hidden_layers = 2,
                            neurons_per_layer = c(64, 64),
                            activations_in_layers = c("relu", "linear"),
                            domain_range = list(c(1, 784)),
                            epochs = 300,
                            learn_rate = 0.00001,
                            early_stopping = T)

# Predicting
mnist_pred = fnn.predict(mnist_example,
                            mnist_data_test, 
                            scalar_cov = NULL,
                            basis_choice = c("bspline"), 
                            num_basis = c(5),
                            domain_range = list(c(1, 784)))

# Rounding predictions (they are probabilities)
rounded_preds = as.factor(apply(mnist_pred, 1, function(x){return(which.max(x) - 1)}))

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))


# Libraries
library(FNN)
library(fda)
library(keras)
library(ggplot2)
library(refund)
library(modEvA)
library(fda.usc)
library(dplyr)

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

# Loading data
yoga_test = read.table("yoga/Yoga_TEST.txt", 
                    header=F,
                    as.is = T)

yoga_train = read.table("yoga/Yoga_TRAIN.txt", 
                       header=F,
                       as.is = T)

str(yoga_test)
str(yoga_train)

# obs
obs = 300

# Making classification bins
yoga_resp = as.factor(yoga_train[1:obs,1] - 1)

# define the time points on which the functional predictor is observed. 
pixels = seq(1, 426, 1)

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1, 426), nbasis)

# convert the functional predictor into a fda object
yoga_fd =  Data2fd(pixels, t(yoga_train[1:obs, -1]), spline_basis)
yoga_deriv1 = deriv.fd(yoga_fd)
yoga_deriv2 = deriv.fd(yoga_deriv1)

# Plotting
plot(yoga_fd)

# Testing with bike data
func_cov_1 = yoga_fd$coefs
func_cov_2 = yoga_deriv1$coefs
func_cov_3 = yoga_deriv2$coefs
yoga_data = array(dim = c(nbasis, obs, 3))
yoga_data[,,1] = func_cov_1
yoga_data[,,2] = func_cov_2
yoga_data[,,3] = func_cov_3

# Indices
ind = sample(1:obs, floor(0.5*obs))

# Splitting response
train_y = yoga_resp[ind]
test_y = yoga_resp[-ind]

# Setting up for FNN
yoga_data_train = array(dim = c(nbasis, length(ind), 3))
yoga_data_test = array(dim = c(nbasis, obs - length(ind), 3))
yoga_data_train = yoga_data[, ind, ]
yoga_data_test = yoga_data[, -ind, ]

# Running FNN for bike
yoga_example <- fnn.fit(resp = train_y, 
                         func_cov = yoga_data_train, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(3),
                         hidden_layers = 4,
                         neurons_per_layer = c(64, 64, 64, 64),
                         activations_in_layers = c("relu", "relu", "relu", "linear"),
                         domain_range = list(c(1, 426)),
                         epochs = 300,
                         learn_rate = 0.0001,
                         early_stopping = T,
                         dropout = T)

# Predicting
yoga_pred = fnn.predict(yoga_example,
                         yoga_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(3),
                         domain_range = list(c(1, 426)))

# Rounding predictions (they are probabilities)
rounded_preds = as.factor(apply(yoga_pred, 1, function(x){return(which.max(x) - 1)}))

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))

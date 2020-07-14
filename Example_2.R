#################################
# FNNs Classification Paper     #
#                               #
# Example 2 code for JSS paper  #
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

# Loading data
load("OJ.RData")

# Combining data
OJ$full_resp = c(OJ$y.learning, OJ$y.test)
OJ$full_df = rbind(OJ$x.learning, OJ$x.test)

# Making classification bins
OJ_resp = as.factor(ifelse(OJ$full_resp > 40, 1, 0))
#table(OJ_resp)

# define the time points on which the functional predictor is observed. 
timepts = seq(1, 700, 1)

# define the fourier basis 
nbasis = 31
spline_basis = create.bspline.basis(c(1, 700), nbasis, norder = 4)

# convert the functional predictor into a fda object
OJ_fd =  Data2fd(timepts, t(OJ$full_df), spline_basis)
OJ_deriv1 = deriv.fd(OJ_fd)

# Testing with bike data
func_cov_1 = OJ_fd$coefs
func_cov_2 = OJ_deriv1$coefs
OJ_data = array(dim = c(nbasis, nrow(OJ$full_df), 2))
OJ_data[,,1] = func_cov_1
OJ_data[,,2] = func_cov_2

# Indices
ind = sample(1:nrow(OJ$full_df), floor(0.5*nrow(OJ$full_df)))

# Splitting response
train_y = OJ_resp[ind]
test_y = OJ_resp[-ind]

# Setting up for FNN
OJ_data_train = array(dim = c(nbasis, length(ind), 2))
OJ_data_test = array(dim = c(nbasis, nrow(OJ$full_df) - length(ind), 2))
OJ_data_train = OJ_data[, ind, ]
OJ_data_test = OJ_data[, -ind, ]

# Running FNN for bike
OJ_example <- fnn.fit(resp = train_y, 
                        func_cov = OJ_data_train, 
                        scalar_cov = NULL,
                        basis_choice = c("fourier", "fourier"), 
                        num_basis = c(7, 9),
                        hidden_layers = 4,
                        neurons_per_layer = c(64, 64, 64, 64),
                        activations_in_layers = c("relu", "relu", "relu", "linear"),
                        domain_range = list(c(1, 700), c(1, 700)),
                        epochs = 300,
                        learn_rate = 0.00001,
                        early_stopping = T,
                        dropout = T)

# Predicting
OJ_pred = fnn.predict(OJ_example,
                        OJ_data_test, 
                        scalar_cov = NULL,
                        basis_choice = c("fourier", "fourier"), 
                        num_basis = c(7, 9),
                        domain_range = list(c(1, 700), c(1, 700)))

# Rounding predictions (they are probabilities)
rounded_preds = ifelse(round(OJ_pred)[,2] == 1, 1, 0)

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))

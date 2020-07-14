#################################
# FNNs Classification Paper     #
#                               #
# Example 1 code for JSS paper  #
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
load("Wine.RData")

# Combining data
Wine$full_resp = c(Wine$y.learning, Wine$y.test)
Wine$full_df = rbind(Wine$x.learning, Wine$x.test)

# Making classification bins
wine_resp = as.factor(ifelse(Wine$full_resp > 12, 1, 0))

# define the time points on which the functional predictor is observed. 
timepts = seq(1, 256, 1)

# define the fourier basis 
nbasis = 71
spline_basis = create.bspline.basis(c(1, 256), nbasis, norder = 4)

# convert the functional predictor into a fda object
wine_fd =  Data2fd(timepts, t(Wine$full_df), spline_basis)
wine_deriv1 = deriv.fd(wine_fd)

# Testing with bike data
func_cov_1 = wine_fd$coefs
func_cov_2 = wine_deriv1$coefs
wine_data = array(dim = c(nbasis, nrow(Wine$full_df), 1))
wine_data[,,1] = func_cov_1
#wine_data[,,2] = func_cov_2

# Indices
ind = sample(1:nrow(Wine$full_df), floor(0.8*nrow(Wine$full_df)))

# Splitting response
train_y = wine_resp[ind]
test_y = wine_resp[-ind]

# Setting up for FNN
wine_data_train = array(dim = c(nbasis, length(ind), 1))
wine_data_test = array(dim = c(nbasis, nrow(Wine$full_df) - length(ind), 1))
wine_data_train[,,1] = wine_data[, ind, ]
wine_data_test[,,1] = wine_data[, -ind, ]

# Running FNN for bike
wine_example <- fnn.fit(resp = train_y, 
                        func_cov = wine_data_train, 
                        scalar_cov = NULL,
                        basis_choice = c("fourier"), 
                        num_basis = c(5),
                        hidden_layers = 2,
                        neurons_per_layer = c(64, 64),
                        activations_in_layers = c("relu", "relu"),
                        domain_range = list(c(0, 1)),
                        epochs = 300,
                        learn_rate = 0.0000001,
                        early_stopping = T)

# Predicting
wine_pred = fnn.predict(wine_example,
                        wine_data_test, 
                        scalar_cov = NULL,
                        basis_choice = c("fourier"), 
                        num_basis = c(5),
                        domain_range = list(c(0, 1)))

# Rounding predictions (they are probabilities)
rounded_preds = ifelse(round(wine_pred)[,2] == 1, 1, 0)

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))

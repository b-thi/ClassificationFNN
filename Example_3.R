#################################
# FNNs Classification Paper     #
#                               #
# Example 3 code for JSS paper  #
#                               #
# Barinder Thind, Jiguo Cao     #
#################################

# Libraries
library(FuncNN)
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


# Obtaining response
octane = range01(gasoline[,1])

# Making classification bins
octane_resp = as.factor(ifelse(octane > 0.5, 1, 0))
#table(octane_resp)

# define the time points on which the functional predictor is observed. 
timepts = seq(900, 1700, 2)

# define the fourier basis 
nbasis = 65
spline_basis = create.fourier.basis(c(900, 1700), nbasis)

# convert the functional predictor into a fda object
gasoline_fd =  Data2fd(timepts, t(gasoline[,-1]), spline_basis)
gasoline_deriv1 = deriv.fd(gasoline_fd)
gasoline_deriv2 = deriv.fd(gasoline_deriv1)

# Testing with bike data
func_cov_1 = gasoline_fd$coefs
func_cov_2 = gasoline_deriv1$coefs
func_cov_3 = gasoline_deriv2$coefs
gasoline_data = array(dim = c(nbasis, 60, 3))
gasoline_data[,,1] = func_cov_1
gasoline_data[,,2] = func_cov_2
gasoline_data[,,3] = func_cov_3

# Indices
ind = sample(1:nrow(gasoline), floor(0.5*nrow(gasoline)))

# Splitting response
train_y = octane_resp[ind]
test_y = octane_resp[-ind]

# Setting up for FNN
gasoline_data_train = array(dim = c(nbasis, length(ind), 3))
gasoline_data_test = array(dim = c(nbasis, nrow(gasoline) - length(ind), 3))
gasoline_data_train = gasoline_data[, ind, ]
gasoline_data_test = gasoline_data[, -ind, ]

# Running FNN for bike
gasoline_example <- fnn.fit(resp = train_y, 
                            func_cov = gasoline_data_train, 
                            scalar_cov = NULL,
                            basis_choice = c("bspline", "bspline", "bspline"), 
                            num_basis = c(5, 7, 9),
                            hidden_layers = 2,
                            neurons_per_layer = c(64, 64),
                            activations_in_layers = c("relu", "sigmoid"),
                            domain_range = list(c(900, 1700), c(900, 1700), c(900, 1700)),
                            epochs = 250,
                            learn_rate = 0.00325,
                            early_stopping = T,
                            dropout = T)

# Predicting
gasoline_pred = fnn.predict(gasoline_example,
                            gasoline_data_test, 
                            scalar_cov = NULL,
                            basis_choice = c("bspline", "bspline", "bspline"), 
                            num_basis = c(5, 7, 9),
                            domain_range = list(c(900, 1700), c(900, 1700), c(900, 1700)))

# Rounding predictions (they are probabilities)
rounded_preds = ifelse(round(gasoline_pred)[,2] == 1, 1, 0)

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))

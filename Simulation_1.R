# Libraries
library(FNN)
library(fda)
library(keras)
library(ggplot2)
library(refund)
library(modEvA)
library(fda.usc)

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
data(phoneme)

# Organizing data
phoneme_full = rbind(phoneme$learn$data, phoneme$test$data)

phoneme$learn

# Obtaining response
phoneme_resp = as.factor(c(phoneme$classlearn, phoneme$classtest))

# define the time points on which the functional predictor is observed. 
timepts = seq(1, 150, 1)

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1, 150), nbasis)

# convert the functional predictor into a fda object
phoneme_fd =  Data2fd(timepts, t(phoneme_full), spline_basis)
phoneme_deriv1 = deriv.fd(phoneme_fd)
phoneme_deriv2 = deriv.fd(phoneme_deriv1)

# Testing with bike data
func_cov_1 = phoneme_fd$coefs
func_cov_2 = phoneme_deriv1$coefs
func_cov_3 = phoneme_deriv2$coefs
phoneme_data = array(dim = c(nbasis, nrow(phoneme_full), 3))
phoneme_data[,,1] = func_cov_1
phoneme_data[,,2] = func_cov_2
phoneme_data[,,3] = func_cov_3

# Indices
ind = sample(1:nrow(phoneme_full), floor(0.5*nrow(phoneme_full)))

# Splitting response
train_y = phoneme_resp[ind]
test_y = phoneme_resp[-ind]

# Setting up for FNN
phoneme_data_train = array(dim = c(nbasis, length(ind), 3))
phoneme_data_test = array(dim = c(nbasis, nrow(phoneme_full) - length(ind), 3))
phoneme_data_train = phoneme_data[, ind, ]
phoneme_data_test = phoneme_data[, -ind, ]

# Running FNN for bike
phoneme_example <- fnn.fit(resp = train_y, 
                            func_cov = phoneme_data_train, 
                            scalar_cov = NULL,
                            basis_choice = c("bspline"), 
                            num_basis = c(7),
                            hidden_layers = 5,
                            neurons_per_layer = c(64, 64, 64, 64, 64),
                            activations_in_layers = c("sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"),
                            domain_range = list(c(1, 150)),
                            epochs = 300,
                            learn_rate = 0.00001,
                            early_stopping = T,
                            dropout = T)

# Predicting
phoneme_pred = fnn.predict(phoneme_example,
                            phoneme_data_test, 
                            scalar_cov = NULL,
                            basis_choice = c("bspline"), 
                            num_basis = c(7),
                            domain_range = list(c(1, 150)))

# Rounding predictions (they are probabilities)
rounded_preds = as.factor(apply(phoneme_pred, 1, function(x){return(which.max(x) - 1)}))

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))

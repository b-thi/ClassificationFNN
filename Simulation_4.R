# Libraries
library(FuncNN)
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
wine_test = read.table("wine/wine_TEST.txt", 
                       header=F,
                       as.is = T)

wine_train = read.table("wine/wine_TRAIN.txt", 
                        header=F,
                        as.is = T)

str(wine_test)
str(wine_train)

# Putting together
wine_ds = rbind(wine_train, wine_test)

# obs
obs = 111

# Making classification bins
wine_resp = as.factor(wine_ds[1:obs,1] - 1)

# define the time points on which the functional predictor is observed. 
pixels = seq(1, 234, 1)

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1, 234), nbasis)

# Scaling data
#wine_ds = scale(wine_ds[1:obs, -1])
#wine_ds = range01(wine_ds[1:obs, -1])
#wine_ds = apply(wine_ds[1:obs, -1], 2, range01)

# convert the functional predictor into a fda object
wine_fd =  Data2fd(pixels, t(wine_ds[1:obs, -1]), spline_basis)
wine_deriv1 = deriv.fd(wine_fd)
wine_deriv2 = deriv.fd(wine_deriv1)

# Plotting
plot(wine_fd)

# Testing with bike data
func_cov_1 = wine_fd$coefs
func_cov_2 = wine_deriv1$coefs
func_cov_3 = wine_deriv2$coefs
wine_data = array(dim = c(nbasis, obs, 3))
wine_data[,,1] = func_cov_1
wine_data[,,2] = func_cov_2
wine_data[,,3] = func_cov_3

# Indices
ind = sample(1:obs, floor(0.5*obs))

# Splitting response
train_y = wine_resp[ind]
test_y = wine_resp[-ind]

# Setting up for FNN
wine_data_train = array(dim = c(nbasis, length(ind), 3))
wine_data_test = array(dim = c(nbasis, obs - length(ind), 3))
wine_data_train = wine_data[, ind, ]
wine_data_test = wine_data[, -ind, ]

# Running FNN for bike
wine_example <- fnn.fit(resp = train_y, 
                        func_cov = wine_data_train, 
                        scalar_cov = NULL,
                        basis_choice = c("fourier", "fourier", "fourier"), 
                        num_basis = c(5, 7, 9),
                        hidden_layers = 3,
                        neurons_per_layer = c(128, 128, 32),
                        activations_in_layers = c("relu", "relu", "relu"),
                        domain_range = list(c(1, 234), c(1, 234), c(1, 234)),
                        epochs = 300,
                        learn_rate = 0.001,
                        early_stopping = T,
                        dropout = T)

# Predicting
wine_pred = fnn.predict(wine_example,
                        wine_data_test, 
                        scalar_cov = NULL,
                        basis_choice = c("fourier", "fourier", "fourier"), 
                        num_basis = c(5, 7, 9),
                        domain_range = list(c(1, 234), c(1, 234), c(1, 234)))

# Rounding predictions (they are probabilities)
rounded_preds = as.factor(apply(wine_pred, 1, function(x){return(which.max(x) - 1)}))

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))

# Comparing with regression model
wine_full = data.frame(rbind(wine_train, wine_test))
wine_full$resp = wine_resp
wine_MV_train = wine_full[ind,]
wine_MV_test = wine_full[-ind,]

# Running regression
fit_rf = randomForest::randomForest(resp ~ ., data = wine_MV_train)
rf_pred = predict(fit_rf, newdata = wine_MV_test, type = "response")

# Confusion matrix
caret::confusionMatrix(rf_pred, as.factor(test_y))

# Libraries
library(FuncNN)
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

# Obtaining response
phoneme_resp = as.factor(c(phoneme$classlearn, phoneme$classtest) - 1)

# define the time points on which the functional predictor is observed. 
timepts = seq(1, 150, 1)

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1, 150), nbasis)

# convert the functional predictor into a fda object
phoneme_fd =  Data2fd(timepts, t(phoneme_full), spline_basis)
phoneme_deriv1 = deriv.fd(phoneme_fd)
phoneme_deriv2 = deriv.fd(phoneme_deriv1)

plot(phoneme_fd)

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
phoneme_data_train = array(dim = c(nbasis, length(ind), 1))
phoneme_data_test = array(dim = c(nbasis, nrow(phoneme_full) - length(ind), 3))
phoneme_data_train = phoneme_data[, ind, ]
phoneme_data_test = phoneme_data[, -ind, ]

# Running FNN for bike
phoneme_example <- fnn.fit(resp = train_y, 
                            func_cov = phoneme_data_train, 
                            scalar_cov = NULL,
                            basis_choice = c("fourier", "fourier", "fourier"), 
                            num_basis = c(3, 5, 7),
                            hidden_layers = 3,
                            neurons_per_layer = c(128, 128, 32),
                            activations_in_layers = c("relu", "relu", "relu"),
                            domain_range = list(c(1, 150), c(1, 150), c(1, 150)),
                            epochs = 300,
                            learn_rate = 0.0001,
                            early_stopping = T,
                            dropout = F)

# Predicting
phoneme_pred = fnn.predict(phoneme_example,
                            phoneme_data_test, 
                            scalar_cov = NULL,
                            basis_choice = c("fourier", "fourier", "fourier"), 
                            num_basis = c(3, 5, 7),
                            domain_range = list(c(1, 150), c(1, 150), c(1, 150)))

# Rounding predictions (they are probabilities)
rounded_preds = as.factor(apply(phoneme_pred, 1, function(x){return(which.max(x) - 1)}))

# Confusion matrix
caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))


# Comparing with regression model
phoneme_full = data.frame(rbind(phoneme$learn$data, phoneme$test$data))
phoneme_full$resp = as.factor(phoneme_resp)
phoneme_MV_train = phoneme_full[ind,]
phoneme_MV_test = phoneme_full[-ind,]

# Running regression
fit_rf = randomForest::randomForest(resp ~ ., data = phoneme_MV_train)
rf_pred = predict(fit_rf, newdata = phoneme_MV_test, type = "response")

# Confusion matrix
caret::confusionMatrix(rf_pred, as.factor(test_y))

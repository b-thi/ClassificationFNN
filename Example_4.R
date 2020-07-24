#################################
# FNNs Classification Paper     #
#                               #
# Example 2 code for JSS paper  #
#                               #
# Barinder Thind, Jiguo Cao     #
#################################

# Libraries
library(fda)
library(keras)
library(ggplot2)
library(refund)
library(modEvA)
library(fda.usc)
source("fnn_preprocess.R")

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
phoneme_resp = as.vector(c(phoneme$classlearn, phoneme$classtest)) - 1

# define the time points on which the functional predictor is observed. 
timepts = seq(1, 150, 1)

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1, 150), nbasis)

# convert the functional predictor into a fda object
phoneme_fd =  Data2fd(timepts, t(phoneme_full), spline_basis)
phoneme_deriv1 = deriv.fd(phoneme_fd)
phoneme_deriv2 = deriv.fd(phoneme_deriv1)

# plotting
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

# Now, let's pre-process
phoneme_pre_train = FNN_Preprocess(func_cov = phoneme_data_train,
                                       basis_choice = c("fourier", "fourier", "fourier"),
                                       num_basis = c(3, 5, 7),
                                       domain_range = list(c(1, 150), c(1, 150), c(1, 150)),
                                       covariate_scaling = T,
                                       raw_data = F)

phoneme_pre_test = FNN_Preprocess(func_cov = phoneme_data_test,
                                      basis_choice = c("fourier", "fourier", "fourier"),
                                      num_basis = c(3, 5, 7),
                                      domain_range = list(c(1, 150), c(1, 150), c(1, 150)),
                                      covariate_scaling = T,
                                      raw_data = F)

# Setting up FNN model
model_fnn <- keras_model_sequential()
model_fnn %>% 
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'sigmoid') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'softmax')

# Setting parameters for FNN model
model_fnn %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Early stopping
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 15)

# Training FNN model
model_fnn %>% fit(phoneme_pre_train$data, 
                  train_y, 
                  epochs = 150,  
                  validation_split = 0.2,
                  callbacks = list(early_stop))

# Getting predictions from FNN model
score <- model_fnn %>% keras::evaluate(phoneme_pre_test$data, test_y)
cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

# Predictions
test_predictions <- model_fnn %>% predict(phoneme_pre_test$data)
preds = apply(test_predictions, 1, function(x){return(which.max(x))}) - 1

# Plotting
caret::confusionMatrix(as.factor(test_y), as.factor(preds))

# Predictions
train_predictions <- model_fnn %>% predict(phoneme_pre_train$data)
preds = apply(train_predictions, 1, function(x){return(which.max(x))}) - 1

# Plotting
caret::confusionMatrix(as.factor(train_y), as.factor(preds))


############# FDA EXAMPLES ##############

# fData Object
phoneme_fdata = fdata(phoneme_full[,-151], argvals = timepts, rangeval = c(1, 150))

# Test and train
train_x = phoneme_fdata[ind,]
test_x = phoneme_fdata[-ind,]
train_y = phoneme_resp[ind]
test_y = phoneme_resp[-ind]


# Predicting
mlearn <- train_x
mlearn2 <- test_x
glearn <- train_y
out20 = classif.DD(glearn, mlearn, depth="mode", classif="glm")

out21=classif.DD(glearn, 
                 list(mlearn, mlearn2), 
                 depth="modep", 
                 classif="glm", 
                 control=list(draw=F))

out20 # univariate functional data



# Functional Linear Model (Basis)
func_basis = fregre.glm(train_y ~ x, data = ldata, family=binomial())
pred_basis = predict(func_basis[[1]], test_x)

?fregre.basis

# Functional Principal Component Regression (No Penalty)
func_pc = fregre.pc.cv(train_x, train_y, 8)
pred_pc = predict(func_pc$fregre.pc, test_x)

# Functional Principal Component Regression (2nd Deriv Penalization)
func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
pred_pc2 = predict(func_pc2$fregre.pc, test_x)

# Functional Principal Component Regression (Ridge Regression)
func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
pred_pc3 = predict(func_pc3$fregre.pc, test_x)

# Functional Partial Least Squares Regression (No Penalty)
func_pls = fregre.pls(train_x, train_y, 1:6)
pred_pls = predict(func_pls, test_x)

# Functional Partial Least Squares Regression (2nd Deriv Penalization)
func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda = 1:3, P=c(0,0,1))
pred_pls2 = predict(func_pls2$fregre.pls, test_x)

# Functional Non-Parametric Regression
func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
pred_np = predict(func_np, test_x)
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
ht_dat = read.table("HT/HT_Sensor_dataset.dat", 
                    header=TRUE,
                    as.is = T)

ht_meta = read.table("HT/HT_Sensor_metadata.dat",
                     header = T,
                     as.is = T)
  
# Structure
str(ht_dat)
table(ht_dat$id)

# Subsetting to get 
ht_dat %>% 
  select(id, )

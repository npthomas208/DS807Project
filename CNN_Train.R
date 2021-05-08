library(jpeg)
library(ggplot2)
library(EnvStats)
library(MASS)
library(dbscan)
library(factoextra)
library(gam)
library(flexmix)
library(keras)
library(tensorflow)
library(dplyr)
library(stringr)
library(tfdatasets)
library(tfruns)

# Hyperparameter flags ---------------------------------------------------------

FLAGS <- flags(
  flag_numeric("dropout1", 0.4), 
  flag_numeric("dropout2", 0.4),
  flag_integer('kernel1', 3 ),
  flag_integer('dense_units1', 128 )
)

# Data Preparation -------------------------------------------------------------

use_condaenv("r-tensorflow")

nn = list()


nn$csv <- as.list(read.delim("./Images/train.csv") %>%
                    mutate(class = as.factor(class)))

nn$train=list()
nn$train$y = array(as.numeric(nn$csv$class, levels(nn$csv$class))-1, dim = c(length(nn$csv$file), 1))

nn$csv$image = rep(NA, length(nn$csv$file))

for(i in 1:length(nn$csv$file)){
  nn$csv$image[i] = list(as.integer(outer(readJPEG(paste("./Images/Train/", nn$csv$file[i], sep="")),  255)))
}

nn$tarray <- function(x) aperm(x, seq_along(dim(x)))
nn$train$x = nn$tarray(array(unlist(nn$csv$image), c(length(nn$csv$image),1024, 1024, 3)))

nn <- list(train = nn$train)


# Define Model -----------------------------------------------------------------

set.seed(1)
nn$modelCNN = keras_model_sequential() %>%
  layer_conv_2d(filters = 16, kernel_size = c(FLAGS$kernel1,FLAGS$kernel1), activation = "relu", input_shape = c(1024,1024,3)) %>%
  layer_dropout(FLAGS$dropout1) %>%
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = "relu", input_shape = c(1024,1024,3)) %>%
  layer_dropout(FLAGS$dropout2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = "relu") #%>% 


nn$modelDense = nn$modelCNN %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 2, activation = "softmax")

nn$model = nn$modelDense %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

# Training & Evaluation --------------------------------------------------------
set.seed(1)
history <- nn$model %>% 
  keras::fit(
    x = nn$train$x, y = nn$train$y,
    epochs = 10,
    validation_split = 0.15,
    verbose = 2
  )

plot(history)

# test data---------------------------------------------------------------------

nn$csv <- as.list(read.delim("./Images/test.csv", sep= ",") %>%
                    mutate(class = as.factor(class)))

nn$test=list()
nn$test$y = array(as.numeric(nn$csv$class, levels(nn$csv$class))-1, dim = c(length(nn$csv$file), 1))

nn$test$x = rep(NA, length(nn$csv$file))

for (i in 1:length(nn$csv$file)){
  nn$test$x[i] <- list(image_to_array(image_load(paste("./Images/Test/", nn$csv$file[i], sep=""),target_size = c(1024,1024))))
}

nn$test$x <- array(unlist(nn$test$x),c(7,1024,1024,3))

print(max(nn$train$x[1,,,1]))
print(max(nn$test$x[1,,,1]))

nn = list(train = nn$train, test = nn$test, model = nn$model)

nn$test$y

nn$model %>%
  predict_classes(nn$test$x)
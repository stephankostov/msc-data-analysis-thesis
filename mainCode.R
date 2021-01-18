setwd("~/Statistics/Diss")
load("~/Statistics/Diss/.FinalData.RData")

library(tidyverse)
library(ggplot2)
library(caret)
library(gridExtra)
library(scales)
library(splitstackshape)
library(GGally)
library(e1071)

# -----------------
# Initial Loading of Data
# -----------------

# Loading data and removing non-informative columns
load("gaia.RData")
GAIA.full <- data
GAIA <- data[, !(names(data) %in% c('line', 'G.mag', 'normal'))]

# Log-transforming temperature and renaming gravity to avoid confusion 
GAIA[,"temperature"] <- log(GAIA$temperature)
names(GAIA)[3] <- "logt"
names(GAIA)[2] <- "logg"

# ---------------
# Checks
# ----------------

# performing checks on the dataset to ensure there are no mistakes

# checks for missing data
which(is.na(GAIA))
# checking if any predictor contains negative value
apply(GAIA[,-c(1:3)], 2, function(x) which(x<0))

# number of levels in each response variables
length(levels(factor(GAIA$temperature)))
length(levels(factor(GAIA$gravity)))
length(levels(factor(GAIA$metallicity)))

# checking normalisation (should be equal to 1)
sum(GAIA[1,-(1:3)])

# --------------
# Visualisation
# --------------

# Visualising the data before and after transformation, 
# highlighting each variables' distribution, and correlation with others

library(tidyr)
library(ggplot2)

# histogram matrix
GAIA %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

GAIA.transformed %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# pairs matrix
ggpairs(GAIA)

# scatterplot matrix for temperature
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x = GAIA.filtered[,-1],
            y = GAIA.t$logt,
            plot = "scatter",
            layout = c(4,4))

# skewness quantification
skewValues <- apply(GAIA, 2, skewness)
skewValues

# -----------------
# Custom Functions
# ---------------

## function to give standard deviation of variable
std <- function(x) sqrt(var(x)/length(x))


## function giving test set score and diagnostic plots
diagnose <- function(model, PCA=FALSE, plot=FALSE){
  
  # obtaining dataset used 
  if (PCA==FALSE){
    train.data <- train.data
    test.data <- test.data
  } else {
    train.data <- train.data.PCA
    test.data <- test.data.PCA
  }
  
  # obtaining response variable being modelled
  param <- as.character(model$terms[[2]])
  
    ## obtaining residuals, fitted values, predicted values
  
    fitted <- predict(model, train.data)
    res <- train.data[,param] - fitted
    res.std <- res/std(res)
    
    actual <- test.data[,param]
    predictions <- predict(model, test.data)
    RMSE <- RMSE(predictions, actual)
    
    
  if (plot==TRUE){
    # storing values into dataframe
    df <- data.frame(fitted=fitted,
                     resid=res,
                     resid.std=res.std)
    df2 <- data.frame(actual=actual,
                      predictions=predictions)
    
    p1 <- ggplot(df, aes(fitted, resid)) +
      geom_point() +
      geom_hline(yintercept=0, col="red", linetype="dashed")+ 
      xlab("Fitted Values") + ylab("Residuals") + 
      theme_bw() + theme(aspect.ratio = 1)
    p2 <- ggplot(df, aes(sample = resid.std)) +
      stat_qq() + stat_qq_line() + 
      xlab("Theoretical Quantiles") + ylab("Standardized Residuals") + 
      theme_bw() + theme(aspect.ratio=1)
    p3 <- ggplot(df2, aes(actual, predictions)) + 
      geom_point() + 
      xlab("Actual Values") + ylab("Fitted Values") + 
      geom_abline(intercept=0, slope=1 , col="red", linetype="dashed") +
      theme_bw() + theme(aspect.ratio = 1)
    grid.arrange(p1, p2, p3, ncol=2)
  }
  
  return(RMSE)
}

# ----------------------
# PCA 
# -----------------

# performing PCA on the dataset in order to address the problems in multicollinearity
# as well as visualising this

## Principle Component Analysis on predictors
PCA <- prcomp(GAIA[,-(1:3)], center=TRUE, scale=TRUE)
summary(PCA)
GAIA.PCA <- as.data.frame(cbind(GAIA[,1:3], PCA$x))

## Plots

# scree plot
library(factoextra)
screeplot <- fviz_screeplot(PCA)
screeplot

# Biplot
library(ggfortify)
PCA.biplot.t <- autoplot(PCA, data=GAIA.PCA, colour='logt',
         loadings=TRUE, loadings.colour="black",
         loadings.label=TRUE)
PCA.biplot.t

PCA.biplot.g <- autoplot(PCA, data=GAIA.PCA, colour='logg',
                         loadings=TRUE, loadings.colour="black",
                         loadings.label=TRUE)
PCA.biplot.g

PCA.biplot.m <- autoplot(PCA, data=GAIA.PCA, colour='metallicity',
                         loadings=TRUE, loadings.colour="black",
                         loadings.label=TRUE)
PCA.biplot.m

# ---------------------------
#  Regression Models (temperature)
# ---------------

# splitting data 
set.seed(123)
data <- GAIA[,-c(1,2)]
data.PCA <- GAIA.PCA[,-c(1,2)]
training.samples <- data$logt %>%
  createDataPartition(p=0.8,list=FALSE)
train.data <- data[training.samples,]
test.data <- data[-training.samples,]
train.data.PCA <- data.PCA[training.samples,]
test.data.PCA <- data.PCA[-training.samples,]


# start parallel processing cluster
library(doSNOW)
cl <- makeCluster(3, type="SOCK")
registerDoSNOW(cl)

# stop cluster
stopCluster(cl)
registerDoSEQ()

# timing the computations
start.time <- Sys.time()

## SVM Radial Basis Kernel

# first tuning phase
svmGrid <- expand.grid(.sigma=c(0.1,0.5,1),
                              .C=c(2^(seq(1,16,by=2))))
svm.t <- train(logt ~ . , 
                      data=train.data,
                      method="svmRadial",
                      preProc=c("scale", "center"),
                      tuneGrid = svmGrid,
                      eps = 0.01,
                      trControl = trainControl(method = "cv", number=10))

# second tuning phase
Sys.time() - start.time
eps <- c(0.005,0.01,0.02,0.03)
results <- vector("numeric",length=length(eps))
start.time <- Sys.time
for (i in 1:length(eps)) {
  
  model <- train(logt ~ . , 
                     data=train.data,
                     method="svmRadial",
                     preProc=c("scale", "center"),
                     tuneGrid = svm.t$bestTune,
                     eps = eps[i],
                     trControl = trainControl(method = "cv", number=10))
  results[i] <- model$results$RMSE
  print(Sys.time() - start.time)
}
svm.eps.t <-  data.frame(eps=eps, RMSE=results)
svm.eps.t <-  rbind(svm.eps.t, data.frame(eps=0.01, RMSE=min(svm.t$results$RMSE))) # binding result from first tuning phase

## SVM Polynomial Kernel
start.time <- Sys.time()
# first tuning phase
svmPolyGrid <- expand.grid(C=c(1,2,3,4),
                            scale=c(0.1,0.5,1.0),
                            degree=(2:3))
svm.poly.t <- train(logt ~ . , 
               data=train.data,
               method="svmPoly",
               preProc=c("scale", "center"),
               tuneGrid = svmPolyGrid,
               eps = 0.05,
               trControl = trainControl(method = "cv", number=10))
Sys.time() - start.time

# second tuning phase
eps <- c(0.02,0.03,0.04,0.06)
results <- vector("numeric",length=length(eps))

for (i in (1:length(eps))){
  model <- train(logt ~ .,
                 data=train.data,
                 method="svmPoly",
                 preProc=c("scale", "center"),
                 tuneGrid=svm.poly.t$bestTune,
                 eps=eps[i],
                 trControl = trainControl(method = "cv", number=10))
  results[i] <- model$results$RMSE
  print(Sys.time())
}
svm.eps.poly.t <-  data.frame(eps=eps, RMSE=results)
svm.eps.poly.t <-  rbind(svm.eps.poly.t, data.frame(eps=0.05, RMSE=min(svm.poly.t$results$RMSE))) # manually adding first tuning value
save.image("~/Statistics/Diss/.DataClean.RData")

# binding result from first tuning phase

## Neural Networks
nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:14),
                        .bag = FALSE)
nnet.t <- train(logt ~ .,
                  data=train.data.PCA,
                  method="avNNet",
                  tuneGrid = nnetGrid,
                  trControl = trainControl(method = "cv", number=10),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 14*(ncol(train.data.PCA)) + 14 + 1,
                  maxit = 500)


## Random Forest
forestGrid <- expand.grid(.mtry=seq(1,13,by=2))
forest.t <- train(logt ~ .,
                    data=train.data,
                    method="rf",
                    tuneGrid=forestGrid,
                    ntrees=4000,
                    trControl=trainControl(method="cv", number=10),
                    importance=TRUE)

## Gradient Boosting
gbmGrid <- expand.grid(shrinkage=c(.01,0.1,.3),
                       interaction.depth = c(1,3,5,7),
                       n.minobsinnode = c(5),
                       n.trees=seq(800,4000,by=400))
bag.fraction <- c(0.3,0.5,1)

gbm.tune.t <-  matrix(NA, ncol=6)
gbm.tune.t <- as.data.frame(gbm.tune.t)
names(gbm.tune.t) <- c("shrinkage", "interaction.depth", "n.trees", "n.minobsinnode", "RMSE", "bag.fraction")

start.time <- Sys.time()

for (i in (1:length(bag.fraction))) {

  gbm.t <- train(logt ~ .,
                 data=train.data,
                 method='gbm',
                 tuneGrid=gbmGrid,
                 trControl=trainControl(method="cv",number=10),
                 bag.fraction=bag.fraction[i],
                 verbose=FALSE)
  
  results <- gbm.t$results[c("shrinkage", "interaction.depth", "n.trees", "n.minobsinnode", "RMSE")]
  results$bag.fraction <- rep(bag.fraction[i],nrow(results))
  gbm.tune.t <- rbind(gbm.tune.t,results)

}
gbm.tune.t <- gbm.tune.t[-1,]


# ---------------------------
#  Regression Models (gravity)
# ---------------

# splitting data 
set.seed(123)
data <- GAIA[,-c(1,3)]
data.PCA <- GAIA.PCA[,-c(1,3)]
training.samples <- data$logg %>%
  createDataPartition(p=0.8,list=FALSE)
train.data <- data[training.samples,]
test.data <- data[-training.samples,]
train.data.PCA <- data.PCA[training.samples,]
test.data.PCA <- data.PCA[-training.samples,]


# start parallel processing cluster
library(doSNOW)
cl <- makeCluster(2, type="SOCK")
registerDoSNOW(cl)

start.time <- Sys.time()

## SVM Radial Basis Kernel

# first tuning phase
svmGrid <- expand.grid(.sigma=c(0.1,0.5,1),
                       .C=c(2^(seq(1,16,by=2))))
svm.g <- train(logg ~ . , 
               data=train.data,
               method="svmRadial",
               preProc=c("scale", "center"),
               tuneGrid = svmGrid,
               eps = 0.1,
               trControl = trainControl(method = "cv", number=10))

# second tuning phase
eps <- c(0.02, 0.04, 0.08, 0.12)
results <- vector("numeric",length=length(eps))
for (i in 1:length(eps)) {
  model <- train(logg ~ . , 
                     data=train.data,
                     method="svmRadial",
                     preProc=c("scale", "center"),
                     tuneGrid = svm.g$bestTune,
                     eps = eps[i],
                     trControl = trainControl(method = "cv", number=10))
  results[i] <- model$results$RMSE
  print(Sys.time() - start.time)
}
svm.eps.g <-  data.frame(eps=eps, RMSE=results)
svm.eps.g <-  rbind(svm.eps.g, data.frame(eps=0.1, RMSE=min(svm.g$results$RMSE))) # binding result from first tuning phase

## SVM Polynomial Kernel

# First tuning phase
svmPolyGrid <- expand.grid(C=c(1,2,3,4),
                           scale=c(0.1,0.5,1.0),
                           degree=(2:4))
svm.poly.g <- train(logg ~ . , 
                    data=train.data,
                    method="svmPoly",
                    preProc=c("scale", "center"),
                    tuneGrid = svmPolyGrid,
                    eps = 0.5,
                    trControl = trainControl(method = "cv", number=10))

# second tuning phase
eps <- c(0.1,0.2,0.3,0.4)
results <- vector("numeric",length=length(eps))

for (i in 1:length(eps)) {
  model <- train(logg ~ . , 
                     data=train.data,
                     method="svmPoly",
                     preProc=c("scale", "center"),
                     tuneGrid = svm.poly.g$bestTune,
                     eps = eps[i],
                     trControl = trainControl(method = "cv", number=10))
  results[i] <- model$results$RMSE
  print(Sys.time() - start.time)
}
svm.eps.poly.g <-  data.frame(eps=eps, RMSE=results)
svm.eps.poly.g <-  rbind(svm.eps.poly.g, data.frame(eps=0.5, RMSE=min(svm.poly.g$results$RMSE))) # binding result from first tuning phase

# final model
svm.poly.f.g <- train(logg ~ . , 
                    data=train.data,
                    method="svmPoly",
                    preProc=c("scale", "center"),
                    tuneGrid = svm.poly.g$bestTune,
                    eps = 0.4,
                    trControl = trainControl(method = "cv", number=10))

## Gradient Boosting
gbmGrid <- expand.grid(shrinkage=c(.01,0.1,.3),
                       interaction.depth = c(1,3,5,7),
                       n.minobsinnode = c(5),
                       n.trees=seq(800,4000,by=400))
bag.fraction <- c(0.3,0.5,1)

gbm.tune.g <-  matrix(NA, ncol=6)
gbm.tune.g <- as.data.frame(gbm.tune.g)
names(gbm.tune.g) <- c("shrinkage", "interaction.depth", "n.trees", "n.minobsinnode", "RMSE", "bag.fraction")

for (i in (1:length(bag.fraction))) {
  
  gbm.g <- train(logg ~ .,
                 data=train.data,
                 method='gbm',
                 tuneGrid=gbmGrid,
                 trControl=trainControl(method="cv",number=10),
                 bag.fraction=bag.fraction[i],
                 verbose=FALSE)
  
  results <- gbm.g$results[c("shrinkage", "interaction.depth", "n.trees", "n.minobsinnode", "RMSE")]
  results$bag.fraction <- rep(bag.fraction[i],nrow(results))
  gbm.tune.g <- rbind(gbm.tune.g,results)
  
}

## Neural Networks
nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:14),
                        .bag = FALSE)
nnet.g <- train(logg ~ .,
                data=train.data.PCA,
                method="avNNet",
                tuneGrid = nnetGrid,
                trControl = trainControl(method = "cv", number=10),
                linout = TRUE,
                trace = FALSE,
                MaxNWts = 14*(ncol(train.data.PCA)) + 14 + 1,
                maxit = 500)


## Random Forest
forestGrid <- expand.grid(.mtry=seq(1,13,by=2))
forest.g <- train(logg ~ .,
                  data=train.data,
                  method="rf",
                  tuneGrid=forestGrid,
                  ntrees=4000,
                  trControl=trainControl(method="cv", number=10),
                  importance=TRUE)

# ---------------------------
#  Regression Models (metallicity)
# ---------------

# splitting data 
set.seed(123)
data <- GAIA[,-c(2,3)]
data.PCA <- GAIA.PCA[,-c(2,3)]
training.samples <- data$metallicity %>%
  createDataPartition(p=0.8,list=FALSE)
train.data <- data[training.samples,]
test.data <- data[-training.samples,]
train.data.PCA <- data.PCA[training.samples,]
test.data.PCA <- data.PCA[-training.samples,]


# start parallel processing cluster
library(doSNOW)
cl <- makeCluster(3, type="SOCK")
registerDoSNOW(cl)

start.time <- Sys.time()

## SVM Radial Basis Kernel

# sigma and cost tuning
svmGrid <- expand.grid(.sigma=c(0.1,0.5,1),
                       .C=c(2^(seq(1,16,by=2))))
svm.m <- train(metallicity ~ . , 
               data=train.data,
               method="svmRadial",
               preProc=c("scale", "center"),
               tuneGrid = svmGrid,
               eps = 0.2,
               trControl = trainControl(method = "cv", number=10))

Sys.time() - start.time

# epsilon tuning
eps <- c(0.2,0.4,0.6,0.8)
results <- vector("numeric",length=length(eps))

for (i in 1:length(eps)) {
  
  svm.eps.m <- train(metallicity ~ . , 
                     data=train.data,
                     method="svmRadial",
                     preProc=c("scale", "center"),
                     tuneGrid = svm.m$bestTune,
                     eps = eps[i],
                     trControl = trainControl(method = "cv", number=10))
  results[i] <- svm.eps.m$results$RMSE
  print(Sys.time() - start.time)
}
svm.eps.m <-  data.frame(eps=eps, RMSE=results)
svm.eps.m <-  rbind(svm.eps.m, data.frame(eps=0.1, RMSE=min(svm.m$results$RMSE))) # binding result from first tuning phase

## SVM Polynomial Kernel
svmPolyGrid <- expand.grid(C=c(1,2,3,4),
                           scale=c(0.1,0.5,1.0),
                           degree=(2:4))
svm.poly.m <- train(metallicity ~ . , 
                    data=train.data,
                    method="svmPoly",
                    preProc=c("scale", "center"),
                    tuneGrid = svmPolyGrid,
                    eps = 0.2,
                    trControl = trainControl(method = "cv", number=10))

# second tuning phase
eps <- c(0.1,0.3,0.4)
results <- vector("numeric",length=length(eps))

for (i in 1:length(eps)) {
  model <- train(metallicity ~ . , 
                 data=train.data,
                 method="svmPoly",
                 preProc=c("scale", "center"),
                 tuneGrid = svm.poly.g$bestTune,
                 eps = eps[i],
                 trControl = trainControl(method = "cv", number=10))
  results[i] <- model$results$RMSE
  print(Sys.time() - start.time)
}
svm.eps.poly.m <-  data.frame(eps=eps, RMSE=results)
svm.eps.poly.m <-  rbind(svm.eps.poly.m, data.frame(eps=0.2, RMSE=min(svm.poly.m$results$RMSE))) # binding result from first tuning phase

## Gradient Boosting
gbmGrid <- expand.grid(shrinkage=c(.01,0.1,.3),
                       interaction.depth = c(1,3,5,7),
                       n.minobsinnode = c(5),
                       n.trees=seq(800,4000,by=400))
bag.fraction <- c(0.3,0.5,1)

gbm.tune.m <-  matrix(NA, ncol=6)
gbm.tune.m <- as.data.frame(gbm.tune.m)
names(gbm.tune.m) <- c("shrinkage", "interaction.depth", "n.trees", "n.minobsinnode", "RMSE", "bag.fraction")

start.time <- Sys.time()

for (i in (1:length(bag.fraction))) {
  
  gbm.m <- train(metallicity ~ .,
                 data=train.data,
                 method='gbm',
                 tuneGrid=gbmGrid,
                 trControl=trainControl(method="cv",number=10),
                 bag.fraction=bag.fraction[i],
                 verbose=FALSE)
  
  results <- gbm.m$results[c("shrinkage", "interaction.depth", "n.trees", "n.minobsinnode", "RMSE")]
  results$bag.fraction <- rep(bag.fraction[i],nrow(results))
  gbm.tune.m <- rbind(gbm.tune.m,results)
  
}

# Neural Networks
nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:14),
                        .bag = FALSE)
nnet.m <- train(metallicity ~ .,
                data=train.data.PCA,
                method="avNNet",
                tuneGrid = nnetGrid,
                trControl = trainControl(method = "cv", number=10),
                linout = TRUE,
                trace = FALSE,
                MaxNWts = 14*(ncol(train.data.PCA)) + 14 + 1,
                maxit = 500)


# Random Forest
forestGrid <- expand.grid(.mtry=seq(1,13,by=2))
forest.m <- train(metallicity ~ .,
                  data=train.data,
                  method="rf",
                  tuneGrid=forestGrid,
                  ntrees=4000,
                  trControl=trainControl(method="cv", number=10),
                  importance=TRUE)

validation.m <- data.frame(RMSE = c(diagnose(knn.m), diagnose(svm.radial.m), diagnose(svm.poly.m),
                                    diagnose(nnetTune.m),diagnose(forestTune.m),diagnose(gbmTune.m)),
                           methods = c("KNN", "SVMr", "SVMp", "Neural Network", "Random Forest", "SGB"))

# ------------
# Plotting tuning 
# ---------------

ggplot(svm.poly.t, highlight=TRUE) + theme_bw()
plot(svm.poly.t, plotType="level")
plot(svm.poly.t, highlight=TRUE)

## SVR Radial
# cost and scale tuning
library(scales)
tuneplot.svm.t1 <- ggplot(svm.t) + theme_bw() +
  scale_x_continuous(trans=log2_trans(),
                     breaks = trans_breaks("log2", function(x) 2^x)(2^seq(1,16,by=2)),
                     labels = trans_format("log2", math_format(2^.x)),
                     limits=2^c(0,16))
tuneplot.svm.t1

# epsilon tuning
tuneplot.svm.t2 <- ggplot(svm.eps.t, aes(eps, RMSE)) + 
  geom_point() + geom_line() +
  scale_x_continuous(limits=c(0,0.03)) + theme_bw()
tuneplot.svm.t2

## SVR Polynomial
tuneplot.svm.p.t1 <- ggplot(svm.poly.t) + scale_y_continuous(limits=c(0.02,0.04)) + theme_bw()
tuneplot.svm.p.t1

tuneplot.svm.p.t2 <- ggplot(svm.eps.poly.t, aes(eps, RMSE)) + 
  geom_point() + geom_line() + theme_bw()
tuneplot.svm.p.t2

## Neural Networks
tuneplot.nnet.t <- ggplot(nnet.t) + scale_x_continuous(breaks=seq(0,14,by=2),
                                                       limits=c(0,14)) +
                    theme_bw()
tuneplot.nnet.t

## Random Forests
tuneplot.forest.t <- ggplot(forest.t) + scale_x_continuous(breaks=seq(0,14,by=2),
                                                         limits=c(0,14)) +
  theme_bw()
tuneplot.forest.t

## SGB
tuneplot.gbm.t <- ggplot(gbm.tune.t, aes(x=n.trees, y=RMSE, group=as.factor(interaction.depth), col=as.factor(interaction.depth))) + 
  geom_line() + geom_point() + facet_grid(shrinkage~bag.fraction) + 
  scale_colour_hue(name="Interaction Depth")+ xlab("# Boosting Iterations") + 
  ylab("RMSE (Cross-Validation)")
tuneplot.gbm.t

tplot.forest.g <- ggplot(gbm.tune.g, aes(x=n.trees, y=RMSE, group=as.factor(interaction.depth), col=as.factor(interaction.depth))) + 
  geom_line() + geom_point() + facet_grid(shrinkage~bag.fraction) + theme_bw()
  scale_colour_hue(name="Interaction Depth")+ xlab("# Boosting Iterations") + 
  ylab("RMSE (Cross-Validation)")
tplot.forest.g

tplot.forest.m <- ggplot(gbm.tune.m, aes(x=n.trees, y=RMSE, group=as.factor(interaction.depth), col=as.factor(interaction.depth))) + 
  geom_line() + geom_point() + facet_grid(shrinkage~bag.fraction) + 
  scale_colour_hue(name="Interaction Depth")+ xlab("# Boosting Iterations") + 
  ylab("RMSE (Cross-Validation)")
tplot.forest.m

# -----------------------
# Model Comparisons
# ----------------------

# Cross Validation Results
allResamples.t <- resamples(list("SVM" = svm.t2,
                                 "Neural Networks" = nnet.t,
                                 "Random Forests" = forest.t,
                                 "Boosted Tree" = gbm.t))
ggplot(allResamples.t)

allResamples.g <- resamples(list("SVM" = svm.g,
                                 "Neural Networks" = nnet.g,
                                 "Random Forests" = forest.g,
                                 "Boosted Tree" = gbm.g))
ggplot(allResamples.g)

# Validation Results
validation.t <- data.frame(method=c("SVM","Neural Network","Random Forests","SGB"),
                           RMSE=c(diagnose(svm.t2),diagnose(nnet.t, PCA=TRUE),diagnose(forest.t),diagnose(gbm.t)))
plotval.t <- ggplot(validation.t, aes(RMSE, method)) +
  geom_bar(stat="identity", fill="steelblue", width=0.4)  + 
  theme_bw() + scale_y_discrete(limits=validation.t$method) + 
  scale_x_continuous(limits=c(0,0.02)) +
  ylab("Model") + xlab("RMSE (test set)")
plotval.t

# Validation Results
validation.g <- data.frame(method=c("SVM","Neural Network","Random Forests","SGB"),
                           RMSE=c(diagnose(svm.g2),diagnose(nnet.g, PCA=TRUE),diagnose(forest.g),diagnose(gbm.g)))
plotval.g <- ggplot(validation.g, aes(RMSE, method)) +
  geom_bar(stat="identity", fill="steelblue", width=0.4)  + 
  theme_bw() + scale_y_discrete(limits=validation.g$method) + 
  scale_x_continuous(limits=c(0,0.4)) +
  ylab("Model") + xlab("RMSE (test set)")
plotval.g

# Validation Results
validation.m <- data.frame(method=c("SVM","Neural Network","Random Forests","SGB"),
                           RMSE=c(diagnose(svm.m2),diagnose(nnet.m, PCA=TRUE),diagnose(forest.m),diagnose(gbm.m)))
plotval.m <- ggplot(validation.m, aes(RMSE, method)) +
  geom_bar(stat="identity", fill="steelblue", width=0.4)  + 
  theme_bw() + scale_y_discrete(limits=validation.m$method) + 
  scale_x_continuous(limits=c(0,NA)) +
  ylab("Model") + xlab("RMSE (test set)")
plotval.m

# ---------------------
# Prediction Intervals
# ---------------

## computing intervals
interval.t <- prediction.interval(svm.t2, GAIA[,-(1:3)], GAIA$logt, eps=0.01)
interval.g <- prediction.interval(svm.g, GAIA[,-(1:3)], GAIA$logg, eps=0.1)
interval.m <- prediction.interval(gbm.m, GAIA[,-(1:3)], GAIA$metallicity)

## function to plot intervals
plot.interval <- function(interval){
  
  # stratified sample of ~ 100 datapoints
  samples <- stratified(interval, "true", size=(100/8286))
  
  # ordering values
  index <- order(samples$true)
  samples <- samples[index,]
  samples$index <- 1:nrow(samples)
  
  # plotting
  ggplot(data=samples) +
    geom_point(aes(x=index, y=true)) + geom_line(aes(x=index, y=median), colour="blue", show.legend=FALSE) +
    geom_ribbon(aes(x=index, ymin=lower, ymax=upper, fill="red"), alpha=0.3, show.legend=FALSE) +
    ylab("response value") +
    theme_bw()
}

## interval plots 
plots.interval.t <- plot.interval(interval.t)
plots.interval.g <- plot.interval(interval.g)
plots.interval.m <- plot.interval(interval.m)

# CV Function

cross_validate <- function(models, data){
  
  results <- data.frame()
  nfolds <- 10
  
  for (k in 1:nfolds) {
    
    # splitting data into train and test
    test_i <- which(folds_i == k)
    test.data <- data[-test_i, ]
    train.data <- data[test_i, ]
    # naming data
    #y.test <- test.data$temperature
    #y.train <- train.data$temperature
    #x.test <- as.matrix(test.data[,5:20])
    #x.train <- as.matrix(train.data[,5:20])
    # training model
    fit <- models[k]
    best_lambda <- ridge$bestTune$lambda
    # test model
    predictions <- predict(ridge, test.data)
    RMSE <- RMSE(predictions, test.data$temperature)
    temp <- data.frame(best_lambda=best_lambda, RMSE=RMSE)
    results <- rbind(results, temp)
  }
  
  results
  
}

# yeo johnson power transformations
transform.log <- function(x){
  if (x<0)
    y <- -log(-x+1)
  if (x>=0)
    y <- log(x+1)
  y
}

# Build Model Function

predictors <- names(which(models==TRUE))
predictors <- paste(predictors, collapse="+")
as.formula(paste0(outcome,"~",predictors))

# splitting data into groups
#unclass(GAIA$temperature)
#groups.t <- factor(GAIA$temperature)
#g <- split(GAIA$temperature, groups.t)
#g <- cut(GAIA$temperature, levels(factor(GAIA$temperature)), labels=1:nlevels(factor(GAIA$temperature)))


# plotting model on 2D temperature plot
lines(sort(GAIA.PCA$PC1), fitted(model)[order(GAIA.PCA$PC1)], col="red")
line <- data.frame(sort(GAIA.PCA$PC1), fitted(model)[order(GAIA.t$PC1)])

plot(GAIA.t$PC1, GAIA.t$temperature.t)
lines(sort(GAIA.t$PC1), fitted(model)[order(GAIA.t$PC1)], col="red")

## Kernel PCA

library(kernlab)
kPCA <- kpca(~., data=GAIA.scaled[,-(1:3)], kernel='splinedot',
             features=2)
components <- pcv(kPCA)
str(components)

df <- data.frame(logt=GAIA.transformed$logt, PC1=components[,1], PC2=components[,2])
ggplot(df, aes(PC1, PC2, col=logt, fill=logt)) +
  geom_point() +
  scale_color_gradientn(colours=hcl.colors(5)) + 
  scale_fill_gradientn(colours=hcl.colors(5))

ggplot(df, aes(PC1, logt)) + 
  geom_point()


# Getting gradient of colours based on response variable
library(rgl)
myColorRamp <- function(colors, values) {
  v <- (values - min(values))/diff(range(values))
  x <- colorRamp(colors)(v)
  rgb(x[,1], x[,2], x[,3], maxColorValue = 255)
}
cols <- myColorRamp(c("blue", "red"), GAIA.PCA$logt)


# 3D Plot
x <- GAIA.PCA$PC1
y <- GAIA.PCA$PC2
z <- GAIA.PCA$PC3

library(scatterplot3d)
s3d <- plot3d(x=x,y=y,z=z, col=cols,
              xlab="PC1", ylab="PC2", zlab="logt")
s3d


## Principle Curves
library(analogue)
predictors <- as.matrix(GAIA.scaled[,-(1:3)])
components <- pcv(kPCA)
prc1 <- prcurve(GAIA[,-(1:3)], trace=FALSE, thresh=0.0005, maxit = 50)
## Plot
op <- par(mar = c(5,4,2,2) + 0.1)
plot(prc1)
par(op)

plot(prc1$s[ord,][,1], GAIA.transformed$logt)

plot(prc1)


## Non metric multidimensional scaling
library(MASS)
dist <- dist(GAIA[,-(1:3)])
mds <- isoMDS(dist, k=2)


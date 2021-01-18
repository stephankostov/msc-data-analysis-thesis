# Bootstrap Prediction Interval
prediction.interval <- function(model, train.x, train.y, alpha=0.05, nbootstraps=200, eps){
  
  n <- nrow(train.x)
  
  method <- model$method
  parameters <- model$bestTune
  
  residuals.val <- matrix(NaN, ncol=n, nrow=nbootstraps)
  bootstrap.preds <- matrix(NaN, ncol=n, nrow=nbootstraps)
  
  # Computing the validation residuals and bootstrap predictions
  for (b in 1:nbootstraps){
    
    index.train <- sample((1:n), size=n, replace=TRUE)
    index.val <- (1:n)[!((1:n) %in% index.train)]
    
    if (model$method == "avNNet") {
      
      model.fit <- train(train.x[index.train,], train.y[index.train],
                         method=model$method,
                         tuneGrid=model$bestTune,
                         linout = TRUE,
                         trace = FALSE,
                         MaxNWts = 14*(ncol(train.data.PCA)) + 14 + 1,
                         maxit = 500,
                         trControl = trainControl(method = "none"))
    } else if (model$method == "svmRadial") {
      
      model.fit <- train(train.x[index.train,], train.y[index.train],
                         method=model$method,
                         tuneGrid=model$bestTune,
                         preProcess = c("center", "scale"),
                         eps = eps, # manual input of best tuned epsilon value
                         trControl = trainControl(method = "none"))
      
    } else {
      
      model.fit <- train(train.x[index.train,], train.y[index.train],
                         method=model$method,
                         tuneGrid=model$bestTune, 
                         verbose=FALSE,
                         trControl = trainControl(method = "none"))
      
    }
    
    # Validation residuals 
    predictions.val <- predict(model.fit, train.x[index.val,])
    residuals <- train.y[index.val] - predictions.val
    residuals.val[b,index.val] <- residuals
    
    # Bootstrap predictions
    bootstrap.preds[b,] <- predict(model.fit, train.x)
    
    print(b)
    
  }
  
  # computing mean validation residuals for each train set point
  residuals.val <- colMeans(residuals.val, na.rm=TRUE)
  residuals.val[is.na(residuals.val)] <- 0
  
  # computing the predictions and training residuals
  
  if (model$method == "avNNet") {
    
    model.fit <- train(train.x[index.train,], train.y[index.train],
                       method=model$method,
                       tuneGrid=model$bestTune,
                       linout = TRUE,
                       trace = FALSE,
                       MaxNWts = 14*(ncol(train.data.PCA)) + 14 + 1,
                       maxit = 500,
                       trControl = trainControl(method = "none"))
  } else if (model$method == "svmRadial") {
    
    model.fit <- train(train.x[index.train,], train.y[index.train],
                       method=model$method,
                       tuneGrid=model$bestTune,
                       preProcess = c("center", "scale"),
                       eps = eps, # manual input of best tuned epsilon value
                       trControl = trainControl(method = "none"))
    
  } else {
    
    model.fit <- train(train.x[index.train,], train.y[index.train],
                       method=model$method,
                       tuneGrid=model$bestTune, 
                       verbose=FALSE,
                       trControl = trainControl(method = "none"))
    
  }
  
  predictions <- predict(model.fit, train.x)
  residuals.train <- train.y - predictions
  
  ## computing the .632+ bootstrap estimate for the sample noise and bias
  combos <- tidyr::crossing(train.y, predictions)
  no.information.err <- mean(abs(combos$train.y - combos$predictions))
  
  generalisation <- abs(residuals.val - residuals.train)
  no.information.val <- abs(no.information.err - residuals.train)
  relative.overfitting.rate <- generalisation / no.information.val
  
  weight <- .632 / (1 - .368 * relative.overfitting.rate)
  residuals <- (1 - weight) * residuals.train + weight * residuals.val
  
  # Constructing the interval and get percentiles
  bootstrap.corrected <- sweep(bootstrap.preds, 2, residuals, "+")
  qs <- c(alpha/2, 0.5, (1-alpha/2))
  bootstrap.qs <- apply(bootstrap.corrected, 2, quantile, probs=qs)
  bootstrap.mean <- apply(bootstrap.corrected, 2, mean)
  
  df <- data.frame(lower=bootstrap.qs[1,], point=predictions, mean=bootstrap.mean, upper=bootstrap.qs[3,])
  
  return(data.frame(true=train.y, lower=bootstrap.qs[1,], point=predictions, median=bootstrap.qs[2,], mean=bootstrap.mean, upper=bootstrap.qs[3,]))
}

# For simulating demand with actual annual demand
ac_simulate_demand <-function(sim,ac.anndemand,nyears=length(sim$hhfit)/seasondays/48,periods=48){
  n <- nyears*seasondays*periods
  afit <- rep(ac.anndemand,each=n)
  
  ##############################################################
  # total below should equal to sim$demand
  total <- c(sim$hhfit)+c(sim$hhres)
  
  # use log for both annual and half-hourly sim
  dem <- exp(total[1:n]) * afit
  
  annmax <- blockstat(dem,seasondays,max,fill=FALSE,periods=periods)
  dem <- matrix(dem,nrow=seasondays*periods,byrow=FALSE)
  
  return(list(demand=ts(dem,frequency=periods,start=1),annmax=annmax))
}

# For simulating demand with proposed model
p_simulate_demand <-
  function(sim,amodel,afcast,nyears=length(sim$hhfit)/seasondays/48,periods=48)
  {
    n <- nyears*seasondays*periods
    
    # forecast demand difference of annual average demand
    afit <- predict(amodel,newdata=afcast,se=TRUE)
    avar <- afit$se^2 + summary(amodel)$sigma^2
    # rnorm
    afit <- rep(rnorm(nyears,afit$fit,sqrt(avar)),each=seasondays*periods)
    
    ##############################################################
    # total below should equal to sim$demand
    total <- c(sim$hhfit)+c(sim$hhres)
    
    # use log for both annual and half-hourly sim
    dem <- exp(total[1:n]) * afit
    
    annmax <- blockstat(dem,seasondays,max,fill=FALSE,periods=periods)
    dem <- matrix(dem,nrow=seasondays*periods,byrow=FALSE)
    
    return(list(demand=ts(dem,frequency=periods,start=1),annmax=annmax))
  }

# Function to preprocess
process_data <- function(data,idx,cutoff){
  # Indexing
  row = nrow(data)
  
  # Remove highly correlated features
  highlyCorDescr <- findCorrelation(cor(data[1:(row-1),-grep("anndemand", colnames(data))]), cutoff = cutoff)
  econ.df.decor <- data[,-highlyCorDescr]
  cat('original:',ncol(data),'removed:',length(highlyCorDescr),'left:',ncol(econ.df.decor))
  
  # Center, scale, nzv the data
  preProcValues <- preProcess(econ.df.decor[1:(row-1),-grep("anndemand", colnames(econ.df.decor))], method = c("center", "scale",'nzv'))
  
  # Splitting the data
  train.econ <- predict(preProcValues, econ.df.decor[1:idx,])
  test.econ <- predict(preProcValues, econ.df.decor[(idx+1):(row-1),])
  validate.econ <- predict(preProcValues, econ.df.decor[row,])
  econ <- rbind(train.econ,test.econ,validate.econ)
  
  return(list(train.econ = train.econ, test.econ = test.econ, validate.econ = validate.econ, econ = econ))
}

# Proposed model to model the long-term effect
proposed_model <- function(train,test,n){
  
  # shuffle the features
  set.seed(123); col.idx = replicate(n,sample(ncol(train)))
  
  cl <- makeCluster(detectCores()[1])
  
  # Parallelize training
  registerDoParallel(cl)
  models <- foreach(i=1:n,.packages = 'caret') %dopar% {
    temp = train(anndemand ~ ., data = train[,col.idx[,i]], 
                 method = 'glmStepAIC',
                 trControl = trainControl(method = "none"),
                 trace = FALSE,
                 direction = 'backward')
  }
  stopCluster(cl)
  
  # Use out of sample error to select model
  mae = c()
  for(i in 1:length(models)){
    pred <- predict(models[[i]],test)
    mae[i] = mean(abs(pred-test[,'anndemand']))
  }
  
  # Refitting the chosen model with all observed data
  newformula <- as.formula(paste0("anndemand ~ ",paste(sprintf("%s", names(coefficients(models[[which.min(mae)]]$finalMode)[-1])),collapse=" + ")))
  newfit <- lm(newformula, data=rbind(train,test))
  
  return(list(fit=newfit,models=models))
}

pinball_loss <- function(tau, y, q) {
  
  y = quantile(y, tau)
  q = quantile(q, tau)
  
  pl_df <- data.frame(tau = tau,
                      y = y,
                      q = q)
  
  pl_df <- pl_df %>%
    mutate(L = ifelse(y>=q,
                      tau/100 * (y-q),
                      (1-tau/100) * (q-y)))
  
  return(sum(pl_df$L))
}

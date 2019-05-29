library(MEFM) # for forecasting models
library(ggplot2) # for plotting
library(foreach) # for parallelism
library(doParallel) # for parallelism
library(caret) # for proposed model
library(dplyr) # for pinball funtions
source("mainfunctions.R") # import self-defined functions

# Read and preprocess the data
ori.sa <- sa
econ.df <- read.csv('econ.csv')
data <- process_data(econ.df,10,0.90)

# Define the formulas to train
formula.hh <- list()
for(i in 1:48)
  formula.hh[[i]] = as.formula(log(ddemand) ~ ns(temp, df=2) + day 
                               + holiday + ns(timeofyear, 9) + ns(avetemp, 3) + ns(dtemp, 3) + ns(lastmin, 3) 
                               + ns(prevtemp1, df=2) + ns(prevtemp2, df=2) 
                               + ns(prevtemp3, df=2) + ns(prevtemp4, df=2) 
                               + ns(day1temp, df=2) + ns(day2temp, df=2) 
                               + ns(day3temp, df=2) + ns(prevdtemp1, 3) + ns(prevdtemp2, 3) 
                               + ns(prevdtemp3, 3) + ns(day1dtemp, 3))

formula.a <- as.formula(anndemand ~ gsp + ddays + resiprice)

# Create lagged temperature variables
sa <- maketemps(sa,2,48)

# Train the based model
sa.model <- demand_model(sa[sa$fyear != 2014,], window(sa.econ,end=2013), formula.hh, formula.a)

# Simulate future normalized half-hourly data
simdemand <- simulate_ddemand(sa.model, sa[sa$fyear != 2014,], simyears=1000)

# Simulate half-hourly data for actual annual demand
a.demand <- ac_simulate_demand(simdemand,data$validate.econ[,'anndemand'])
a.weekly_max <- blockstat(a.demand$demand,7,max,fill=FALSE,periods=48)
a.ann_max <- a.demand$annmax

# Simulate half-hourly data for base model
demand <- simulate_demand(simdemand, window(sa.econ,start=2014))
weekly_max <- blockstat(demand$demand,7,max,fill=FALSE,periods=48)
ann_max <- demand$annmax

# Fitting the proposed model
p.fit <- proposed_model(data$train.econ,data$test.econ,300)
summary(p.fit$fit)

# Simulate half-hourly data for proposed model
p.demand <- p_simulate_demand(simdemand, p.fit$fit, data$validate.econ)
p.weekly_max <- blockstat(p.demand$demand,7,max,fill=FALSE,periods=48)
p.ann_max <- p.demand$annmax

##### Visualization ######
# Load demand vs Temperature
ggplot(ori.sa[ori.sa$year %in% c(2013,2014),],aes(x=temp1,y=demand))+ 
  geom_point(colour = "steelblue", size = 0.5) + theme_classic()+
  labs(title='Non-Linear Relationship between Residential Load Demand and Temperature',y='Load Demand (GW)',x='Temperature (°C)')+
  geom_smooth(colour = 'lightcoral')

# Long-term effect model
autoplot(ts(econ.df[,'anndemand'],start=2000))+ theme_classic() +
  autolayer(ts(predict(sa.model$a,econ.df),start=2000),series='Base Model') +
  autolayer(ts(predict(p.fit$fit,data$econ),start=2000),series='Proposed Model') +
  labs(colour='Model',y='Annual electric load demand (GW)',
       title='Base model vs Proposed model') +
  geom_vline(xintercept=2013, linetype="dotted", color = "black", size=0.5)

# Weekly Peak density
ac.weekly.peak <- data.frame('value' = blockstat(sa[sa$fyear == 2014,]$demand,7,max,fill=FALSE,periods=48))
df = rbind(data.frame('value'=a.weekly_max,'Model'='Density with actual annual base demand'),
           data.frame('value'=weekly_max,'Model'='Base model'),
           data.frame('value'=p.weekly_max,'Model'='Proposed model'))

ggplot(df,aes(value,color=Model))+geom_density(size=0.8)+theme_classic()+ geom_hline(yintercept=0, colour="white", size=2) +
  labs(x= 'Load Demand (GW)',y='Density',title = 'Weekly Peak Load Demand Density') +
  theme(legend.justification = c(1, 1), legend.position = c(1, 1))+
  geom_point(data=ac.weekly.peak,aes(x = value,y=0,color='Actual Weekly Peak Demand'),shape = 17,size=2)

# Annual Peak density
ac.annual.peak <- data.frame('value' = blockstat(sa[sa$fyear == 2014,]$demand,length(sa[sa$fyear == 2014,]$demand),max,fill=FALSE,periods=48))
an_df = rbind(data.frame('value'=a.ann_max,'Model'='Density with actual annual base demand'),
           data.frame('value'=ann_max,'Model'='Base model'),
           data.frame('value'=p.ann_max,'Model'='Proposed model'))

ggplot(an_df,aes(value,color=Model))+geom_density(size=0.8)+theme_classic()+ geom_hline(yintercept=0, colour="white", size=2) +
  labs(x= 'Load Demand (GW)',y='Density',title = 'Annual Peak Load Demand Density') +
  theme(legend.justification = c(1, 1), legend.position = c(1, 1))+
  geom_point(data=ac.annual.peak,aes(x = value,y=0,color='Actual Annual Peak Demand'),shape = 17,size=2)

##### Error Measures ######
# Pinball loss
cat('Weekly Pinball loss for proposed model is ',pinball_loss(seq(0.01,0.99,0.01),a.weekly_max,p.weekly_max))
cat('Weekly Pinball loss for base model is ',pinball_loss(seq(0.01,0.99,0.01),a.weekly_max,weekly_max))
cat('Annual Pinball loss for proposed model is ',pinball_loss(seq(0.01,0.99,0.01),a.ann_max,p.ann_max))
cat('Annual Pinball loss for base model is ',pinball_loss(seq(0.01,0.99,0.01),a.ann_max,ann_max))
# MAE for Point forecast 
cat('Proposed vs weekly actual',abs(mean(p.ann_max) - ac.annual.peak)[[1]])
cat('Base vs weekly actual',abs(mean(ann_max) - ac.annual.peak)[[1]])
cat('Proposed vs annual actual',mean(abs(rowMeans(matrix(p.weekly_max,ncol=1000))[1:nrow(ac.weekly.peak)] - ac.weekly.peak)[[1]]))
cat('Base vs annual actual',mean(abs(rowMeans(matrix(weekly_max,ncol=1000))[1:nrow(ac.weekly.peak)] - ac.weekly.peak)[[1]]))

##### Other ######
# Calculate the appearance of each coefficients
cof <- c()
for(i in p.fit$models){
  cof <- append(cof,names(coefficients(i$finalModel)))
  #print(names(coefficients(i$finalModel)))
  #print(deviance(i$finalModel))
}
table(cof)[order(-table(cof))]

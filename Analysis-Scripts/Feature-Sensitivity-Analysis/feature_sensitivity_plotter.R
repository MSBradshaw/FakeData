library(readr)
library(ggplot2)
library(dplyr)

data <- read_csv('Analysis-Scripts/Feature-Sensitivity-Analysis/results.csv')

std_error <- function(x){
  sd(x)/sqrt(length(x))
}

get_errors <- function(x){
  #x <- rf
  errors = c()
  for(i in unique(x$features)){
    d <- x[x$features==i,]$score
    n <- std_error(d)
    errors <- c(errors,n)
  }
  errors
}

rf <- data[data$learner == 'Random Forest',]
errors <- get_errors(rf)
rf <- aggregate(rf,by=list(rf$features),FUN=mean)
rf$learner <- 'Random Forst'
rf$type <- 'Impuation'
rf$error <- errors

gb <- data[data$learner == 'GBC',]
errors <- get_errors(gb)
gb <- aggregate(gb,by=list(gb$features),FUN=mean)
gb$learner <- 'GBC'
gb$type <- 'Impuation'
gb$error <- errors

nb <- data[data$learner == 'Naive Bayes',]
errors <- get_errors(nb)
nb <- aggregate(nb,by=list(nb$features),FUN=mean)
nb$learner <- 'Naive Bayes'
nb$type <- 'Impuation'
nb$error <- errors

knn <- data[data$learner == 'KNN',]
errors <- get_errors(knn)
knn <- aggregate(knn,by=list(knn$features),FUN=mean)
knn$learner <- 'KNN'
knn$type <- 'Impuation'
knn$error <- errors

d <- bind_rows(knn,nb,gb,rf)

d$size = apply(d,1,function(row){
  groups <- c(10,20,30,40,50,60,70,80,90,100,500,1000,2000,4000,6000,8000,10000,12000,14000,16000,17000)
  i <- match(as.numeric(row[6]),groups)
  i
})
colnames(d) <- c("Group.1","X1","score","Learner","type","features","error","size")

cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


ggplot(data = d,aes(x=size,y=score,color=Learner))  + geom_point() + 
  geom_errorbar(aes(ymin=score-error, ymax=score+error),width=.4) + 
  xlab('Number of Measurements') +
  ylab('Mean Accuracy of 100 Replicates') + 
  geom_line() + scale_x_continuous(breaks = 1:21,
                                   labels = c('10','20','30','40','50','60','70','80','90',
                                              '100','500','1000','2000','4000','6000','8000',
                                              '10000','12000','14000','16000','17000')) +
  theme_bw() +
  theme(plot.title = element_text(size=48), 
        legend.text=element_text(size=20),
        legend.title=element_text(size=20),
        axis.title.y=element_text(size=20),
        axis.title.x=element_text(size=20),
        panel.grid.minor = element_line(size = 1), 
        panel.grid.major = element_line(size = 1),) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)
ggsave('Analysis-Scripts/Feature-Sensitivity-Analysis/Figure-4.png',units='in',width = 15, height = 10)

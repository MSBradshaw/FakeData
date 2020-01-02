library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
data <- read_csv('Analysis-Scripts/Partially-Fake-Analysis/results.csv',col_names = FALSE)

data$group <- paste(unlist(data$X3),unlist(data$X5),sep = '%')
d <- aggregate(data[,2],list(data$group),mean)
d <- separate(d,Group.1,c('Model','Value'),'%')
d$Value <- as.numeric(d$Value)
p <- ggplot(data=d,aes(x=Value,y=X2,color=Model)) + geom_line() + ylim(0,1) + 
  xlab('% Fake') + ylab('Accuracy')
p

library(readr)
library(ggplot2)
library(dplyr)
library(scales)


data <- read_csv('Analysis-Scripts/Feature-Sensitivity-Analysis/results.csv')
colnames(data) <- c('X1','score','Model','type','features')

data = data[!is.na(data$features),]

std_error <- function(x){
  #print(x)
  sd(x)/sqrt(length(x))
}

get_errors <- function(x){
  #x <- rf
  errors = c()
  for(i in unique(x$features)){
    #print(i)
    d <- x[x$features==i,]$score
    n <- std_error(d)
    #print(n)
    errors <- c(errors,n)
  }
  errors
}

rf <- data[data$Model == 'Random Forest',]
errors <- get_errors(rf)
rf <- aggregate(rf,by=list(rf$features),FUN=mean)
rf$Model <- 'Random Forst'
rf$type <- 'Impuation'
rf$error <- errors

gb <- data[data$Model == 'GBC',]
errors <- get_errors(gb)
gb <- aggregate(gb,by=list(gb$features),FUN=mean)
gb$Model <- 'GBC'
gb$type <- 'Impuation'
gb$error <- errors

nb <- data[data$Model == 'Naive Bayes',]
errors <- get_errors(nb)
nb <- aggregate(nb,by=list(nb$features),FUN=mean)
nb$Model <- 'Naive Bayes'
nb$type <- 'Impuation'
nb$error <- errors

knn <- data[data$Model == 'KNN',]
errors <- get_errors(knn)
knn <- aggregate(knn,by=list(knn$features),FUN=mean)
knn$Model <- 'KNN'
knn$type <- 'Impuation'
knn$error <- errors

svm <- data[data$Model == 'SVM',]
errors <- get_errors(svm)
svm <- aggregate(svm,by=list(svm$features),FUN=mean)
svm$Model <- 'SVM'
svm$type <- 'Impuation'
svm$error <- errors

d <- bind_rows(knn,nb,gb,rf,svm)

d <- d[d$features > 10,]

d$size = apply(d,1,function(row){
  groups <- c(10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,4000,6000,8000,10000,12000,14000,16000,17000)
  i <- match(as.numeric(row[6]),groups)
  i
})
colnames(d) <- c("Group.1","X1","score","Model","type","features","error","size")

cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



# ggplot(data = d,aes(x=size,y=score,color=Model))  + geom_point() + 
#   geom_errorbar(aes(ymin=score-error, ymax=score+error),width=.4) + 
#   xlab('Number of Measurements') +
#   ylab('Mean Accuracy of 100 Replicates') + 
#   geom_line() + scale_x_continuous(breaks = 1:28,
#                                    labels = c('10 ','20 ','30 ','40 ','50 ','60 ','70 ','80 ','90 ',
#                                               '100 ','200 ','300 ','400 ','500 ','600 ','700 ','800 ',
#                                               '900 ','1000 ','2000 ','4000 ','6000 ','8000 ',
#                                               '10000 ','12000 ','14000 ','16000 ','17000 ')) +
#   theme_bw() +
#   theme(plot.title = element_text(size=24), 
#         legend.text=element_text(size=10),
#         legend.title=element_text(size=10),
#         axis.title.y=element_text(size=10),
#         axis.title.x=element_text(size=10),
#         axis.text.x = element_text(size=7,angle=45),
#         axis.text.y = element_text(size=7),
#         panel.grid.minor = element_line(size = .5), 
#         panel.grid.major = element_line(size = .5),) + 
#   scale_fill_manual(values=cbPalette) + 
#   scale_colour_manual(values=cbPalette)

# ggsave('Figures/Figure-4.png',device='png',units='in',width = 7.5, height = 5,dpi=300)
# ggsave('Figures/Figure-4.tiff',device='tiff',units='in',width = 7.5, height = 5,dpi=300)

# ggplot(data = d,aes(x=Group.1,y=score,color=Model))  + geom_point() + 
#   geom_errorbar(aes(ymin=score-error, ymax=score+error),width=.05) + 
#   xlab('Number of Measurements') +
#   ylab('Mean Accuracy of 100 Replicates') + 
#   geom_line() + scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
#                               labels = trans_format("log10", math_format(10^.x))) +
#   annotation_logticks()  +
#   theme_bw() +
#   theme(plot.title = element_text(size=24), 
#         legend.text=element_text(size=10),
#         legend.title=element_text(size=10),
#         axis.title.y=element_text(size=10),
#         axis.title.x=element_text(size=10),
#         axis.text.x = element_text(size=7,angle=45),
#         axis.text.y = element_text(size=7),
#         panel.grid.minor = element_line(size = .5), 
#         panel.grid.major = element_line(size = .5),) + 
#   scale_fill_manual(values=cbPalette) + 
#   scale_colour_manual(values=cbPalette)
# 
# ggsave('Figures/Figure-4.png',device='png',units='in',width = 8, height = 5,dpi=300)
# ggsave('Figures/Figure-4.tiff',device='tiff',units='in',width = 8, height = 5,dpi=300)


ggplot(data = d,aes(x=Group.1,y=score,color=Model))  +
  geom_ribbon(aes(ymin = score-error, ymax = score+error, fill=Model),alpha = 0.3) +
  xlab('Number of Measurements') +
  ylab('Mean Accuracy of 50 Replicates') + 
  geom_line() + scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                              labels = trans_format("log10", math_format(10^.x))) +
  annotation_logticks()  +
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        axis.title.x=element_text(size=10),
        axis.text.x = element_text(size=7,angle=45),
        axis.text.y = element_text(size=7),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)
ggsave('Figures/Figure-4.png',device='png',units='in',width = 8, height = 5,dpi=300)
ggsave('Figures/Figure-4.tiff',device='tiff',units='in',width = 8, height = 5,dpi=300)

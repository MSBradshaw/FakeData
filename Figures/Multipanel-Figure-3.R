library(readr)
library(tibble)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(grid)
library(gridExtra) 

#read in imputed data
dataImp <- read_csv('F1Results/imputation_benford_4.csv')

for(i in seq(5,50)){
  print(i)
  temp <- read_csv(paste('F1Results/imputation_benford_',i,'.csv',sep=''))
  dataImp <- bind_rows(dataImp,temp)
}

colnames(dataImp) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataImp = dataImp[dataImp$Model != 'MLP',]
#dataImp = dataImp[dataImp$Model != 'SVC',]
dataImp[dataImp$Model == 'SVC',]$Model <- 'SVM'
dataImp[dataImp$Model == 'Random Forest',]$Model <- 'RF                                                   '
dataImp[dataImp$Model == 'Naive Bayes',]$Model <- 'NB'
f1dataImp = dataImp

cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

f1impP <- ggplot(dataImp, aes(x=Model, y=score, fill=Model),show.legend = FALSE) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE,show.legend = FALSE) +
  ylab('Imputation F1') +
  ylim(0.0,1.0) + 
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)

f1imp_with_leg <- ggplot(dataImp, aes(x=Model, y=score, fill=Model)) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model)) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE) +
  ylab('Imputation F1') +
  ylim(0.0,1.0) + 
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)


#read in random data
dataRan <- read_csv('F1Results/random_benford_1.csv')
for(i in seq(2:50)){
  temp <- read_csv(paste('F1Results/random_benford_',i,'.csv',sep=''))
  dataRan <- bind_rows(dataRan,temp)
}


colnames(dataRan) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataRan = dataRan[dataRan$Model != 'MLP',]
#dataRan = dataRan[dataRan$Model != 'SVC',]
dataRan[dataRan$Model == 'SVC',]$Model <- 'SVM'
dataRan[dataRan$Model == 'Random Forest',]$Model <- 'RF'
dataRan[dataRan$Model == 'Naive Bayes',]$Model <- 'NB'
f1dataRan = dataRan

f1ranP <- ggplot(dataRan, aes(x=Model, y=score, fill=Model)) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE,show.legend = FALSE) + 
  ylab('Random F1') +
  ylim(0.0,1.0) +
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)


#read in resampling data
dataRes <- read_csv('F1Results/resampled_benford_1.csv')
for(i in seq(2:50)){
  temp <- read_csv(paste('F1Results/resampled_benford_',i,'.csv',sep=''))
  dataRes <- bind_rows(dataRes,temp)
}


colnames(dataRes) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataRes = dataRes[dataRes$Model != 'MLP',]
#dataRes = dataRes[dataRes$Model != 'SVC',]
dataRes[dataRes$Model == 'SVC',]$Model <- 'SVM'
dataRes[dataRes$Model == 'Random Forest',]$Model <- 'RF'
dataRes[dataRes$Model == 'Naive Bayes',]$Model <- 'NB'
f1dataRes = dataRes

f1resP <- ggplot(dataRes, aes(x=Model, y=score, fill=Model)) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE,show.legend = FALSE) + 
  ylab('Resampled F1') +
  ylim(0.0,1.0) +
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)


# Read In Accuracy Info Too


#read in imputed data
dataImp <- read_csv('Results/imputation_benford_4.csv')

for(i in seq(5,50)){
  print(i)
  temp <- read_csv(paste('Results/imputation_benford_',i,'.csv',sep=''))
  dataImp <- bind_rows(dataImp,temp)
}

colnames(dataImp) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataImp = dataImp[dataImp$Model != 'MLP',]
#dataImp = dataImp[dataImp$Model != 'SVC',]
dataImp[dataImp$Model == 'SVC',]$Model <- 'SVM'
dataImp[dataImp$Model == 'Random Forest',]$Model <- 'RF                                                   '
dataImp[dataImp$Model == 'Naive Bayes',]$Model <- 'NB'

cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

impP <- ggplot(dataImp, aes(x=Model, y=score, fill=Model),show.legend = FALSE) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE,show.legend = FALSE) +
  ylab('Imputation Accuracy') +
  ylim(0.0,1.0) + 
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)


#read in random data
dataRan <- read_csv('Results/random_benford_1.csv')
for(i in seq(2:50)){
  temp <- read_csv(paste('Results/random_benford_',i,'.csv',sep=''))
  dataRan <- bind_rows(dataRan,temp)
}


colnames(dataRan) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataRan = dataRan[dataRan$Model != 'MLP',]
#dataRan = dataRan[dataRan$Model != 'SVC',]
dataRan[dataRan$Model == 'SVC',]$Model <- 'SVM'
dataRan[dataRan$Model == 'Random Forest',]$Model <- 'RF'
dataRan[dataRan$Model == 'Naive Bayes',]$Model <- 'NB'

ranP <- ggplot(dataRan, aes(x=Model, y=score, fill=Model)) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE,show.legend = FALSE) + 
  ylab('Random Accuracy') +
  ylim(0.0,1.0) +
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)


#read in resampling data
dataRes <- read_csv('Results/resampled_benford_1.csv')
for(i in seq(2:50)){
  temp <- read_csv(paste('Results/resampled_benford_',i,'.csv',sep=''))
  dataRes <- bind_rows(dataRes,temp)
}


colnames(dataRes) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataRes = dataRes[dataRes$Model != 'MLP',]
#dataRes = dataRes[dataRes$Model != 'SVC',]
dataRes[dataRes$Model == 'SVC',]$Model <- 'SVM'
dataRes[dataRes$Model == 'Random Forest',]$Model <- 'RF'
dataRes[dataRes$Model == 'Naive Bayes',]$Model <- 'NB'

resP <- ggplot(dataRes, aes(x=Model, y=score, fill=Model)) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE,show.legend = FALSE) + 
  ylab('Resampled Accuracy') +
  ylim(0.0,1.0) +
  theme_bw() +
  theme(plot.title = element_text(size=24), 
        legend.text=element_text(size=10),
        legend.title=element_text(size=10),
        axis.title.y=element_text(size=10),
        panel.grid.minor = element_line(size = .5), 
        panel.grid.major = element_line(size = .5),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)


leg <- get_legend(f1imp_with_leg)
leg <- as_ggplot(leg)

p7 <- ggarrange(legend,impP,labels = c('','C'), nrow=2, font.label = list(size = 10))
p7 <- ggarrange(ggdraw(legend),impP,labels = c('','C'), ncol=2, font.label = list(size = 10))


tripple <- ggarrange(ranP,resP, impP,labels = c('A','B','C'), ncol=3, font.label = list(size = 10))
f1tripple <- ggarrange(f1ranP,f1resP, f1impP,labels = c('D','E','F'), ncol=3, font.label = list(size = 10))
f1_and_acc <- ggarrange(tripple,f1tripple, nrow=2, font.label = list(size = 10))
f1_and_acc_legend <- ggarrange(f1_and_acc,leg, ncol=2, font.label = list(size = 10),widths = c(4,1))
f1_and_acc_legend


#multipanel <- ggarrange(ranP,resP, impP,NULL,leg, f1ranP,f1resP,f1impP, labels = c('A','B','C','','','D','E','F'), ncol = 5, nrow=2,widths = c(2,2,2,1,1))
#multipanel

multipanel <- ggarrange(ranP,f1ranP, resP,f1resP, impP,f1impP,NULL,leg, labels = c('A','B','C','D','E','F','',''), ncol = 2, nrow=4,heights = c(2,2,2,1,1))
multipanel

ggsave(plot = multipanel,'Figures/MultiPanel-Figure-3.png', device='png',width = 7.5, height = 10, units = c("in"),dpi=300)
ggsave(plot = multipanel,'Figures/MultiPanel-Figure-3.tiff', device='tiff',width = 7.5, height = 10, units = c("in"),dpi=300)


print_acc <- function(d){
  final = ''
  for(x in unique(d$Model)){
    acc = paste(as.character(round((mean(d[d$Model == x,]$score) * 100))), '%', sep='')
    std = paste(as.character(round((sd(d[d$Model == x,]$score) * 100),digits=1)), '%)', sep='')
    std = paste('(+/- ', std, sep='')
    s = paste(x,  acc,  std)
    final = paste(final,s,sep=', ')
  }
  print(final)
}

print_f1 <- function(d){
  final = ''
  for(x in unique(d$Model)){
    acc = paste(as.character(round(mean(d[d$Model == x,]$score),digits=2)), '', sep='')
    std = paste(as.character(round(sd(d[d$Model == x,]$score),digits=2)), ')', sep='')
    std = paste('(+/- ', std, sep='')
    s = paste(x,  acc,  std)
    
    final = paste(final,s,sep=', ')
  }
  print(final)
}


print_acc(dataRan)
print_f1(f1dataRan)

print_acc(dataRes)
print_f1(f1dataRes)

print_acc(dataImp)
print_f1(f1dataImp)

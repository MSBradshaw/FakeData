library(readr)
library(tibble)
library(ggplot2)
library(dplyr)
library(ggpubr)


#read in imputed data
dataImp <- read_csv('Analysis-Scripts/Generic-Analysis/imputation-cna-results/imputation-cna-results-4.csv')

for(i in seq(5,50)){
  print(i)
  temp <- read_csv(paste('Analysis-Scripts/Generic-Analysis/imputation-cna-results/imputation-cna-results-',i,'.csv',sep=''))
  dataImp <- bind_rows(dataImp,temp)
}

colnames(dataImp) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataImp = dataImp[dataImp$Model != 'MLP',]
dataImp = dataImp[dataImp$Model != 'SVC',]
dataImp[dataImp$Model == 'Random Forest',]$Model <- 'RF                                                   '
dataImp[dataImp$Model == 'Naive Bayes',]$Model <- 'NB'

cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

impP <- ggplot(dataImp, aes(x=Model, y=score, fill=Model)) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model)) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=1,alpha = 0.5,notch = TRUE) +
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
dataRan <- read_csv('Analysis-Scripts/Generic-Analysis/random-cna-results/random-cna-results-1.csv')
for(i in seq(2:50)){
  temp <- read_csv(paste('Analysis-Scripts/Generic-Analysis/random-cna-results/random-cna-results-',i,'.csv',sep=''))
  dataRan <- bind_rows(dataRan,temp)
}


colnames(dataRan) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataRan = dataRan[dataRan$Model != 'MLP',]
dataRan = dataRan[dataRan$Model != 'SVC',]
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
dataRes <- read_csv('Analysis-Scripts/Generic-Analysis/cna-resampling-results/resample-cna-1.csv')
for(i in seq(2:50)){
  temp <- read_csv(paste('Analysis-Scripts/Generic-Analysis/cna-resampling-results/resample-cna-',i,'.csv',sep=''))
  dataRes <- bind_rows(dataRes,temp)
}


colnames(dataRes) <- c('X1','score','Model')
#remove the MLP and SVC from the results because they are crappy
dataRes = dataRes[dataRes$Model != 'MLP',]
dataRes = dataRes[dataRes$Model != 'SVC',]
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


p5 <- ggarrange(ranP,resP,labels = c('A','B'), ncol=2, font.label = list(size = 10))

p6 <- ggarrange(p5,impP,labels = c('','C'), nrow=2, font.label = list(size = 10))
p6

ggsave(plot = p6,'Figures/Figure-3.png', device='png',width = 5, height = 3.75, units = c("in"),dpi=300)
ggsave(plot = p6,'Figures/Figure-3.tiff', device='tiff',width = 5, height = 3.75, units = c("in"),dpi=300)

library(readr)
library(tibble)
library(ggplot2)
library(dplyr)
library(ggpubr)


#read in resampling data
dataImp <- read_csv('Analysis-Scripts/Generic-Analysis/imputation-cna-50-ff/imputation-cna-results-ff-4.csv')
for(i in seq(5,50)){
  temp <- read_csv(paste('Analysis-Scripts/Generic-Analysis/imputation-cna-50-ff/imputation-cna-results-ff-',i,'.csv',sep=''))
  dataImp <- bind_rows(dataImp,temp)
}
colnames(dataImp) <- c('X1','score','Model')
dataImp[dataImp$Model == 'Random Forest',]$Model <- 'RF                                                   '
dataImp[dataImp$Model == 'Naive Bayes',]$Model <- 'NB'



cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

impP <- ggplot(dataImp, aes(x=Model, y=score, fill=Model)) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model)) +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=2,alpha = 0.5,notch = TRUE) +
  ylab('Imputation Accuracy') +
  ylim(0.0,1.0) + 
  theme_bw() +
  theme(plot.title = element_text(size=48), 
        legend.text=element_text(size=20),
        legend.title=element_text(size=20),
        axis.title.y=element_text(size=20),
        panel.grid.minor = element_line(size = 1), 
        panel.grid.major = element_line(size = 1),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)
impP

#read in resampling data
ranP <- ggplot(dataRan, aes(x=Model, y=score, fill=Model)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=2,alpha = 0.5,notch = TRUE,show.legend = FALSE) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  ylab('Random Accuracy') +
  ylim(0.0,1.0) +
  theme_bw() +
  theme(plot.title = element_text(size=48), 
        legend.text=element_text(size=20),
        legend.title=element_text(size=20),
        axis.title.y=element_text(size=20),
        panel.grid.minor = element_line(size = 1), 
        panel.grid.major = element_line(size = 1),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)
ranP

dataRes <- read_csv('Analysis-Scripts/Generic-Analysis/resample-cna-50-ff/resample-cna-results-ff-1.csv')
for(i in seq(2,50)){
  temp <- read_csv(paste('Analysis-Scripts/Generic-Analysis/resample-cna-50-ff/resample-cna-results-ff-',i,'.csv',sep=''))
  dataRes <- bind_rows(dataRes,temp)
}
colnames(dataRes) <- c('X1','score','Model')
dataRes[dataRes$Model == 'Random Forest',]$Model <- 'RF'
dataRes[dataRes$Model == 'Naive Bayes',]$Model <- 'NB'

resP <- ggplot(dataRes, aes(x=Model, y=score, fill=Model)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=2,alpha = 0.5,notch = TRUE,show.legend = FALSE) + 
  geom_jitter(shape=16, position=position_jitter(0.2),aes(color=Model),show.legend = FALSE) +
  ylab('Resampled Accuracy') +
  ylim(0.0,1.0) +
  theme_bw() +
  theme(plot.title = element_text(size=48), 
        legend.text=element_text(size=20),
        legend.title=element_text(size=20),
        axis.title.y=element_text(size=20),
        panel.grid.minor = element_line(size = 1), 
        panel.grid.major = element_line(size = 1),
        axis.title.x = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)


p5 <- ggarrange(ranP,resP,labels = c('A','B'), ncol=2)

p6 <- ggarrange(p5,impP,labels = c('','c'), nrow=2, font.label = list(size = 20))
p6

ggsave(plot = p6,'Analysis-Scripts/Generic-Analysis/Figure-2.png',width = 10, height = 7.5, units = c("in"))

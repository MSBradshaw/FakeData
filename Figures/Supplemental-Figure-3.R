library(readr)
library(ggplot2)

ran <- read_csv('Data/Random-Data-Set/CNA-100/test_cna_random1.csv')
res <- read_csv('Data/Distribution-Data-Set/CNA-100/test_cna_distribution1.csv')
imp <- read_csv('Data/Imputation-Data-Set/cna-50/CNA-imputed.csv')

real <- imp[imp$labels == 'real',]
ran <- ran[ran$labels == 'phony',]
res <- res[res$labels == 'phony',]
imp <- imp[imp$labels == 'phony',]

plotter <- function(data,gene1,gene2,title){
  indexes <- sort(match(c(gene1,gene2),colnames(data)))
  print(indexes)
  names <- colnames(data)
  names[indexes] <- c('One','Two')
  colnames(data) <- names
  
  pv <- cor(data$One, data$Two, method='spearman')
  pv <- round(pv, digits = 2)
  print(pv)
  p <- ggplot(data=data,aes(x=One,y=Two)) + geom_point(size = .75) +
    xlab(gene1) + 
    ylab(gene2) + 
    ggtitle(paste(title,'\nSpeareman: ',pv, sep='')) +
    theme_bw() +
    theme(plot.title = element_text(size=6), 
          legend.text=element_text(size=6),
          legend.title=element_text(size=6),
          axis.title.y=element_text(size=6),
          axis.title.x=element_text(size=6),
          axis.text.x=element_text(size=6),
          axis.text.y=element_text(size=6),
          panel.grid.minor = element_line(size = .5), 
          panel.grid.major = element_line(size = .5))
  return(p)
}
impAdjP <- plotter(imp,'HES4','PLEKHN1','Adjacent Gene Pair, Imputation Data, ')


ranAdjP <- plotter(ran,'HES4','PLEKHN1','Adjacent Gene Pair, Random Data ')
resAdjP <- plotter(res,'HES4','PLEKHN1','Adjacent Gene Pair, Resampled Data ')
impAdjP <- plotter(imp,'HES4','PLEKHN1','Adjacent Gene Pair, Imputation Data ')
realAdjP <- plotter(real,'HES4','PLEKHN1','Adjacent Gene Pair, Real Data ')

ranDisP <- plotter(ran,'OR4F5','DFFB','Distant Gene Pair, Random Data ')
resDisP <- plotter(res,'OR4F5','DFFB','Distant Gene Pair, Resampled Data ')
impDisP <- plotter(imp,'OR4F5','DFFB','Distant Gene Pair, Imputation Data ')
realDisP <- plotter(real,'OR4F5','DFFB','Distant Gene Pair, Real Data ')

p <- ggarrange(realAdjP,realDisP,ranAdjP,ranDisP,resAdjP,resDisP,impAdjP,impDisP,
          labels = c('A','B','C','D','E','F','G','H'), nrow=4,ncol=2,font.label = list(size = 10))

ggsave(plot = p,'Figures/Supplemental-Figure-3.png', device='png',width = 4, height = 5.5, units = c("in"),dpi=300)
ggsave(plot = p,'Figures/Supplemental-Figure-3.tiff', device='tiff',width = 4, height = 5.5, units = c("in"),dpi=300)

library(readr)
library(ggplot2)
library(tibble)
library(dplyr)


imp <- read_csv('Data/Imputation-Data-Set/cna-50/CNA-imputaion-test-1.csv')
names <- colnames(imp)
names[(length(names))] <- 'labels'
colnames(imp) <- names

res <- read_csv('Data/Distribution-Data-Set/CNA-100/test_cna_distribution1.csv')
colnames(res) <- names

ran <- read_csv('Data/Random-Data-Set/CNA-100/test_cna_random1.csv')
colnames(ran) <- names

res[res$labels == 'phony',]$labels <- 'Resampled'
imp[imp$labels == 'phony',]$labels <- 'Imputation'
ran[ran$labels == 'phony',]$labels <- 'Random'
ran[ran$labels == 'real',]$labels <- 'Real'
imp[imp$labels == 'real',]$labels <- 'Real'
res[res$labels == 'real',]$labels <- 'Real'

df <- bind_rows(ran,res[res$labels!='real',],imp[imp$labels!='real',])

labels <- as.character(df$labels)

df$labels <- NULL
df[is.na(df)] <- 0
pca <- prcomp(df)
pca$x[,1]

eigs <- pca$sdev^2
pc1 <- eigs[1] / sum(eigs)
pc2 <- eigs[2] / sum(eigs)

data <- as_tibble(pca$x)

data['Type'] <- labels

# a color blind friendly color scheme
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

ggplot(data = data,aes(x=PC1,y=PC2,color=Type)) + geom_point() + 
  xlab(paste('PC1 ',(round(pc1,digits=4) * 100),'%')) + 
  ylab(paste('PC2 ',(round(pc2,digits=4) * 100),'%')) + 
  theme_bw() +
  geom_point(size = 3) + 
  theme(plot.title = element_text(size=48), 
        legend.text=element_text(size=20),
        legend.title=element_text(size=20),
        axis.title.x=element_text(size=20),
        axis.title.y=element_text(size=20),
        panel.grid.minor = element_line(size = 1), 
        panel.grid.major = element_line(size = 1)) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)

ggsave('Figures/Figure-1.png', device='png',width = 12.5, height = 10, units = c("in"),dpi = 300)
ggsave('Figures/Figure-1.tiff', device='tiff',width = 12.5, height = 10, units = c("in"),dpi = 300)

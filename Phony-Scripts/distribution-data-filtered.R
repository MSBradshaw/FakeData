library(readr)
library(tibble)
library(ggplot2)

set.seed(0)

setwd('C:/Users/Michael/Documents/Holden/')
cna <- read_tsv('Data/Data-Uncompressed-Original/CNA.cct')

#get the protein names
names <- unlist(cna['idx'])
#remove the row names
cna <- cna[,2:ncol(cna)]
#transpose the data
cna <- as.tibble(t(cna))

colnames(cna) <- names

#plot the dist of the real data
plot <- ggplot(cna, aes(x=OR4F5)) + geom_density()
plot

cna_clone <- cna

#plot the fake data
cna_clone$fakeOR4F5 <- sample(unlist(cna$OR4F5),50)

plot <- ggplot(cna, aes(x=fakeOR4F5)) + geom_density()
plot

#plot the fake random data
man_val <- max(as.numeric(unlist(cna$OR4F5)))
min_val <- min(as.numeric(unlist(cna$OR4F5)))
cna_clone$randOR4F5 <- runif(80, man_val, min_val)

plot <- ggplot(cna, aes(x=randOR4F5)) + geom_density()
plot

#function take from: https://rdrr.io/cran/stackoverflow/src/R/chunk2.R
chunk2 <- function(x,n) split(x, cut(seq_along(x), n, labels = FALSE))

#creates phony samples by randomly sampling from the list of values found for each protein
create_phony_samples <- function(df,n_samples=2) {
  #get n_samples from each protein's distribution
  phonies <- apply(df,2,function(col){
    sample(col,n_samples)
  })
  phony_list <- chunk2(as.numeric(unlist(phonies)),n_samples)
  phony_tibble <- tibble(phony_list[[1]])
  for(i in seq(2,n_samples)) {
    phony_tibble[,i] <- phony_list[[i]]
  }
  return( as.tibble(t(phony_tibble)) )
}

#number of phonies samples to be created
number_of_phonies <- 50

#create the phonie samples
phonies <- create_phony_samples(cna,number_of_phonies)

#add classificiation labels to the data
phonies['labels'] <- replicate(number_of_phonies,'phony')
cna['labels'] <- replicate(nrow(cna),'real')

#combind phony and real data
cna[101:150,] <- phonies


cna_numerical['labels'] <- NULL
cna_numerical[is.na(cna_numerical)] <- 0
cna_pca <- prcomp(cna_numerical)
cna_pca['labels'] <- cna$labels
pca_tibble <- tibble(pc1=cna_pca$x[,1],pc2=cna_pca$x[,2],pc3=cna_pca$x[,3],pc4=cna_pca$x[,4],pc5=cna_pca$x[,5],pc6=cna_pca$x[,6],labels=cna$labels)

#PCA plot
plot <- ggplot(pca_tibble,aes(x=pc1,y=pc2,colour=labels)) + geom_point() +
  ggtitle('PCA Real & Fake Distribution Data') + 
  xlab(paste('PC1',summary(cna_pca)$importance[2,1])) +
  ylab(paste('PC2',summary(cna_pca)$importance[2,2])) 
plot
ggsave('Analysis-Scripts/Distribution-Analysis/pca-normal.png',plot)

#remove the real points that differ greatly from the fake
pca_filtered <- pca_tibble[(pca_tibble$pc2 > 0 & pca_tibble$pc1 < 10),]
plot <- ggplot(pca_filtered,aes(x=pc1,y=pc2,colour=labels)) + geom_point() +
  ggtitle('PCA Real & Fake Distribution Data') + 
  xlab(paste('PC1',summary(cna_pca)$importance[2,1])) +
  ylab(paste('PC2',summary(cna_pca)$importance[2,2]))
plot
ggsave('Analysis-Scripts/Distribution-Analysis/pca-filtered.png',plot)

plot <- ggplot(pca_filtered,aes(x=pc3,y=pc4,colour=labels)) + geom_point() +
  ggtitle('PCA Real & Fake Distribution Data') + 
  xlab(paste('PC1',summary(cna_pca)$importance[2,3])) +
  ylab(paste('PC2',summary(cna_pca)$importance[2,4]))
plot

plot <- ggplot(pca_filtered,aes(x=pc5,y=pc6,colour=labels)) + geom_point() +
  ggtitle('PCA Real & Fake Distribution Data') + 
  xlab(paste('PC1',summary(cna_pca)$importance[2,5])) +
  ylab(paste('PC2',summary(cna_pca)$importance[2,6]))
plot



cna_filtered <- cna[(pca_tibble$pc2 > 0 & pca_tibble$pc1 < 10),]
real_filtered <- cna_filtered[cna_filtered$labels=='real',]
phony_filtered <- cna_filtered[cna_filtered$labels=='phony',]

#takes in a tibble
#makes two random 50% splits of the tibble
#returns a list containing the two newly created tibbles
get_random_halves<- function(df){
  indicies <- seq(1,nrow(df))
  train_indicies <- sample.int(nrow(df), (nrow(df)/2))
  test_indicies <- setdiff(indicies, train_indicies)
  return(list(df[train_indicies,],df[test_indicies,]))
}

#get random splits of the phony and real datasets
phony_splits <- get_random_halves(phony_filtered)
real_splits <- get_random_halves(real_filtered)

#extract the train and test sets from the lists return from get_random_halves()
phony_train <- phony_splits[[1]]
phony_test <- phony_splits[[2]]
real_train <- real_splits[[1]]
real_test <- real_splits[[2]]

#combind the real and fake data into the train and test sets
train <- rbind(real_train,phony_train)
test <- rbind(real_test,phony_test)

#write train and test sets as csv
write_csv(train,'Data/Distribution-Data-Set/train_cna_distribution_filtered.csv')
write_csv(test,'Data/Distribution-Data-Set/test_cna_distribution_filtered.csv')





library(readr)
library(tibble)
library(ggplot2)

make_fake_data <- function(data,train_file_name,test_file_name,plot_file_name,seed=0){
  
  set.seed(seed)
  cna <- data
  cna[is.na(cna)] <- 0
  
  #get the protein names
  names <- unlist(cna['idx'])
  #remove the row names
  cna <- cna[,2:ncol(cna)]
  #transpose the data
  cna <- as.tibble(t(cna))
  
  colnames(cna) <- names
  
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
  
  #add labels onto the list of column names
  names <- c(names,'labels')
  
  colnames(phonies) <- names
  colnames(cna) <- names
  
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
  phony_splits <- get_random_halves(phonies)
  real_splits <- get_random_halves(cna)
  
  #extract the train and test sets from the lists return from get_random_halves()
  phony_train <- phony_splits[[1]]
  phony_test <- phony_splits[[2]]
  real_train <- real_splits[[1]]
  real_test <- real_splits[[2]]
  
  #combind the real and fake data into the train and test sets
  train <- rbind(real_train,phony_train)
  test <- rbind(real_test,phony_test)
  #give column names to the train and test
  colnames(train) <- names
  colnames(test) <- names
  
  #write train and test sets as csv
  write_csv(train,train_file_name)
  write_csv(test,test_file_name)
  
  #make a PCA plot of the combind data
  combind <- test
  combind[(nrow(combind)+1):(nrow(combind)+nrow(train)),] <- train
  pca <- prcomp(combind[,(1:ncol(combind)-1)])
  summary_pca <- summary(pca)
  pca_tibble <- tibble('pc1'=pca$x[,1],'pc2'=pca$x[,2],'label'=combind$labels)
  plot <- ggplot(data = pca_tibble, aes(x=pc1,y=pc2,colour=label)) +
    geom_point() +
    ggtitle('PCA Resampling Transcriptomics') +
    xlab(paste('PC1',summary_pca$importance[2,1]) ) +
    ylab(paste('PC2',summary_pca$importance[2,2]) )
  plot
  ggsave(plot_file_name,plot)
}


setwd('C:/Users/Michael/Documents/Holden/')
inputdata <- read_tsv('Data/Data-Uncompressed-Original/transcriptomics.cct')
for(i in seq(1,100)){
  train_name <- paste('Data/Distribution-Data-Set/Transcriptomics-100/train_transcriptomics_distribution',i,'.csv',sep='')
  test_name <- paste('Data/Distribution-Data-Set/Transcriptomics-100/test_transcriptomics_distribution',i,'.csv',sep='')
  plot_name <- paste('Data/Distribution-Data-Set/Transcriptomics-100/plots/pca-resampling-transcriptomics',i,'.png',sep='')
  make_fake_data(inputdata,train_name,test_name,plot_name,i)
  print(i)
}

inputdata <- read_tsv('Data/Data-Uncompressed-Original/CNA.cct')
for(i in seq(1,100)){
  train_name <- paste('Data/Distribution-Data-Set/CNA-100/train_cna_distribution',i,'.csv',sep='')
  test_name <- paste('Data/Distribution-Data-Set/CNA-100/test_cna_distribution',i,'.csv',sep='')
  plot_name <- paste('Data/Distribution-Data-Set/CNA-100/plots/pca-resampling-cna',i,'.png',sep='')
  make_fake_data(inputdata,train_name,test_name,plot_name,i)
  print(i)
}

inputdata <- read_tsv('Data/Data-Uncompressed-Original/proteomics.cct')
for(i in seq(1,100)){
  train_name <- paste('Data/Distribution-Data-Set/Proteomics-100/train_proteomics_distribution',i,'.csv',sep='')
  test_name <- paste('Data/Distribution-Data-Set/Proteomics-100/test_proteomics_distribution',i,'.csv',sep='')
  plot_name <- paste('Data/Distribution-Data-Set/Proteomics-100/plots/pca-resampling-proteomics',i,'.png',sep='')
  make_fake_data(inputdata,train_name,test_name,plot_name,i)
  print(i)
}
library(readr)
library(tibble)
library(ggplot2)

setwd('/fslhome/mbrad94/Holden2.0/')
#cna <- read_tsv('Data/Data-Uncompressed-Original/CNA.cct')
make_fake_data <- function(data,train_file_name,test_file_name,plot_file_name,seed=0){
  set.seed(seed)
  #get the protein names
  names <- unlist(data['idx'])
  #remove the row names
  cna <- data[,2:ncol(data)]
  cna[is.na(cna)] <- 0
  #transpose the data
  cna <- as.tibble(t(cna))
   
  #get the min and the max of each protein
  maxs <-  apply(cna,2,function(col){
    max(as.numeric(unlist(col)))
  })
   
  mins <-  apply(cna,2,function(col){
    min(as.numeric(unlist(col)))
  })
  
   
  #given a set of data (df) this function will create additional rows of data for this dataset
  #for each protein randomly a random value between the min and max for that protien will be selected
  #a tibble with phony new data, samples being represented as rows
  create_phony_samples <- function(df,n_samples=1) {
    maxs <-  apply(df,2,function(col){
      max(as.numeric(unlist(col)))
    })
    mins <-  apply(df,2,function(col){
      min(as.numeric(unlist(col)))
    })
    minmax <- tibble(mins,maxs)
    phonies <- tibble()
    for(i in seq(1,n_samples)) {
      phony <- apply(minmax,1,function(row){
        runif(1, row[1], row[2])
      })
      if(i == 1) {
        phonies <- tibble(phony)
        colnames(phonies) <- c('phony_1')
      }else{
        phonies[paste('phony_',i,sep='')] <- phony
      }
    }
    return(as.tibble(t(phonies)))
  }
   
  #number of phonies samples to be created
  number_of_phonies <- 50
   
  #create the phonie samples
  phonies <- create_phony_samples(cna,number_of_phonies)
   
  #add classificiation labels to the data
  phonies['labels'] <- replicate(number_of_phonies,'phony')
  cna['labels'] <- replicate(nrow(cna),'real')
   
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
   
  #add labels onto the list of column names
  names <- c(names,'labels')
   
  #combind the real and fake data into the train and test sets
  train <- rbind(real_train,phony_train)
  test <- rbind(real_test,phony_test)
   
  #give column names to the train and test
  colnames(train) <- names
  colnames(test) <- names
   
  #write train and test sets as csv
  write_csv(train,train_file_name)
  write_csv(test,test_file_name)
  
}

#setwd('C:/Users/Michael/Documents/Holden2/')
inputdata <- read_tsv('Data/Data-Uncompressed-Original/transcriptomics.cct')
for(i in seq(1,100)){
  train_name <- paste('Data/Random-Data-Set/Transcriptomics-100/train_transcriptomics_random',i,'.csv',sep='')
  print(train_name)
  test_name <- paste('Data/Random-Data-Set/Transcriptomics-100/test_transcriptomics_random',i,'.csv',sep='')
  plot_name <- paste('Data/Random-Data-Set/Transcriptomics-100/plots/pca-random-transcriptomics',i,'.png',sep='')
  make_fake_data(inputdata,train_name,test_name,plot_name,i)
  print(i)
}


inputdata <- read_tsv('Data/Data-Uncompressed-Original/CNA.cct')
for(i in seq(1,100)){
  train_name <- paste('Data/Random-Data-Set/CNA-100/train_cna_random',i,'.csv',sep='')
  test_name <- paste('Data/Random-Data-Set/CNA-100/test_cna_random',i,'.csv',sep='')
  plot_name <- paste('Data/Random-Data-Set/CNA-100/plots/pca-random-cna',i,'.png',sep='')
  make_fake_data(inputdata,train_name,test_name,plot_name,i)
  print(i)
}

inputdata <- read_tsv('Data/Data-Uncompressed-Original/proteomics.cct')
for(i in seq(78,100)){
  train_name <- paste('Data/Random-Data-Set/Proteomics-100/train_proteomics_random',i,'.csv',sep='')
  test_name <- paste('Data/Random-Data-Set/Proteomics-100/test_proteomics_random',i,'.csv',sep='')
  plot_name <- paste('Data/Random-Data-Set/Proteomics-100/plots/pca-random-proteomics',i,'.png',sep='')
  make_fake_data(inputdata,train_name,test_name,plot_name,i)
  print(i)
}






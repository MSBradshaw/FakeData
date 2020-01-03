library(readr)
library(tibble)
library(dplyr)

data_og <- read_tsv('Data/Data-Uncompressed-Original/CNA.cct')
data_og[is.na(data_og)] <- 0

make_fake <- function(data, inds, percent){
  data <<- data
  # inds are which indexes to duplicate and fake
  to_fake <- data[,inds]
  #loop over all the to-be faked samples
  first <<- TRUE
  fakes <- apply(to_fake,2,function(sample){
    # pick indexes of values to-be faked
    # loop over each value in a to-be faked sample
    vals_to_fake <- sample(nrow(data),(nrow(data)*percent))
    iter <<- 1
    new_sample <- sapply(sample,function(i){
      new_val <- i
      if(iter %in% vals_to_fake){
        random_index <- sample(ncol(data),1)
        new_val <- data[iter,random_index]
      }
      iter <<- iter + 1
      new_val <- as.numeric(new_val)
      if(!is.numeric(new_val)){
        print(new_val)
      }
      return(new_val)
    })
    new_sample <- as.numeric(new_sample)
    return(new_sample)
  })
  return(fakes)
}

make_test_train(data,inds,percent,trainfile,testfile){
  # returns a matrix of the 50 fake samples
  d <- make_fake(data,inds,percent)
  colnames(d) <- paste('fake-',c(1:ncol(d)),sep='')
  
  # split both in half
  # choose half
  test_cols_fake <- sample(c(1:ncol(d)),(ncol(d)/2))
  train_cols_fake <- setdiff(c(1:ncol(d)),test_cols_fake)
  
  test_cols_real <- sample(c(1:ncol(data)),(ncol(data)/2))
  train_cols_real <- setdiff(c(2:ncol(data)),test_cols_real)
  
  #divide into training and test sets
  train_real <- data[,train_cols_real]
  test_real <- data[,test_cols_real]
  
  train_fake <- d[,train_cols_fake]
  test_fake <- d[,test_cols_fake]
  
  colnames(train_real) <- NULL
  colnames(test_real) <- NULL
  colnames(train_fake) <- NULL
  colnames(test_fake) <- NULL
  
  trr <- t(train_real)
  ter <- t(test_real)
  trf <- t(train_fake)
  tef <- t(test_fake)
  
  train <- bind_rows(data.frame(trr),data.frame(trf))
  test <- bind_rows(data.frame(ter),data.frame(tef))
  
  # add colnames
  # add training labels
  labels <- c(rep('real',50), rep('phony',25))
  train$labels <- labels
  test$labels <- labels
  
  colnames(train) <- data$idx
  colnames(test) <- data$idx
  
  write_csv(train,trainfile)
  write_csv(test,testfile)
}

x <- make_fake(data_og[,2:ncol(data_og)],samples,.5)

#for the percentages 10 - 90
#for the replicates 20 each
for(p in c(.1,.2,.3,.4,.5,.6,.7,.8,.9)){
  for(j in seq(1:20)){
    test_name <- paste('Data/Partially-Fake/CNA-',p,'/partially-fake-test-',j,'.csv',sep='')
    train_name <- paste('Data/Partially-Fake/CNA-',p,'/partially-fake-train-',j,'.csv',sep='')
    samples <- sample(1:100,50)
    make_fake(data_og[,2:ncol(data_og)],samples,p)
  }
}
















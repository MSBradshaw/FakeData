library(readr)
library(tibble)
library(ggplot2)

data <- read_tsv('Data/Data-Uncompressed-Original/proteomics.cct')


#get the protein names
names <- unlist(data['idx'])
#remove the row names
data <- data[,2:ncol(data)]
#transpose the data
data <- as.tibble(t(data))


data['labels'] <- replicate(nrow(data), "real")
names <- c(names,'labels')
colnames(data) <- names

print(tail(names))

for( i in c(1:50)){
  print(i)
  n0 <- read_csv(paste(i,'-0-CNA3.csv',sep=''))
  n1 <- read_csv(paste(i,'-1-CNA3.csv',sep=''))
  n2 <- read_csv(paste(i,'-2-CNA3.csv',sep=''))
  n3 <- read_csv(paste(i,'-3-CNA3.csv',sep=''))
  n4 <- read_csv(paste(i,'-4-CNA3.csv',sep=''))
  n <- c(as.numeric(n0[1,]),as.numeric(n1[1,]),as.numeric(n2[1,]),as.numeric(n3[1,]),as.numeric(n4[1,]),'phony')
  # add the new row to the original data
  data[(nrow(data)+1),] <- n
  print(dim(data))
}
print(dim(data))
names <- as.character(colnames(data))
write_csv(data,'CNA-imputed.csv')

get_random_halves<- function(df){
  print(0)
  indicies <- seq(1,nrow(df))
  print(1)
  train_indicies <- sample.int(nrow(df), (nrow(df)/2))
  print(2)
  test_indicies <- setdiff(indicies, train_indicies)
  print(3)
  return(list(df[train_indicies,],df[test_indicies,]))
}

#get random splits of the phony and real datasets
phony_splits <- get_random_halves(data[data$labels=='phony',])
real_splits <- get_random_halves(data[data$labels=='real',])

#extract the train and test sets from the lists return from get_random_halves()
phony_train <- phony_splits[[1]]
phony_test <- phony_splits[[2]]
real_train <- real_splits[[1]]
real_test <- real_splits[[2]]

#combind the real and fake data into the train and test sets
train <- rbind(real_train,phony_train)
test <- rbind(real_test,phony_test)

print(tail(names))
#give column names to the train and test
colnames(train) <- names
colnames(test) <- names

#write train and test sets as csv
write_csv(train,'CNA3-imputation-train.csv')
write_csv(test,'CNA3-imputation-test.csv')

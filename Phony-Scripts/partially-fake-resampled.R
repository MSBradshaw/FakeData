#THIS SCRIPT TAKES 2 COMMAND LINE ARGUMENTS
# 1: the starting index
# 2: Percent fake, how much of each sample to fake as an integer
library(readr)
library(tibble)
args = commandArgs(trailingOnly=TRUE)

make_partially_fake <- function(data,portion,filename){
  #remove the fake samples
  data <- data[data$labels != 'phony',]
  #randomly pick half the samples
  fake_ids <- sample(1:nrow(data),floor(nrow(data)/2),replace = FALSE)
  real_ids <- c(1:nrow(data))[!(c(1:nrow(data)) %in% fake_ids)]
  col_range <- c(1:ncol(data))
  for( i in fake_ids){
    #selected a number of ids based of the portion of data to be made fake
    selected_ids <- sample(col_range,floor(ncol(data)*portion))
    print('length: ')
    print(length(selected_ids))
    for(j in selected_ids){
      #select a value actually seen for that protein
      num <- sample(as.numeric(unlist(data[,j])),1,replace = TRUE)
      data[i,j] <- num
    }
  }
  #relabel the not partially fake data as phony
  data$labels[fake_ids] <- "phony"
  write_csv(data,filename)
}

#setwd('/Users/mibr6115/Holden/')
setwd('/Users/michael/Holden/')
inputdata <- read_csv(paste('Data/Distribution-Data-Set/CNA-100/test_cna_distribution',args[1],'.csv',sep=''))
i=0

args <- commandArgs(trailingOnly = TRUE)
print('args')
print(args)
num <- as.numeric(args[1])
percent_fake = as.numeric(args[2])/100
print(num)
test_name <- paste('Data/Partially-Fake-Resampled/CNA/test_partially_resampled',num,'_fake_',percent_fake,'.csv',sep='')
make_partially_fake(inputdata,percent_fake,test_name)

# these files being created as test sets will be created using originl CNA Resampled Test Data, 
# this way they can be train on an already existing training file of all fake data

















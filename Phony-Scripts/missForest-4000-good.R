library(missForest)
library(readr)
library(tibble)

args = commandArgs(trailingOnly=TRUE)

file = args[1] # the input file
row_number = as.numeric(args[2]) #the sample number to duplicate
seed = as.numeric(args[3]) # the random seed to be used
start_portion = as.numeric(args[4]) # a number 0 - 4
base_name = paste(args[5],'.csv',sep='') #the type of data being run, CNA, Pro or Tran

outfile = paste(row_number,start_portion,base_name,sep='-')
print(outfile)
set.seed(seed)

#file = 'Holden/Data/Data-Uncompressed-Original/CNA.cct'

#read in the data, the .cct file is just a normal tab delimited file
data <- read_tsv(file)

#transpose the data and give it colnames like it should
transposed <- as.tibble(t(data))
colnames(transposed) <- transposed[1,]

#remove the first row which contains the potein names
input <- transposed[2:nrow(transposed),]

input[, seq(1,ncol(input))] <- sapply(input[, seq(1,ncol(input))], as.numeric)

create_part_of_person <- function(data,row,start,span=100){

  #calculate the end point for the knockouts
  end <- start + span - 1
  if(start == 3001){
    end <- end - 1
  }
  if(end > ncol(data)){
    end = ncol(data)
  }
  #duplicate the given row
  input_mis <- data
  input_mis[(nrow(input_mis)+1),] <- input_mis[row,]
  
  #create the missing values
  input_mis[nrow(input_mis),start:end] <- NA
  
  #impute the missing values
  is_out <-missForest(as.data.frame(input_mis))
  
  #extract the new data
  new <- as.numeric(is_out$ximp[nrow(is_out$ximp),start:end])

  #print(new)
  
  return (new)
}

#this will chunk the 
start <- start_portion * 4000
end <- start + 3000 #this is plus 3000 because it marks the last start spot not the end point
if(start == 0 ){
  start <- 1
  end <- 4000
}
if(end > ncol(input)){
  end <- ncol(input)
}

row <- c()
by = 1000
print('---')
print(by)
print(start)
print(end)
for( i in seq(start,end,by=by) ){
  print(i)
  ptm <- proc.time()
  new <- create_part_of_person(input,row_number,i,by)
  row <- c(row,new)
  print( paste( 'row len: ',length(row) ) )
  print(paste('Time to generate: ',i,sep=''))
  print(proc.time() - ptm)  
}
out <- as_tibble(t(tibble(row)))

write_csv(out,outfile)




#input 1 is the input data file

# $1 is the input data
# $2 is the random need
# $3 is the output file data type label, for CNA, Pro or Tran
# i is the row number
# 0-4 is the portion of the data to input, broken into chuncks of 4 1000
vals=($(seq -s ' ' 48 50))
for i in "${vals[@]}"
do
    echo  cna-$i
    bash create-missForest-people.sh ../Data/Data-Uncompressed-Original/CNA.cct $i cna-$i
done

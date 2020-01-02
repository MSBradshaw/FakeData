#input 1 is the input data file

# $1 is the input data
# $2 is the random need
# $3 is the output file data type label, for CNA, Pro or Tran
# i is the row number
# 0-4 is the portion of the data to input, broken into chuncks of 4 1000
vals=($(seq 50))
for i in "${vals[@]}"
do
    echo $i
    sbatch missForest-person-job.sh $1 $i $2 0 $3
    sbatch missForest-person-job.sh $1 $i $2 1 $3
    sbatch missForest-person-job.sh $1 $i $2 2 $3
    sbatch missForest-person-job.sh $1 $i $2 3 $3
    sbatch missForest-person-job.sh $1 $i $2 4 $3
done


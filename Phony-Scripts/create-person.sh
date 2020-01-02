#input 1 is the input data file
#input 2 is the index that needs to be duplicated and faked

#17156 is the number of columns in the CNA dataset, this will need to be tweak for the other two datasets
vals=($(seq 0 2400 17156))
for i in "${vals[@]}"
do
    echo $i $(expr $i + 2400)
    bash imputation.sh $1 $2 $i $(expr $i + 2400)
done

#tells the job to not exit untill all the parallel threads have finished
wait

echo $1
for i in {1..100}
do
	/Users/mibr6115/python3/Python-3.7.4/python Classification-Downsample-Features.py ../../Data/Imputation-Data-Set/CNA-imputation-train.csv ../../Data/Imputation-Data-Set/CNA-imputation-test.csv results.csv Imputed $1
done

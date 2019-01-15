# Cancer-methylation project
# Introduction
In the field of cancer diagnose, lots of DNA methylation markers used to predict cancer have been mined. But there is no one research that tries to find DNA methylation marker used to diagnose various types of cancer. In this research, we tried to mine DNA methylation markers, including CpG markers and promoter markers, that could be used to differ various types of cancer sample from matched normal sample. We collected whole-genome methylation data of 9000 cancer samples and 4000 matched normal sample from TCGA and GEO, and we divided all samples into three data set, including train data set, validation data set and test data data set. We used machine learning to analyse and mine DNA methylation markers, specially, we classified and predicted cancer and normal sample by deep learning. Consequently, we mined two types of marker. One type of marker contains 12 DNA methylation sites. In the test data set, prediction accuracy is 96%, sensitivity is 96%, specificity is 93%. Another type of marker contains 13 promoter areas. In the test data set, prediction accuracy is 96%, sensitivity is 96%, specificity is 93%. In our test data set, some types of cancer have not been trained in the train data set, but we can still use our deep learning model to classify cancer samples and normal samples in these types of cancer, which means our model can not only diagnose trained cancer, but also diagnose cancer that are never being trained.

# prerequisites
Python (3.6). Python 3.6.4 is recommended.
Numpy (>=1.14.2)
tensorflow-gpu (>=1.4.0)
Scikit-learn (>=0.19.1)
matplotlib (>=2.1.1)

# Data
Whole-genome methylation data, such as methylation beta value from Illumina’s Infinium HumanMethylation450 BeadChip. The format is as follow.
![image](https://github.com/BiaoLiu2017/Cancer-methylation/blob/master/images/input.png)

Each column is a sample, and each row is a marker(cg id should be sorted from small to large). If there is just only one sample, the file will have only two column. It is fine. And separator is 'tab'. The file should be renamed as 'input.txt'.

# Process & predict

The documents in 'files' directory is the results of GSE108462 (prostate cancer cfDNA).

## 1)Get CpG markers matrix
python get_CpG_matrix.py input.txt CpG_matrix.txt

## 2)Get promoter markers matrix
python trans_array.py input.txt input_trans.txt

python get_promoter_matrix.py input_trans.txt promoter_matrix.txt

## 3)Standardization
python standard_CpG.py CpG_matrix.txt CpG_matrix_standard.txt

python standard_promoter.py promoter_matrix.txt promoter_matrix_standard.txt

## 4)Predict
python predict_CpG.py CpG_matrix_standard.txt sigmoid_CpG.txt predict_CpG.txt

python predict_promoter.py promoter_matrix_standard.txt sigmoid_promoter.txt predict_promoter.txt

# Reference
BiaoLiu, et al. (2018) Mining DNA methylation markers for cancer prediction by machine learning.

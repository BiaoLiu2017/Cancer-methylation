# Cancer-methylation project
# Introduction
In the field of cancer diagnosis, many DNA methylation markers have been mined. But few researches have tried to mine DNA methylation markers used to diagnose diverse types of cancers, namely pan-cancer. In this study, we tried to mine DNA methylation markers used to differentiate cancer samples from matched normal samples in diverse types of cancers. We collected whole-genome methylation data of twenty-seven types of cancers containing 10,140 cancer samples and 3,386 normal samples, and we divided all samples into five data sets, including a train data set, a validation data set and three test data sets. We used machine learning to mine DNA methylation markers, and specially, we constructed prognostic prediction models by deep learning. We mined two types of markers: twelve CpG markers and thirteen promoter markers. Three of twelve CpG markers and four of thirteen promoter markers locate at cancer-related genes. For CpG markers, average sensitivity and specificity of all three test data sets (twenty-seven types of cancers, containing 4,943 cancer samples and 1,009 normal samples) were 92.8% and 90.1% respectively. And for promoter markers, average sensitivity and specificity of all three test data sets were 89.8% and 81.1% respectively. We tested performance of these markers in cell free DNA methylation data of 163 prostate cancer samples and 2 unrelated normal samples. Sensitivity and specificity for CpG markers were both 100%, and for promoter markers were 92% and 100% respectively. Because lack of abundant normal cell free DNA samples, specificity in cell free samples remains to verify further. To conclude, we mined methylation markers to diagnose diverse cancers, which might be applied to liquid biopsy of pan-cancer.

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

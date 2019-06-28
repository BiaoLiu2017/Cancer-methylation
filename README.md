# Cancer-methylation project
# Abstract
For cancer diagnosis, many DNA methylation markers have been identified. However, few studies have tried to find DNA methylation markers to diagnose diverse cancer types simultaneously, i.e., pan-cancers. In this study, we tried to identify DNA methylation markers to differentiate cancer samples from the respective normal samples in pan-cancers. We collected whole genome methylation data of 27 cancer types containing 10,140 cancer samples and 3,386 normal samples, and divided all samples into five data sets, including one training data set, one validation data set and three test data sets. We applied machine learning to identify DNA methylation markers, and specifically, we constructed diagnostic prediction models by deep learning. We identified two categories of markers: 12 CpG markers and 13 promoter markers. Three of 12 CpG markers and four of 13 promoter markers locate at cancer-related genes. With the CpG markers, our model achieves an average sensitivity and specificity on test data sets as 92.8% and 90.1%, respectively. For promoter markers, the average sensitivity and specificity on test data sets were 89.8% and 81.1%, respectively. Furthermore, in cell-free DNA methylation data of 163 prostate cancer samples, the CpG markers achieve the sensitivity as 100%, and the promoter markers achieve 92%. For both marker types, the specificity of normal whole blood is 100%. To conclude, we identified methylation markers to diagnose pan-cancers, which might be applied to the liquid biopsy of cancers.


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
BiaoLiu, et al. (2018) DNA methylation markers for pan-cancer prediction by deep learning.

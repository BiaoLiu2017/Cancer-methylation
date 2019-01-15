import sys
f_name = sys.argv[1]
fs_name = sys.argv[2]
f = open(f_name, 'r')
fs = open(fs_name, 'a')
f_gene_gc = open("gene_CGs_13promoter.txt", 'r')
dict_gc = {}
gc_name = f.readline()
gc_name = gc_name.strip('\n')
gc_list = gc_name.split('\t')
for i in range(len(gc_list)):
    dict_gc[gc_list[i]] = i
for line in f.readlines():
    line = line.strip('\n')
    val_list = line.split('\t')
    beta_mean_list = []
    for line_gene in f_gene_gc.readlines():
        line_gene = line_gene.strip('\n')
        gene_gc_list = line_gene.split('\t')
        beta_sum = 0.0
        num = 0
        beta_val_mean = 0.0
        for j in range(1, len(gene_gc_list)):
            if gene_gc_list[j] in dict_gc:
                if val_list[dict_gc[gene_gc_list[j]]] != 'NA':
                    num += 1
                    beta_sum += float(val_list[dict_gc[gene_gc_list[j]]])
        if num !=0 :
            beta_val_mean = beta_sum/num
            beta_val_mean = "{:.3f}".format(beta_val_mean)
            beta_mean_list.append(beta_val_mean)
        else:
            beta_mean_list.append("NA")
    seq = '\t'
    beta_mean_str = seq.join(beta_mean_list)
    fs.write(beta_mean_str)
    fs.write('\n')
    f_gene_gc.seek(0)
f.close()
fs.close()
f_gene_gc.close()





import numpy as np
import sys
import math
nm = np.loadtxt("mean_of_train_promoter.txt")
nv = np.loadtxt("var_of_train_promoter.txt")
am = []
av = []
for i in nm:
    am.append(i)
for i in nv:
    av.append(math.sqrt(i))
f = open(sys.argv[1], 'r')
fs = open(sys.argv[2], 'a')
for line in f.readlines():
    aa = line.strip('\n').split('\t')
    st = ''
    for i in range(len(aa)):
        x = format((float(aa[i]) - am[i])/av[i], '.3f')
        st += str(x)
        st += '\t'
    st = st.strip('\t')
    st += '\n'
    fs.write(st)
f.close()
fs.close()






import os
import re
import sys
f = open(sys.argv[1], 'r')
fs = open(sys.argv[2], 'a')
a = []
for line in f.readlines():
    line = line.strip('\n')
    ll = line.split('\t')
    a.append(ll)
ll_trans = list(map(list, zip(*a)))
for line_trans in ll_trans:
    stt = ''
    s = "\t"
    stt = s.join(line_trans)
    fs.write(stt)
    fs.write('\n')
f.close()
fs.close()

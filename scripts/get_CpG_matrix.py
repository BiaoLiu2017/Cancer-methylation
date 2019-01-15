import sys
f = open('CpG_12markers.txt', 'r')
f1 = open(sys.argv[1], 'r')
fs = open(sys.argv[2], 'a')
a = []
array_fs = []
for line in f.readlines():
    a.append(line.strip('\n'))
for line in f1.readlines():
    aa = line.strip('\n').split('\t')
    if aa[0] in a:
        array_fs.append(aa[1:])
        
ll_trans = list(map(list, zip(*array_fs)))
for line_trans in ll_trans:
    stt = ''
    s = "\t"
    stt = s.join(line_trans)
    fs.write(stt)
    fs.write('\n')
f.close()
f1.close()
fs.close()
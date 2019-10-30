import re

regex = r"<\/*e\d{1}>"
subst = ""

inp_file='../training_full.txt'
out_file='training_all_classes.csv'

fo=open(out_file,'w')

fo.write('sent\tlabel\n')

all_labels=['Other','Cause-Effect', 'Instrument-Agency', 'Product-Producer', 'Content-Container', 'Entity-Origin', 'Entity-Destination', 'Component-Whole', 'Member-Collection', 'Message-Topic']

line_num=0
for ln in open(inp_file,'r').readlines():
    if line_num%4==0:
        sent=ln.replace('\n','').split('\t')
        sent=sent[1].strip()
        sent=re.sub(regex, subst, sent, 0, re.MULTILINE)
    if line_num%4==1:
        label='-1'
        for li in range(len(all_labels)):
            if all_labels[li] in ln:
                label=str(li)
        if label!='-1':
            fo.write(sent[1:-1]+'\t'+label+'\n')
    line_num+=1
fo.close()

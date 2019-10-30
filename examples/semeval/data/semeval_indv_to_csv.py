import re

regex = r"<\/*e\d{1}>"
subst = ""

inp_file='../training_full.txt'
out_file='training_entity_origin.csv'

fo=open(out_file,'w')

fo.write('sent\tlabel\n')

line_num=0
for ln in open(inp_file,'r').readlines():
    if line_num%4==0:
        sent=ln.replace('\n','').split('\t')
        sent=sent[1].strip()
        sent=re.sub(regex, subst, sent, 0, re.MULTILINE)
    if line_num%4==1:
        if 'Entity-Origin' in ln:
            label='1'
        else:
            label='0'
        fo.write(sent[1:-1]+'\t'+label+'\n')
    line_num+=1
fo.close()

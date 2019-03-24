# -*- coding: utf-8 -*-
import subprocess
from sys import platform

def calculate_ter(references,hypothesis,temp_ref_file="refs.temp",temp_hyp_file="hyps.temp"):
    temp_references = []
    temp_hypothesis = []
    
    for idx,sents in enumerate(zip(references,hypothesis)):
        temp_references.append(sents[0]+"\t(sent{})".format(idx+1))
        temp_hypothesis.append(sents[1]+"\t(sent{})".format(idx+1))
        
    with open(temp_ref_file,'w',encoding="utf-8") as fref:
        fref.writelines('%s\n' % refs for refs in temp_references)
    
    with open(temp_hyp_file,'w',encoding="utf-8") as fhyp:
        fhyp.writelines('%s\n' % hyps for hyps in temp_hypothesis)
    
    cmd_line = "java -jar tercom-0.7.25/tercom.7.25.jar -r " + temp_ref_file + " -h " + temp_hyp_file
    out = subprocess.check_output(cmd_line,shell=True)
    out = out.decode("utf-8")
    ter = 0.0
    if platform=='linux':
        ter = out.split("\n")[-5].split()[2]
    elif platform=='win32':
        ter = out.split("\r\n")[-5].split()[2]
    
    return ter

#t = calculate_ter(['ram is a good but not a boy','we should study more but not often','i doesnt want that now'],['<UNK>','we should study more','he doesnt want that now'])
#print(t)
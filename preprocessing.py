import sentencepiece as spm
import numpy as np
import pickle
import gzip

from tqdm import tqdm
#TODO: masked language model & nest sentence prediction

path = '/hdd/user16/HT/'

spm_model = '/hdd/user16/HT/30k-clean.model'
sp = spm.SentencePieceProcessor()
sp.load(spm_model)

files = ['xaa', 'xab', 'xac', 'xad', 'xae', 'xaf', 'xag']
files = [path+x for x in files]

# read data
docs = []

for file in files:
    with open(file, 'r') as f:
        tmp = []
        current_head = 1e9
        for i, x in enumerate(tqdm(f)):
            line = line.decode('utf-8')
            if i == (current_head+1):
                continue
            if line == '\n':
                current_head = i
                docs.append(tmp)
                tmp = []
            tmp.append(sp.EncodeAsIds(line))


## Abandoned. See notebook_preprocess.
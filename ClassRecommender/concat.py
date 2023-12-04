import glob
from bs4 import BeautifulSoup
import json
import csv

files = glob.glob("data_HTML/GLO/*.tsv")

train = []
test = []

fileIndex = 0
for file in files:
    print(str(file).replace(' ', '').replace("data_HTML/GLO\\", ''))

    f = open(file,'r', encoding='utf-8')
    rdr = csv.reader(f, delimiter='\t')
    
    i = 0
    for line in rdr:
        # print(line)
        if(i % 5 != 0):
            train.append((line[0].replace('\t', '').replace('\n', ''), fileIndex))
        else:
            test.append((line[0].replace('\t', '').replace('\n', ''), fileIndex))
        i = i+1
    f.close()

    fileIndex = fileIndex+1

with open('GLO_dict.tsv', "w", encoding="utf-8") as fp:
    fileIndex = 0
    for file in files:
        fp.write(str(fileIndex) + '\t' + str(file).replace(' ', '').replace("data_HTML/GLO\\", '').replace('.tsv', '') + '\n')
        fileIndex = fileIndex+1

with open('GLO_train.tsv', "w", encoding="utf-8") as fp:
    for _train in train:
        fp.write(str(_train[0]) + '\t' + str(_train[1]) + '\n')

with open('GLO_test.tsv', "w", encoding="utf-8") as fp:
    for _test in test:
        fp.write(str(_test[0])  + '\t' + str(_test[1]) + '\n')

import csv

f = open('SCI_dict.tsv','r', encoding='utf-8') # *.dict.tsv 파일은 강의 인덱스 - 강의 이름+교수명 매핑 파일
rdr = csv.reader(f, delimiter='\t')
mapper = []
for row in rdr:
    mapper.append(row[1])
f.close()

f = open('SCI_data.tsv','r', encoding='utf-8') # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
rdr = csv.reader(f, delimiter='\t')
train_data = [[] for _ in list(range(len(mapper)))]
test_data = []
for row in rdr:
    train_data[int(row[1])].append(str(row[0]))
    test_data.append(row)
f.close()

import random

train_examples = []
test_examples = []
train_count = 800 # hyper_params.
test_count = 200
batch_size = 16
epochs = 32

for k in range(train_count): # training data 생성. (강의A_강의평1, 강의A_강의평2, 강의B_강의평1) 형식의 튜플을 train_example에 저장
    i = random.randint(0, len(train_data)-1)
    j = -1
    while(j <0 or j == i):
        j = random.randint(0, len(train_data)-1)

    train_examples.append([train_data[i][random.randint(0, len(train_data[i])-1)], train_data[i][random.randint(0, len(train_data[i])-1)], train_data[j][random.randint(0, len(train_data[j])-1)]])

randomList = []
k = 0
while(k < test_count):
    l = random.randint(0, len(test_data)-1)

    if(l in randomList):
        continue
    else:
        test_examples.append(test_data[l])
        k = k+1

f = open('train_data.csv','w', encoding='utf-8')
for train_example in train_examples:
    f.write(train_example[0] + ',' + train_example[1] + ',' + train_example[2] + '\n')
f.close()

f = open('test_data.csv','w', encoding='utf-8')
for test_example in test_examples:
    f.write(test_example[0] + ',' + test_example[1] + '\n')
f.close()
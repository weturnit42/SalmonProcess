import csv
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch

'''
CLA 고전읽기영역
ETC 일반영역
FUT 미래산업과창업영역
GLO 글로벌언어와문화영역
LIT 인문과예술영역
SCI 과학과기술영역
SOC 사회와세계영역
SOF 소프트웨어영역
VIR 가상대학영역
'''

tags = ['CLA', 'ETC', 'FUT', 'GLO', 'LIT', 'SCI', 'SOC', 'SOF', 'VIR']
tag = tags[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 8000 # hyper_params.
test_count = 2000
batch_size = 16
epochs = 128

f = open('data/' + tag + '_dict.tsv','r', encoding='utf-8') # *.dict.tsv 파일은 강의 인덱스 - 강의 이름+교수명 매핑 파일
rdr = csv.reader(f, delimiter='\t')
mapper = []
for row in rdr:
    mapper.append(row[1])
f.close()

f = open('data/' + tag + '_data.tsv','r', encoding='utf-8')  # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
rdr = csv.reader(f, delimiter='\t')
test_data = [[] for _ in list(range(len(mapper)))]
temp_data = []
for row in rdr:
    test_data[int(row[1])].append(str(row[0]))
    temp_data.append(row)
f.close()

test_examples = []
f = open('test/' + tag + '_test_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

import os
from collections import OrderedDict

try:
    model_state_dict = torch.load("fineTunedModel/" + tag + "/" + tag + "_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  "_dataset_"  + str(train_count+test_count) +  ".pt", map_location=device)
except FileNotFoundError:
    os.chdir('C:/Users/SAMSUNG/Desktop/4학년 2학기/자연어처리론/팀 프로젝트/model/KR-SBERT/')
    model_state_dict = torch.load("fineTunedModel/" + tag + "/" + tag + "_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  "_dataset_"  + str(train_count+test_count) +  ".pt", map_location=device)

try:
    model.load_state_dict(model_state_dict)
except RuntimeError:
    model.load_state_dict(model_state_dict, strict=False)

import numpy as np

vectors = [[] for _ in list(range(len(mapper)))]
try:
    os.chdir("C:/Users/SAMSUNG/Desktop/4학년 2학기/자연어처리론/팀 프로젝트/SalmonProcess/KR-SBERT/")
    f = open('vector/' + tag + '/' + tag + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in enumerate(rdr):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row), 0.3396976888179779, "int(temp_data[i][1])", int(temp_data[i][1]))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

except FileNotFoundError:
    os.chdir("C:/Users/SAMSUNG/Desktop/4학년 2학기/자연어처리론/팀 프로젝트/SalmonProcess/KR-SBERT/")
    f = open('vector/' + tag + '/' + tag + '_vectors_' + str(train_count+test_count) + '.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in list(range(len(mapper))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
        for j in list(range(len(test_data[i]))):
            vector = model.encode(test_data[i][j])
            listedVector = vector.tolist()
            writer.writerow(listedVector)
        print(i, "encoding done")

    f.close()

    f = open('vector/' + tag + '/' + tag + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in enumerate(rdr):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

dimList = []
vectorData = [[] for _ in list(range(len(mapper)))]
for i in list(range(len(vectors))):
    for j in list(range(len(vectors[i]))):
        vectorData[i].append(vectors[i][j][0])
        dimList.append(np.array(vectors[i][j][0]).shape)

targetText = "논어를 잘 알려주셔서 좋습니다." #상상 강의평
targetVector = model.encode([targetText]) # targetVector는 데스트 할 text string의 sentence vector

# results = []
# answerList = []
# simList = [[] for _ in list(range(len(vectorData)))]

# for i in list(range(len(vectorData))):
#     for j in list(range(len(vectorData[i]))):
#         tmp = vectorData[i][j].astype(np.float32)

#         similarities = util.cos_sim(tmp, targetVector) # compute similarity between sentence vectors
#         simList[i].append(float(similarities))

# meanSim = []
# stdSim = []

# for sim in simList:
#     meanSim.append(np.mean(np.array(sim)))
#     stdSim.append(np.std(np.array(sim)))

# for i in list(range(len(simList))):
#     results.append((i, mapper[i], meanSim[i], stdSim[i]))

# results.sort(key = lambda x : -x[2])

# print(text, "에 적합한 강의는\n")
# for result in results[:10]:
#     # if(result[2] >= 0.5):
#     #     print(result)
#     print(result[0], result[1], round(result[2], 4), round(result[3], 4), len(simList[result[0]]))
# print("\n입니다.")
# print("=========================================")

vectors = [] 
for i in list(range(len(mapper))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
    vector = model.encode(test_data[i])
    vector = np.mean(vector, axis=0)
    vectors.append(vector)
    # print(i, "mean cal. done")

acc = 0
hitsAt3 = 0
hitsAt5 = 0
hitsAt10 = 0
rankingBasedMetric = 0

for i in list(range(test_count)):
    targetVector = model.encode([targetText]) # targetVector는 데스트 할 text string의 sentence vector
    results = []
    answerList = []
    for j in list(range(len(mapper))):
        similarities = util.cos_sim(vectors[j], targetVector) # compute similarity between sentence vectors
        results.append((j, mapper[j], float(similarities)))
    results.sort(key = lambda x : -x[2])

# from datetime import datetime
# now = datetime.now()

# fileNameString = str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + str(now.second) + "_result.txt"
# fileName = open('results/' + fileNameString, 'w', encoding='utf-8')

import unicodedata # 한글이 포함된 문자열에 간격 맞추기 솔루션을 제공하는 라이브러리
def preFormat(string, width, align='<', fill=' '):
    count = (width - sum(1 + (unicodedata.east_asian_width(c) in "WF") for c in string))
    return {
        '>': lambda s: fill * count + s, # lambda 매개변수 : 표현식
        '<': lambda s: s + fill * count,
        '^': lambda s: fill * (count / 2)
                       + s
                       + fill * (count / 2 + count % 2)
    }[align](string)

print(targetText, "에 적합한 강의는")
print("번호" + " " + "강의명" + " " + "교수명" + " " + "score")
print("="*45)
for result in results[:10]:
    printedString = [str(result[0]), result[1].split('_')[0], result[1].split('_')[1], str(round(result[2], 3))]

    print(printedString[0] + " " + printedString[1] + " " + printedString[2] + " " + printedString[3])
print("입니다.")

# import matplotlib.pyplot as plt
# plt.hist(simList[results[0][0]], bins=20)
# plt.show()
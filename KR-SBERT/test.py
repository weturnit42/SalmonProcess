# sentence_transformers 꼭 설치

import csv
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers import util
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
import torch

f = open('SCI_dict.tsv','r', encoding='utf-8') # *.dict.tsv 파일은 강의 인덱스 - 강의 이름+교수명 매핑 파일
rdr = csv.reader(f, delimiter='\t')
mapper = []
for row in rdr:
    mapper.append(row[1])
f.close()

f = open('SCI_data.tsv','r', encoding='utf-8') # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
rdr = csv.reader(f, delimiter='\t')
train_data = [[] for _ in list(range(len(mapper)))]
for row in rdr:
    train_data[int(row[1])].append(str(row[0]))
f.close()

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=torch.device('cuda:0')) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

import random

train_examples = []
train_count = 1000 # hyper_params.
batch_size = 16
epochs = 32
for k in range(train_count): # training data 생성. (강의A_강의평1, 강의A_강의평2, 강의B_강의평1) 형식의 튜플을 train_example에 저장
    if(k % 50 == 0):
        print(k)
    i = random.randint(0, len(train_data)-1)
    j = -1
    while(j <0 or j == i):
        j = random.randint(0, len(train_data)-1)

    train_examples.append(InputExample(texts=[train_data[i][random.randint(0, len(train_data[i])-1)], train_data[i][random.randint(0, len(train_data[i])-1)], train_data[j][random.randint(0, len(train_data[j])-1)]]))

train_dataset = SentencesDataset(train_examples, model) # train_dataset 생성
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size) # DataLoader 초기화
train_loss = losses.TripletLoss(model=model) # loss 정의. TripletLoss로

# with torch.no_grad():
# Tune the model
# train_loss.requires_grad_(True)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100) # fit

torch.save(model.state_dict(), "SCI_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  ".pt") # 모델 저장

# model_state_dict = torch.load("SCI_epochs_16.pt", map_location=torch.device('cuda:0'))
# model.load_state_dict(model_state_dict)

import numpy as np

vectors = [] 
for i in list(range(len(mapper))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
    vector = model.encode(train_data[i])
    vector = np.mean(vector, axis=0)
    vectors.append(vector)
    print(i, "done")

text = '생명공학이란 무엇인가를 배울 수 있었습니다.' #상상 강의평
targetVector = model.encode([text])
results = []
for i in list(range(len(mapper))):
    similarities = util.cos_sim(vectors[i], targetVector) # compute similarity between sentence vectors
    results.append((i, mapper[i], float(similarities)))
results.sort(key = lambda x : -x[2])
print(text, "에 적합한 강의는")
for result in results[:10]:
    print(result)
print("입니다.")
print("=========================================")
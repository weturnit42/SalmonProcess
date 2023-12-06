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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 8000 # hyper_params.
test_count = 2000
batch_size = 16
epochs = 64

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

train_examples = []
f = open('train_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    train_examples.append(InputExample(texts=[row[0], row[1], row[2]]))
f.close()

test_examples = []
f = open('test_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

# print("test_examples[0]", test_examples[0])
# print("test_examples[1]", test_examples[1])
# print("test_examples[2]", test_examples[2])

train_dataset = SentencesDataset(train_examples, model) # train_dataset 생성
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size) # DataLoader 초기화
train_loss = losses.TripletLoss(model=model) # loss 정의. TripletLoss로

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100) # fit

torch.save(model.state_dict(), "SCI_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  "_dataset_"  + str(train_count+test_count) +  ".pt") # 모델 저장

# model_state_dict = torch.load("SCI_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  ".pt", map_location=device)
# model.load_state_dict(model_state_dict)

f = open('SCI_data.tsv','r', encoding='utf-8') # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
rdr = csv.reader(f, delimiter='\t')
train_data = [[] for _ in list(range(len(mapper)))]
for row in rdr:
    train_data[int(row[1])].append(str(row[0]))
f.close()

import numpy as np

vectors = [] 
for i in list(range(len(mapper))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
    vector = model.encode(train_data[i])
    vector = np.mean(vector, axis=0)
    vectors.append(vector)
    print(i, "done")

acc = 0
hitsAt3 = 0
hitsAt5 = 0
hitsAt10 = 0
rankingBasedMetric = 0

for i in list(range(test_count)):
    text = test_examples[i][0] #상상 강의평
    answer = int(test_examples[i][1]) # answer label

    targetVector = model.encode([text]) # targetVector는 데스트 할 text string의 sentence vector
    results = []
    answerList = []
    for j in list(range(len(mapper))):
        similarities = util.cos_sim(vectors[j], targetVector) # compute similarity between sentence vectors
        results.append((j, mapper[j], float(similarities)))
    results.sort(key = lambda x : -x[2])

    for _ in results:
        answerList.append(_[0])
    # print(text, "에 적합한 강의는")
    # for result in results[:10]:
    #     print(result)
    # print("입니다.")
    # print("=========================================")
    
    # print(answer, answerList[:20])
    if(answerList[0] == answer): # acc 계산
        acc = acc+1
    if(answer in answerList[:3]): # hits@3 계산
        hitsAt3 = hitsAt3+1
    if(answer in answerList[:5]): # hits@5 계산
        hitsAt5 = hitsAt5+1
    if(answer in answerList[:10]): # hits@10 계산
        hitsAt10 = hitsAt10+1
    # rankingBasedMetric 계산. 정답을 n개 중에서 1등으로 예측하면 1점, 2등으로 예측하면 (1-1/n)점, 3등으로 예측하면 (1-2/n)점, ... 이런 식으로 계산
    rankingBasedMetric = rankingBasedMetric + (1 - (answerList.index(answer)-1) / len(answerList)) 

# 전체 metric은 최댓값이 1입니다.
acc = acc / test_count
hitsAt3 = hitsAt3 / test_count
hitsAt5 = hitsAt5 / test_count
hitsAt10 = hitsAt10 / test_count
rankingBasedMetric = rankingBasedMetric / test_count

# 출력.
print("acc", acc)
print("hitsAt3", hitsAt3)
print("hitsAt5", hitsAt5)
print("hitsAt10", hitsAt10)
print("rankingBasedMetric", rankingBasedMetric)
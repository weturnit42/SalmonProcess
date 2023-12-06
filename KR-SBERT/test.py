import csv
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers import util
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
import torch

f = open('SCI_dict.tsv','r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
mapper = []
for row in rdr:
    mapper.append(row[1])
f.close()

f = open('SCI_data.tsv','r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
train_data = [[] for _ in list(range(len(mapper)))]
for row in rdr:
    train_data[int(row[1])].append(str(row[0]))
f.close()

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') 

# import random

# train_examples = []
# train_count = 1000

# for k in range(train_count):
#     if(k % 50 == 0):
#         print(k)
#     i = random.randint(0, len(train_data)-1)
#     j = -1
#     while(j <0 or j == i):
#         j = random.randint(0, len(train_data)-1)

#     train_examples.append(InputExample(texts=[train_data[i][random.randint(0, len(train_data[i])-1)], train_data[i][random.randint(0, len(train_data[i])-1)], train_data[j][random.randint(0, len(train_data[j])-1)]]))

# train_dataset = SentencesDataset(train_examples, model)
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
# train_loss = losses.TripletLoss(model=model)

# # Tune the model
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=16, warmup_steps=100)
# torch.save(model.state_dict(), "SCI_epochs_16" +  ".pt")

model_state_dict = torch.load("SCI_epochs_16.pt", map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)

import numpy as np

vectors = []
for i in list(range(len(mapper))):
    vector = model.encode(train_data[i])
    vector = np.mean(vector, axis=0)
    vectors.append(vector)
    print(i, "done")

text = '생명공학이란 무엇인가를 배울 수 있었습니다.'
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
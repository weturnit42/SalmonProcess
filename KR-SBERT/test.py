import csv
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers import util
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

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
# torch.save(model.state_dict(), "test" +  ".pt")

import numpy as np

vectors = []
for i in list(range(len(mapper))):
    vector = model.encode(train_data[i])
    vector = np.mean(vector, axis=0)
    vectors.append(vector)
    print(i, "done")

text = '교수님이 귀엽습니다!'
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
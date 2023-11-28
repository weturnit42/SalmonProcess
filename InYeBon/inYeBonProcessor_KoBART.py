import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

f = open("inYeBonTextv3.txt", "r", encoding='utf-8')
text = f.read()
f.close()

text = text.replace('\n', ' ')

raw_input_ids = tokenizer.encode(text, truncation=True)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=1024,  eos_token_id=1)

print(tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True))
import numpy as np
import torch
from datasets import list_datasets, load_dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from transformers.data.datasets import squad
# datasets = list_datasets()
# pprint(datasets, compact=True)
from torch.utils.data import Dataset
device = torch.device('cuda:0')
import mybertQA
from pytorch_pretrained_bert import BertForQuestionAnswering
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model = mybertQA.bertqa()

# model = torch.load('demo2')
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
#
max_len = 512
#
#
tokenizer = BertTokenizer.from_pretrained('E:\python\model/bert_pretrain')
#
squad_train = load_dataset('E:\dataset\cmrc2018', split='train')
# squad_valid = load_dataset('cmrc2018', split='validation')
# context_valid = squad_valid['context']
# question_valid = squad_valid['question']
question_train = squad_train['question']
context_train = squad_train['context']
answer = squad_train['answers']


def encode(question_token,text_token):
   total_token = ['[CLS]'] + question_token + ['[SEP]'] + text_token + ['[SEP]']
   tokenId = tokenizer.convert_tokens_to_ids(total_token)
   padding = [0] * (512 - len(tokenId))
   mask = [1] * len(tokenId) + padding
   tokenId = tokenId + padding
   segment = [0] * (2 + len(question_token)) + [1] * (1 + len(text_token)) + [0] * len(padding)
   tokenId = torch.tensor([tokenId], dtype=torch.long)
   segment = torch.tensor([segment], dtype=torch.long)
   mask = torch.tensor([mask], dtype=torch.long)
   return tokenId,segment,mask


def package(context,question):
   total_token = None
   total_segment = None
   total_mask = None
   for i in range(0,len(context)-1,2):
      tokenId, segment, mask = encode([context[i]], [question[i]])
      tokenId2, segment2, mask2 = encode([context[i+1]], [question[i+1]])
      i+=1
      token = torch.cat([tokenId, tokenId2], dim=0)
      segmen = torch.cat([segment, segment2], dim=0)
      mas = torch.cat([mask, mask2], dim=0)
      if total_token is None:
         total_token = token
         total_segment = segmen
         total_mask = mas
      else:
         total_token = torch.cat([total_token,token],dim=0)
         total_mask = torch.cat([total_mask,mas],dim=0)
         total_segment = torch.cat([total_segment,segmen],dim=0)

   # out = bert(token, segmen, mas)
   return total_token,total_segment,total_mask

answers = []
contexts = []
questions = []

answers_val = []
contexts_val = []
questions_val = []

for idx, context in enumerate(context_train):
    if len(context)<450:
        questions.append(question_train[idx])
        contexts.append(context)
        answers.append(answer[idx])

    ##这个是val数据集
# for idx, context in enumerate(context_valid):0
#     if len(context)<450:
#         questions_val.append(question_train[idx])
#         contexts_val.append(context)
#         answers_val.append(answer[idx])


# inputdata = question_train, context_train
inputvaldata = questions_val,contexts_val
inputdata = questions, contexts


index = []
answer_end = []

for idx,i in enumerate(answers):
    index.append(int(i['answer_start'][0])+len(question_train[idx])+2)
    answer_end.append(int(i['answer_start'][0])+len(i['text'][0])+len(question_train[idx])+2)


label = index,answer_end


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.question,self.context = data_tensor
        self.start,self.end = target_tensor
    # 返回数据集大小
    def __len__(self):
        return len(self.question)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.question[index],self.context[index],self.start[index],self.end[index]

datasets = MyDataset(inputdata,label)
dataloader = DataLoader(datasets,batch_size=8,shuffle=True,sampler=None,batch_sampler=None,collate_fn=None,
pin_memory=False,drop_last=False,timeout=0,worker_init_fn=None,multiprocessing_context=None)
if __name__ == '__main__':
    epoch = 30
    model.to(device)
    for i in range(epoch):
        for idx, (question,context, start,end) in enumerate(dataloader):
            total_token, total_segment, total_mask = package(context, question)
            result = model.forward(tokenId=total_token.to(device),segment=total_segment.to(device),mask=total_mask.to(device))
            labelx = start,end
            total_loss = model.mylossFunction(result, labelx)
            total_loss.requires_grad_(True).to(device)
            print(i)
            print(total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()



        torch.save(model, 'demo2')




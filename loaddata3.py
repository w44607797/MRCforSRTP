import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from datasets import list_datasets, load_dataset

import MRCmodel

squad_train = load_dataset('E:\dataset\cmrc2018', split='train')
question_train = squad_train['question']
context_train = squad_train['context']
answer = squad_train['answers']
tokenizer = BertTokenizer.from_pretrained('E:\python\model/bert_pretrain')
model = MRCmodel.bertqa()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
device = 'cuda:0'
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
answers = []
contexts = []
questions = []
size = 0

for idx, context in enumerate(context_train):
    if len(context)<450:
        size+=1
        questions.append(question_train[idx])
        contexts.append(context)
        answers.append(answer[idx])

index = []
answer_end = []

for idx,i in enumerate(answers):
    index.append(int(i['answer_start'][0]))
    answer_end.append(int(i['answer_start'][0])+len(i['text'][0]))
print()
def encode(sentense):
   inputs = tokenizer(sentense, return_tensors='pt')
   token_type_ids = inputs['token_type_ids']
   attention_mask = inputs['attention_mask']
   inputs_ids = inputs['input_ids']
   return inputs_ids,token_type_ids,attention_mask


def package(sentense):
   total_token = []
   total_segment = []
   total_mask = []
   for i in range(0,len(sentense)-1):
      tokenId, segment, mask = encode(sentense[i])
      total_token.append(tokenId)
      total_segment.append(segment)
      total_mask.append(mask)
   return total_token,total_segment,total_mask

ques_total_token,ques_total_segment,ques_total_mask = package(questions)
context_total_token,context_total_segment,context_total_mask = package(contexts)

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, ques_tensor,context_tensor, target_tensor,size):
        self.ques_token_id,self.ques_segment,self.ques_mask = ques_tensor
        self.context_token_id,self.context_segment,self.context_mask = context_tensor
        self.start,self.end = target_tensor
        self.size = size
    # 返回数据集大小
    def __len__(self):
        return self.size
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.ques_token_id[index],self.ques_segment[index],self.ques_mask[index],\
               self.context_token_id[index],self.context_segment[index],self.context_mask[index],\
               self.start[index],self.end[index]

def my_collate(batch):
    data = [item[0] for item in batch]
    data2 = [item[1] for item in batch]
    data3 = [item[2] for item in batch]
    data4 = [item[3] for item in batch]
    data5 = [item[4] for item in batch]
    data6 = [item[5] for item in batch]
    data7 = [item[6] for item in batch]
    data8 = [item[7] for item in batch]
    return [data,data2,data3,data4,data5,data6,data7,data8]

ques_tensor = ques_total_token,ques_total_segment,ques_total_mask
context_tensor = context_total_token,context_total_segment,context_total_mask
target_tensor = index,answer_end
datasets = MyDataset(ques_tensor,context_tensor,target_tensor,size)
dataloader = DataLoader(datasets,batch_size=8,shuffle=False,sampler=None,batch_sampler=None,collate_fn=my_collate,
pin_memory=False,drop_last=False,timeout=0,worker_init_fn=None,multiprocessing_context=None)



if __name__ == '__main__':
    epoch = 30
    model.to(device)
    for i in range(epoch):
        for idx,(ques_token_id,ques_segment,ques_mask,context_token_id,context_segment,context_mask
                  , start,end) in enumerate(dataloader):
            ques_tensor = ques_token_id, ques_segment, ques_mask
            context_tensor = context_token_id,context_segment,context_mask
            result = model.forward(ques_tensor,context_tensor)
            labelx = start,end
            total_loss = model.loss(result, labelx)
            total_loss.requires_grad_(True).to(device)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

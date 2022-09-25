import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import MRCmodel
from torch.optim import AdamW
model = MRCmodel.bertqa()
# model = torch.load('best.pt')
optimizer = AdamW(model.parameters(), lr=3e-5)
model.train()

batch_size = 8

model_checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

device = 'cuda:0'
model.to(device)
squad_train = load_dataset('E:\dataset\cmrc2018', split='train')
squad_valid = load_dataset('E:\dataset\cmrc2018', split='validation')
question_train = squad_train[0:60]['question']
context_train = squad_train[0:60]['context']
answers_train = squad_train[0:60]['answers']
question_valid = squad_valid[0:60]['question']
context_valid = squad_valid[0:60]['context']
answers_valid = squad_valid[0:60]['answers']

del squad_valid,squad_train

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.token_id,self.segment,self.mask = data_tensor
        self.start,self.end = target_tensor
    # 返回数据集大小
    def __len__(self):
        return len(self.token_id)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.token_id[index],self.segment[index],self.mask[index],self.start[index],self.end[index]


def dealdata(question,context):

    inputs = tokenizer(question,context,max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length")

    start_positions = []
    end_positions = []

    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = inputs["overflow_to_sample_mapping"][i]
        answer = answers_train[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        ##判断是不是在这一个span中
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    return inputs,start_positions,end_positions

def skipthe0label(inputs,start_label,end_label):
    start = []
    end = []
    input_ids = []
    input_token_type_id = []
    input_attention_mask = []
    for i in range(len(start_label)):
        if start_label[i] != 0 and end_label != 0:
            start.append(start_label[i])
            end.append(end_label[i])
            input_ids.append(inputs['input_ids'][i])
            input_token_type_id.append(inputs['token_type_ids'][i])
            input_attention_mask.append(inputs['attention_mask'][i])
    return input_ids,input_token_type_id,input_attention_mask,start,end

def preprocess_data(question, context):
    inputs,start_positions,end_positions = dealdata(question, context)
    input_ids,token_type_ids,mask,start_positions,end_positions = skipthe0label(inputs,start_positions,end_positions)
    # input_ids = inputs['input_ids']
    input_ids = torch.tensor(input_ids)
    # token_type_ids = inputs['token_type_ids']
    token_type_ids = torch.tensor(token_type_ids)
    # mask = inputs['attention_mask']
    mask = torch.tensor(mask)
    tensor = input_ids,token_type_ids,mask
    start_positions = torch.tensor(start_positions)
    end_positions = torch.tensor(end_positions)
    target = start_positions,end_positions
    train_datasets = MyDataset(tensor,target)
    return train_datasets


train_datasets = preprocess_data(question_train, context_train)
valid_datasets = preprocess_data(question_valid,context_valid)

train_dataloader = DataLoader(train_datasets,batch_size=batch_size,shuffle=True,sampler=None,batch_sampler=None,
pin_memory=False,drop_last=False,timeout=0,worker_init_fn=None,multiprocessing_context=None)
valid_dataloader = DataLoader(valid_datasets,batch_size=batch_size,shuffle=True,sampler=None,batch_sampler=None,
pin_memory=False,drop_last=False,timeout=0,worker_init_fn=None,multiprocessing_context=None)
epoch = 45
if __name__ == '__main__':
    for i in range(epoch):
        for idx, (token_id, segment, mask, start, end
                  ) in enumerate(train_dataloader):
            model.train()
            target = start.to(device), end.to(device)
            predict = model.forward(token_id.to(device), segment.to(device), mask.to(device))
            total_loss = model.loss(predict, target)
            total_loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('train预测的loss为{}'.format(total_loss))
            del token_id, segment, mask
            del start, end

        with torch.no_grad():
            model.eval()
            for idx, (token_id, segment, mask, start, end
                      ) in enumerate(valid_dataloader):
                predict = model.forward(token_id.to(device), segment.to(device), mask.to(device))
                target = start.to(device), end.to(device)
                val_loss = model.loss(predict, target)
                del token_id, segment, mask
                del start, end
                print('val预测的loss为{}'.format(val_loss))

            torch.save(model,'attention{}.pt'.format(i))
            print(i)

            del val_loss

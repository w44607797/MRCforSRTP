import math

import torch
from keras import Model
from torch import nn, dropout
from transformers import BertConfig,BertForQuestionAnswering
from pytorch_pretrained_bert import BertForQuestionAnswering
from transformers import BertTokenizer,BertModel
import  torch.nn.functional as F
device = torch.device('cuda:0')
path = 'E:\python\model/bert_pretrain'
model_config = BertConfig.from_pretrained(path)  # 加载BERT配置文件
max_len = 512
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class bertqa(nn.Module):
    def __init__(self,):
        super(bertqa, self).__init__()
        self.attention_head_size = 768
        self.W_q = nn.Linear(768, 768, bias=False)
        self.W_k = nn.Linear(768, 768, bias=False)
        self.W_v = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout()

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('E:\python\model/bert_pretrain')
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        self.fc = nn.Linear(1,2)


    def forward(self,ques_tensor,context_tensor):
        start_logit = []
        end_logit = []
        ques_token_id, ques_segment, ques_mask = ques_tensor
        context_token_id,context_segment,context_mask = context_tensor
        for i in range(len(ques_token_id)):
            ques_embeding = self.bert(ques_token_id[i].to(device),ques_segment[i].to(device),ques_mask[i].to(device))
            contxt_embeding = self.bert(context_token_id[i].to(device), context_segment[i].to(device), context_mask[i].to(device))
            contxt_embeding = torch.tensor(contxt_embeding['last_hidden_state']).to(device)
            ques_embeding = torch.tensor(ques_embeding['last_hidden_state']).to(device)
            query = self.W_q(ques_embeding).to(device)
            keys = self.W_k(contxt_embeding).to(device)
            value = self.W_v(contxt_embeding).to(device)

            attention_scores = torch.matmul(query, keys.transpose(1, 2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            out = torch.matmul(attention_probs, value)  # (1,length,768)

            logits = self.fc(out)  # (1,length,2)

            start_logits, end_logits = logits.split(1, dim=-1)  # ((B, T, 1),(B, T, 1))
            start_logits = start_logits.squeeze(-1)  # (B, T)
            end_logits = end_logits.squeeze(-1)  # (B, T)
            start_logit.append(start_logits)
            end_logit.append(end_logits)

        return start_logit,end_logit

    def loss(self,y,label):
        start_logits, end_logits = y
        start_position, end_position = label
        # start_position = start_position.unsqueeze(start_position, axis=-1)
        # end_position = end_position.unsqueeze(end_position, axis=-1)
        start_position = torch.tensor(start_position).to(device)
        end_position = torch.tensor(end_position).to(device)
        # start_logits = torch.tensor(start_logits).to(device)
        # end_logits = torch.tensor(end_logits).to(device)
        total_loss = 0
        length = len(start_logits)

        for i in range(length):
            start = start_logits[i]
            end = end_logits[i]
            start = start.squeeze(0)
            end = end.squeeze(0)
            start_loss = nn.CrossEntropyLoss()(
                start, start_position[i]).to(device)
            end_loss = nn.CrossEntropyLoss()(
                end, end_position[i]).to(device)
            total_loss += (start_loss+end_loss)/2

        total_loss = total_loss/length

        return total_loss


    def mylossFunction(self,y,label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_loss = 0
        end_loss = 0
        for index, start in enumerate(start_position):
            pre = torch.argmax(start_logits[index]).to(device)
            start_logits_exp = torch.exp(start_logits[index]).to(device)
            start_logits_exp = torch.tensor(start_logits_exp).to(device)
            start_logits_exp = torch.sum(start_logits_exp).to(device)
            start_loss += -start_logits[index][start] + math.log(math.exp(abs(pre - start))) + torch.log(
                start_logits_exp).to(device)

        for index, end in enumerate(end_position):
            pre = torch.argmax(end_logits[index]).to(device)
            end_logits_exp = torch.exp(end_logits[index]).to(device)
            end_logits_exp = torch.tensor(end_logits_exp).to(device)
            end_logits_exp = torch.sum(end_logits_exp).to(device)
            end_loss += -end_logits[index][end] + math.log(math.exp(abs(pre - end))) + torch.log(
                end_logits_exp).to(device)

        return (start_loss+end_loss)/(2*len(start_position))




if __name__ == '__main__':
    model = bertqa()





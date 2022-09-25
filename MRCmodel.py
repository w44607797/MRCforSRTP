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
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class bertqa(nn.Module):
    def __init__(self,):
        super(bertqa, self).__init__()
        self.attention_head_size = 768
        self.W_q = nn.Linear(768, 768, bias=False)
        self.W_k = nn.Linear(768, 768, bias=False)
        self.W_v = nn.Linear(768, 768, bias=False)
        self.dropout = nn.Dropout(0.3)

        # self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained('E:\python\model/bert_pretrain')
        self.tokenizer = tokenizer
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        # self.rnn = nn.GRU(
        #     input_size=1,
        #     hidden_size=88,           #(batch_size,src_length,dim)
        #     num_layers=3,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        self.fc = nn.Linear(768,2)
        # self.combine = nn.Linear()

    def forward(self,token_id, segment, mask):
        start_logit = []
        end_logit = []

        embeding = self.bert(token_id,segment,mask)
        embeding = torch.tensor(embeding['last_hidden_state']).to(device)  #（8，384，768）
        #     (device)
        query = self.W_q(embeding).to(device)   #（8，384，768）
        keys = self.W_k(embeding).to(device)  #（8，384，768）
        value = self.W_v(embeding).to(device)    #（8，384，1）

        attention_scores = torch.matmul(query, keys.transpose(1, 2))    #（8，384，384）
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)     #（8，384，384）
        attention_probs = nn.Softmax(dim=-1)(attention_scores)      #（8，384，384）
        out = torch.matmul(attention_probs, value)  # (1,length,1)     #（8，384，1）
        # out = self.rnn(out)

        logits = self.fc(out).to(device)  # (1,length,2)

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

        start = start_logits[0].to(device)
        end = end_logits[0].to(device)
        # start = start.squeeze(0)
        # end = end.squeeze(0)
        start_loss = nn.CrossEntropyLoss()(
            start, start_position).to(device)
        end_loss = nn.CrossEntropyLoss()(
            end, end_position).to(device)
        total_loss = (start_loss+end_loss)/2

        return total_loss


    # def mylossFunction(self,y,label):
    #     start_logits, end_logits = y
    #     start_position, end_position = label
    #     start_loss = 0
    #     end_loss = 0
    #     pre = torch.argmax(start_logits).to(device)
    #     start_logits_exp = torch.exp(start_logits).to(device)
    #     start_logits_exp = torch.tensor(start_logits_exp).to(device)
    #     start_logits_exp = torch.sum(start_logits_exp).to(device)
    #     start_loss += -start_logits[start] + math.log(math.exp(abs(pre - start))) + torch.log(
    #         start_logits_exp).to(device)
    #
    #     pre = torch.argmax(end_logits).to(device)
    #     end_logits_exp = torch.exp(end_logits).to(device)
    #     end_logits_exp = torch.tensor(end_logits_exp).to(device)
    #     end_logits_exp = torch.sum(end_logits_exp).to(device)
    #     end_loss += -end_logits[end] + math.log(math.exp(abs(pre - end))) + torch.log(
    #         end_logits_exp).to(device)
    #
    #     return (start_loss+end_loss)/(2*len(start_position))


if __name__ == '__main__':
    model = bertqa()





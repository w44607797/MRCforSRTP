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

class bertqa(nn.Module):
    def __init__(self,):
        super(bertqa, self).__init__()
        self.W_q = nn.Linear(768, 768, bias=False)
        self.W_k = nn.Linear(768, 768, bias=False)
        self.W_v = nn.Linear(768, 1, bias=False)
        self.dropout = nn.Dropout()

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=path)
        self.tokenizer = BertTokenizer.from_pretrained('E:\python\model/bert_pretrain')
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        # self.rnn = nn.GRU(
        #     input_size=768,
        #     hidden_size=768,           #(batch_size,src_length,dim)
        #     num_layers=3,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.queries,self.keys =
        # self.attn =
        self.fc = nn.Linear(768,2)


    def forward(self,tokenId,segment,mask):
        out = self.bert(tokenId,segment,mask)   # (batch_size, 512 , 768 )
        out = torch.tensor(out['last_hidden_state'])
        # out = self.rnn(out)
        # out = out[0]
        query = self.W_q(out)  #(batch_size, 512 , 768)
        keys = self.W_k(out)  #(batch_size,512, 768)
        value = self.W_v(out) #(batch_size,512, 1)
        logits = self.fc(out)

        start_logits, end_logits = logits.split(1, dim=-1)  # ((B, T, 1),(B, T, 1))
        start_logits = start_logits.squeeze(-1)  # (B, T)
        end_logits = end_logits.squeeze(-1)  # (B, T)


        return start_logits,end_logits

    def loss(self,y,label):
        start_logits, end_logits = y
        start_position, end_position = label
        # start_position = start_position.unsqueeze(start_position, axis=-1)
        # end_position = end_position.unsqueeze(end_position, axis=-1)
        start_position = torch.tensor(start_position).to(device)
        end_position = torch.tensor(end_position).to(device)
        start_logits = torch.tensor(start_logits).to(device)
        end_logits = torch.tensor(end_logits).to(device)
        # start_logits = start_logits.unsqueeze(0)
        # end_logits = end_logits.unsqueeze(0)
        start_loss = nn.CrossEntropyLoss()(
            start_logits,start_position).to(device)

        end_loss = nn.CrossEntropyLoss()(
            end_logits,end_position).to(device)
        total_loss = (start_loss+end_loss)/2
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





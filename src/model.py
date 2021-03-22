import config
import utils
from torch import nn
import torch


class BertForFeatureExtraction(nn.Module):
    def __init__(self):
        super(BertForFeatureExtraction, self).__init__()
        self.bert = utils.load_bert_model(config.BERT_PATH)
        
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        return last_hidden_state[:, 0, :].cpu().numpy()

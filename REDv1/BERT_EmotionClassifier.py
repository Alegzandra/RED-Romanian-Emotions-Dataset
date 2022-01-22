from torch import nn, optim
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

class EmotionsClassifier(nn.Module):
  def __init__(self, n_classes, model_name):
    super(EmotionsClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(model_name)
    #definesc dropout-ul
    self.drop = nn.Dropout(p=0.3)
    #definesc output-ul
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    #definesc functia pierdere
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)
    output = self.out(output)
    #return self.softmax(output)
    return output

from bert.configuration_bert import BertConfig
from bert.modeling_bert import BertModel
import torch
from torch import nn

class BertClassifier(nn.Module):
    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        output = self.classifier(pooled_output)
        return output

if __name__ == "__main__":
    conf = BertConfig(num_labels=10)
    bert = BertClassifier(conf)
    bert.eval()
    bert.train_mask = True
    a = torch.zeros(20, 512).long()
    output = bert(a)

    # print(output, output.size())
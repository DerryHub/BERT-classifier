import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from dataset import MyDataset
from torch.utils.data import DataLoader
from bert.configuration_bert import BertConfig
from bertClassifier import BertClassifier
from tokenizer import Tokenizer
# from transformers import BertConfig, BertForSequenceClassification

BATCH_SIZE = 8
EPOCHS = 100
MAX_LEN = 256
NUM_WORKERS = 4

tokenizer = Tokenizer(vocab='data/cnews/vocab.txt', maxLen=MAX_LEN)

dataset = MyDataset(tokenizer, file='data/cnews/cnews.train.txt')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

dataset_val = MyDataset(tokenizer, file='data/cnews/cnews.val.txt')
loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE*4, num_workers=NUM_WORKERS)

# config = BertConfig.from_pretrained('model/bert-base-chinese-config.json')

# config.num_labels = dataset.countCls
config = BertConfig(
    vocab_size=tokenizer.count,
    num_labels=dataset.countCls,
    max_position_embeddings=MAX_LEN
)

bertClassifier = BertClassifier(config)
# bertClassifier.bert.from_pretrained('model/bert-base-chinese-pytorch_model.bin', config=config)
# bertClassifier = BertForSequenceClassification(config)

# bertClassifier.bert.embeddings.word_embeddings = nn.Embedding(tokenizer.count, config.hidden_size, padding_idx=0)

bertClassifier = bertClassifier.cuda()

cost = nn.CrossEntropyLoss()

param_optimizer = list(bertClassifier.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

opt = optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

for epoch in range(EPOCHS):
    print()
    loss_all = 0
    acc = 0
    total = 0
    # mask_num = 0
    loader_t = tqdm(loader)
    loader_t.set_description_str(' Train [{}/{}]'.format(epoch+1, EPOCHS))
    bertClassifier.train()
    for data_d in loader_t:
        labels = data_d['label'].cuda()
        input_ids = data_d['cont'].cuda()
        mask = data_d['mask'].cuda()
        output = bertClassifier(
            input_ids=input_ids, 
            attention_mask=mask)
        # mask_num += mask.sum()
        loss = cost(output, labels.view(-1))
        total += labels.size(0)
        acc += (torch.argmax(output, dim=1)==labels.view(-1)).sum().float()
        # print(torch.argmax(output, dim=1), labels.view(-1))
        loss_all += loss.data
        opt.zero_grad()
        loss.backward()
        opt.step()
    # print(mask_num/total)

    print('Loss of epoch [{}/{}] is {}'.format(epoch+1, EPOCHS, loss_all))
    print('Train accuracy of epoch [{}/{}] is {}'.format(epoch+1, EPOCHS, acc/total))

    acc = 0
    total = 0
    loader_t = tqdm(loader_val)
    loader_t.set_description_str(' Evaluate [{}/{}]'.format(epoch+1, EPOCHS))
    bertClassifier.eval()
    for data_d in loader_t:
        labels = data_d['label'].cuda()
        input_ids = data_d['cont'].cuda()
        mask = data_d['mask'].cuda()
        with torch.no_grad():
            output = bertClassifier(input_ids=input_ids, attention_mask=mask)
        total += labels.size(0)
        acc += (torch.argmax(output, dim=1)==labels.view(-1)).sum().float()
    
    print('Evaluate accuracy of epoch [{}/{}] is {}'.format(epoch+1, EPOCHS, acc/total))

    scheduler.step()

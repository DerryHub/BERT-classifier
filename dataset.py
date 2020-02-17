import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer
from tqdm import tqdm
import random
import pandas as pd

class MyDataset(Dataset):
    def __init__(
        self, 
        tokenizer,
        file='data/cnews/cnews.train.txt', 
        labels='data/cnews/labels.txt', 
        maxLen=512
    ):
        super(MyDataset, self).__init__()

        with open(file, 'r') as f:
            lines = f.readlines()
        with open(labels, 'r') as f:
            labels = f.readlines()
        
        random.shuffle(lines)
        # lines = lines[:500]

        self.dicC2I = {}
        self.dicI2C = {}
        index = 0
        for c in labels:
            c = c.strip()
            self.dicC2I[c] = index
            self.dicI2C[index] = c
            index += 1
        
        self.countCls = len(self.dicC2I)

        self.labels = []
        self.conts = []
        
        lines_t = tqdm(lines)
        lines_t.set_description_str(' Dealing Data')
        self.mask_labels = []
        for line in lines_t:
            line = line.split('\t')
            c = line[0]
            cont = line[1]
            self.labels.append(self.dicC2I[c])
            cont = tokenizer(cont)
            self.conts.append(cont)
        # print(cont, len(cont))
        self.attention_masks = []
        for seq in self.conts:
            seq_mask = [float(i>0) for i in seq]
            self.attention_masks.append(seq_mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        c = self.labels[index]
        cont = self.conts[index]
        mask = self.attention_masks[index]
        c = torch.LongTensor([c])
        cont = torch.LongTensor(cont)
        mask = torch.Tensor(mask)
        return {'label':c, 'cont':cont, 'mask':mask}

# class MyDataset(Dataset):
#     def __init__(
#         self, 
#         tokenizer,
#         file='data/cola_public_1.0/raw/in_domain_train.tsv', 
#         maxLen=512):
#         super(MyDataset, self).__init__()

#         df = pd.read_csv(
#             file, 
#             delimiter='\t', 
#             header=None,
#             names=['sentence_source', 'label', 'label_notes', 'sentence'])
        
#         sentencses=['[CLS] ' + sent + ' [SEP]' for sent in df.sentence.values]
#         self.labels=df.label.values[:1000]
#         self.tokenized_sents=[tokenizer(sent) for sent in sentencses][:1000]

#         self.attention_masks = []
#         for seq in self.tokenized_sents:
#             seq_mask = [float(i>0) for i in seq]
#             self.attention_masks.append(seq_mask)


#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         c = self.labels[index]
#         cont = self.tokenized_sents[index]
#         mask = self.attention_masks[index]
#         c = torch.LongTensor([c])
#         cont = torch.LongTensor(cont)
#         mask = torch.Tensor(mask)

#         return {'class':c, 'cont':cont, 'mask':mask}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    tokenizer = Tokenizer()
    dataset = MyDataset(tokenizer, mask=True)
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    # print(dataset[10])
    for d in loader:
        print(d)
        break
    
    

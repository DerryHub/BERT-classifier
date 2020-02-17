import jieba
from tqdm import tqdm
import random
from bert.tokenization_bert import BertTokenizer

class Tokenizer:
    def __init__(
        self, 
        vocab='data/cnews/vocab.txt', 
        stop_use='data/cnews/stop_use.txt', 
        maxLen=512
    ):
        self.maxLen = maxLen

        self.dicW2I = {}
        self.dicI2W = {}

        self.dicW2I['[PAD]'] = 0
        self.dicW2I['[CLS]'] = 1
        self.dicW2I['[SEP]'] = 2
        self.dicW2I['[MASK]'] = 3
        self.dicW2I['[UNK]'] = 4

        for k in self.dicW2I.keys():
            self.dicI2W[self.dicW2I[k]] = k

        with open(stop_use, 'r') as f:
            self.stop_words = f.readlines()
            self.stop_words = [word.strip() for word in self.stop_words]

        # print(stop_words)
        
        index = 5
        with open(vocab, 'r') as f:
            lines = f.readlines()

        lines_t = tqdm(lines)
        lines_t.set_description_str(' Dealing Words')
        for line in lines_t:
            line = line.strip()
            self.dicW2I[line] = index
            self.dicI2W[index] = line
            index += 1
        
        self.count = len(self.dicW2I)
    
    def __call__(self, text):
        cut_raw = jieba.cut(text)
        cut = []
        for word in cut_raw:
            word = word.strip()
            if word not in self.stop_words and word:
                cut.append(word)
        output = []
        output.append(self.dicW2I['[CLS]'])
        for word in cut:
            try:
                output.append(self.dicW2I[word])
            except:
                output.append(self.dicW2I['[UNK]'])
        while len(output) < self.maxLen:
            output.append(self.dicW2I['[PAD]'])
        if len(output) > self.maxLen:
            output = output[:self.maxLen]
        return output

# class Tokenizer:
#     def __init__(self, vocab='model/bert-base-chinese-vocab.txt', maxLen=512):
#         self.maxLen = maxLen

#         self.tokenizer = BertTokenizer.from_pretrained(vocab)

    
#     def __call__(self, text):
#         text = '[CLS] '+text
#         cut = self.tokenizer.tokenize(text)
#         output = self.tokenizer.convert_tokens_to_ids(cut)
#         while len(output) < self.maxLen:
#             output.append(0)
#         if len(output) > self.maxLen:
#             output = output[:512]
#         return output

if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.set_mask(True)
    print(tokenizer('这意味着我们已经看到（我们第一次意识到）在论文《Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates》使用Adam所获得的超级收敛效果（Super-Convergence）！超级收敛是在采用较大的学习率训练神经网络时发生的一种现象，使训练速度加快一倍。在了解这一现象之前，将CIFAR10训练达到94％的准确度大约需要100个epochs。'))
    # with open('data/cnews/cnews.test.txt', 'r') as f:
    #     line = f.readline()
    # line = line.split('\t')[1]
    # print(len(tokenizer(line)))
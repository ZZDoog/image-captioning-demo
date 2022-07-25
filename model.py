
import torch
from torch import nn
from torch.nn import functional as f
from torchvision import models
from data import Dictionary

def cuda_check():
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
        device = "cuda:0"
    else:
        print('No GPU available, training on CPU.')
        device = "cpu"
    return device

def cuda_device():
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        device = "cuda:0"
    else:
        device = "cpu"
    return device

class Encoder(nn.Module):

    def __init__(self, emb_size):
        # the image encode part of the captioning net
        # output is a 256 dimension vector
        super(Encoder, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        self.fc = nn.Linear(1000, emb_size)

    def forward(self, image):

        x = self.vgg(image)
        x = f.relu(self.fc(x))

        return x


class Decoder(nn.Module):

    def __init__(self, dictionary, emb_size):
        # decode an image embedding to a nature language captioning
        super(Decoder, self).__init__()
        self.dictionary = dictionary
        self.vocab_size = len(dictionary)
        self.emb_size = emb_size

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size) # Embedding的输入用索引就行了
        self.output = nn.Linear(self.emb_size, self.vocab_size)

        self.lstm = nn.LSTMCell(input_size=emb_size, hidden_size=emb_size)

    def forward(self, image_emb, captions, max_len):
        # 生成caption的前向计算
        h = image_emb
        c = image_emb
        output = []
        for word_idx in captions:

            if self.dictionary.idx2word[word_idx] != '<eos>':
                # 在caption还未结束的时候 继续计算下一个词
                input_emb = self.embedding(word_idx)
                h, c = self.lstm(input_emb, (h, c))
                output.append(f.relu(self.output(h)))

            else:
                while len(output) < max_len:
                    padding = torch.tensor(torch.nn.functional.one_hot(torch.tensor(self.dictionary.word2idx['<pad>']),
                                                              num_classes=self.vocab_size), dtype=torch.float32)
                    padding = padding.to(cuda_device())
                    output.append(padding)
                break

        return output

























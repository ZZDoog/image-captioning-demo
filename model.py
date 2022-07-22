
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

class Encoder(nn.Module):

    def __init__(self):
        # the image encode part of the captioning net
        # output is a 256 dimension vector
        super(Encoder, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        self.fc = nn.Linear(1000, 256)

    def forward(self, image):

        x = self.vgg(image)
        x = nn.ReLU(self.fc(x))

        return x


class Decoder(nn.Module):

    def __init__(self, dictionary, emb_size, ):
        # decode a image embedding to a nature language captioning
        super(Decoder, self).__init__()
        self.dictionary = dictionary
        self.vocab_size = len(dictionary)
        self.emb_size = emb_size

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.output = nn.Linear(self.embedding, self.vocab_size)

        self.lstm = nn.LSTMCell(input_size=emb_size, hidden_size=emb_size)

    #def forward(self):














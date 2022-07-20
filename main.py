# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2 as cv
import math
import os
import torch
import linecache  # 用来读取txt文件中特定行的内容
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms


PATH_caption_test = "D:\Flickr8k\captions_test.txt"
PATH_caption_train = "D:\Flickr8k\captions_train.txt"
PATH_image_folder = "D:\Flickr8k\Images"


def cuda_check():
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
        device = "cuda:0"
    else:
        print('No GPU available, training on CPU.')
        device = "cpu"


class get_dataset(Dataset):
    def __init__(self, caption_path, image_folder_path, dictionary):
        # self
        self.caption_path = caption_path
        self.image_path = image_folder_path
        self.dictionary = dictionary

    def __getitem__(self, idx):
        # get item
        content = linecache.getline(self.caption_path, idx)
        content = content.split(',', -1)

        image_name = content[0]
        caption = content[1]
        words = caption.split(' ', -1) + ['<eos>']
        word_idx = []

        for word in words:
            word_idx.append(dictionary.word2idx.get(word))

        image_path = os.path.join(self.image_path, image_name)
        image = cv.imread(image_path, 1)



        return image, word_idx

    def __len__(self):
        len(open(self.caption_path, 'r').readlines())


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    # Add words to the dictionary
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    # Add a file content to the dictionary
    def add_file(self, file_path):
        with open(file_path, 'r', encoding="utf8") as f:
            for line in f:
                content = line.split(',', -1)
                caption = content[1]
                words = caption.split(' ', -1) + ['<eos>']
                for word in words:
                    self.add_word(word)

    def __len__(self):
        return len(self.idx2word)



if __name__ == '__main__':

    cuda_check()

    dictionary = Dictionary()
    dictionary.add_file(PATH_caption_test)
    dictionary.add_file(PATH_caption_train)

    print("loading train data.......")
    train_data = get_dataset(PATH_caption_train, PATH_image_folder, dictionary)
    print("train data load success!")
    print("loading test data.......")
    test_data = get_dataset(PATH_caption_train, PATH_image_folder, dictionary)
    print("test data load success!")







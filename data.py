import cv2 as cv
import os
import torch
import linecache  # 用来读取txt文件中特定行的内容
from torch.utils.data import Dataset
from torchvision import transforms



class get_dataset(Dataset):
    def __init__(self, caption_path, image_folder_path, dictionary, caption_max_len):
        # self
        self.caption_path = caption_path
        self.image_path = image_folder_path
        self.dictionary = dictionary
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),  # 缩放
                            transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),  # 这组参数是使用
                            ])
        self.caption_max_len = caption_max_len

    def __getitem__(self, idx):
        # get item
        content = linecache.getline(self.caption_path, idx+1)
        content = content.split(',', -1)

        image_name = content[0]
        caption = content[1]
        words = caption.split(' ', -1) + ['<eos>']
        words.insert(0, '<cls>')
        while len(words) < self.caption_max_len:
            words.append('<pad>')
        word_idx = []

        for word in words:
            word_idx.append(self.dictionary.word2idx.get(word))
        word_idx = torch.tensor(word_idx)

        image_path = os.path.join(self.image_path, image_name)
        image = cv.imread(image_path, 1)

        image_pil = transforms.ToPILImage()(image)
        image = self.transform(image_pil)

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
                words.insert(0, '<cls>')
                for word in words:
                    self.add_word(word)

    def __len__(self):
        return len(self.idx2word)

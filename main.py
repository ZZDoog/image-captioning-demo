# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import torch
from torch import nn
import torch.optim as optim


import model
from data import get_dataset, Dictionary



# the path of both caption txt file and image folder
PATH_caption_test = "D:\Flickr8k\captions_test.txt"
PATH_caption_train = "D:\Flickr8k\captions_train.txt"
PATH_image_folder = "D:\Flickr8k\Images"



if __name__ == '__main__':

    # check if the cuda is available
    model.cuda_check()

    # load the dictionary from train caption and test caption
    dictionary = Dictionary()
    dictionary.add_file(PATH_caption_test)
    dictionary.add_file(PATH_caption_train)
    dictionary.add_word('<pad>')


    # create the dataset
    print("loading train data.......")
    train_data = get_dataset(PATH_caption_train, PATH_image_folder, dictionary, caption_max_len=30)
    print("train data load success!")
    print("loading test data.......")
    test_data = get_dataset(PATH_caption_train, PATH_image_folder, dictionary, caption_max_len=30)
    print("test data load success!")

    image, caption = train_data[0]
    nature_caption = []
    for idx in caption:
        nature_caption.append(dictionary.idx2word[idx])
    print(len(dictionary))










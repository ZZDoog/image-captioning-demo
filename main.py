# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import datetime
from torch import nn
import torch.optim as optim

import model
from model import Encoder, Decoder
from data import get_dataset, Dictionary
from torch.utils.data import DataLoader

# the path of both caption txt file and image folder
PATH_caption_test = "D:\Flickr8k\captions_test.txt"
PATH_caption_train = "D:\Flickr8k\captions_train.txt"
PATH_image_folder = "D:\Flickr8k\Images"

CAPTION_MAX_LEN = 30
BATCH_SIZE = 1
EPOCH = 200


if __name__ == '__main__':

    # check if the cuda is available
    device = model.cuda_check()

    # load the dictionary from train caption and test caption
    dictionary = Dictionary()
    dictionary.add_file(PATH_caption_test)
    dictionary.add_file(PATH_caption_train)
    dictionary.add_word('<pad>')

    # create the dataset
    print("loading train data.......")
    train_data = get_dataset(PATH_caption_train, PATH_image_folder, dictionary, caption_max_len=CAPTION_MAX_LEN)
    print("train data load success!")
    print("loading test data.......")
    test_data = get_dataset(PATH_caption_train, PATH_image_folder, dictionary, caption_max_len=CAPTION_MAX_LEN)
    print("test data load success!")

    # create the dataloader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    # 构建Encoder和Decoder网络
    encoder = Encoder(emb_size=256)
    decoder = Decoder(dictionary=dictionary, emb_size=256)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 冻结Encoder中预训练的vgg的参数
    for name, param in encoder.named_parameters():
        if "vgg" in name:
            param.requires_grad = False

    # Loss Function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_encoder = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)
    optimizer_decoder = optim.SGD(decoder.parameters(), lr=0.001, momentum=0.9)

    # the training steps
    for epoch in range(0, EPOCH):
        # Train Model
        encoder.train()
        decoder.train()
        running_loss = 0.0
        for image, caption_idx in train_data:

            # calculate the running time
            start = datetime.datetime.now()

            # Move to GPUs
            # image, caption_idx = data
            image = image.to(device)
            caption_idx = caption_idx.to(device)

            # Forward prop
            image = image.unsqueeze(0)
            image_embedding = encoder(image)
            image_embedding = image_embedding.squeeze()
            caption_output = decoder(image_embedding, caption_idx, max_len=CAPTION_MAX_LEN)

            # Calculate loss
            loss = 0.0
            for i, output in enumerate(caption_output):
                loss += criterion(output, caption_idx[i])
            running_loss += loss

            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()

            end = datetime.datetime.now()
            print(end-start)

        print('[%d , %5d] loss: %.5f' % (epoch + 1, EPOCH, running_loss / len(train_data)))






















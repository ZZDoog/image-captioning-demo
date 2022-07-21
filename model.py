import torch


def cuda_check():
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
        device = "cuda:0"
    else:
        print('No GPU available, training on CPU.')
        device = "cpu"
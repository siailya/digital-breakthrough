#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
address matching
'''
import os
import torch
import numpy as np
import torch.optim as optim
from torch import nn
from torchtext import data
from Utils.data_utils import load_address_match_data
from Utils.utils import get_device
from train_eval_match import train, evaluate
from Model import args, AMM
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print('*****using:{}*****'.format(os.environ['CUDA_VISIBLE_DEVICES']))


def main(config):
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print("*****the model name is {}*****".format(config.model_name))
    device, n_gpu = get_device()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    '''data processor'''
    tokenize = lambda s: [char for char in s]

    def tokenize2(x):
        return [int(char) for char in x.split(',')]

    text_field = data.Field(tokenize=tokenize, lower=True, include_lengths=True, fix_length=config.sequence_length)
    feature_field = data.Field(tokenize=tokenize2, include_lengths=True, use_vocab=False, pad_token=62,
                               fix_length=config.sequence_length)  # sequential=True
    feature_field2 = data.Field(tokenize=tokenize2, include_lengths=True, use_vocab=False, pad_token=4,
                                fix_length=config.sequence_length)  # sequential=True
    label_field = data.LabelField(dtype=torch.long)

    # load data
    train_iterator, dev_iterator, test_iterator, train_text, word_to_id = \
        load_address_match_data(
        config.data_path, text_field,
        feature_field, feature_field2,
        label_field, config.batch_size,
        device, config.word_embedding_file,
        config.cache_path)

    # embedding
    pretrained_embeddings = text_field.vocab.vectors

    '''model'''
    model_file = config.model_dir + 'model_AMM.pt'
    print('*****save model name: {}*****'.format(model_file))

    if config.model_name == 'AMM':
        model = AMM(config.word_embedding_dim,
                                            config.output_dim,
                                            config.hidden_size,
                                            config.num_layers,
                                            config.bidirectional,
                                            config.dropout,
                                            pretrained_embeddings)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    print('**************model config**************')
    print('lr={}'.format(config.lr))
    print('do_train={}'.format(config.do_train))

    if config.do_train:
        print("**************start training******************")
        train(config.epoch_num, model, train_iterator, dev_iterator, optimizer, criterion,
              ['0', '1'], model_file, config.log_dir, config.print_step, device)
    else:
        print("************don't train, load saved model*************")

    print("**********testing model:{}**************".format(model_file))
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    test_loss, test_acc, test_report = evaluate(
        model, test_iterator, criterion, ['0', '1'], device,
        is_test=True)
    print("-------------- Test -------------")
    print("test_loss:{}".format(test_loss))
    print("\t Loss: {} | Acc: {} | Macro avg F1: {} | Weighted avg F1: {}".format(
        test_loss, test_acc, test_report['macro avg']['f1-score'], test_report['weighted avg']['f1-score']))
    print('test_report:')
    for item in test_report.items():
        print(item)


if __name__ == '__main__':
    model_name = "AMM" 
    data_dir = "./data/shenzhen_address_match_data/"
    cache_dir = "./cache/"
    embedding_folder = "data/shenzhen_address_match_data/"
    model_dir = "./save_models/"
    log_dir = "log/"
    main(args.get_args(data_dir, cache_dir, embedding_folder, model_dir, log_dir))

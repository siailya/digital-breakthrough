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
    
    # load data
    train_iterator, dev_iterator, test_iterator = \
        load_address_match_data(config.data_path, device, config.batch_size)


    '''model'''
    model_file = config.model_dir + 'model_AMM.pt'
    print('*****save model name: {}*****'.format(model_file))


    model = AMM.AMM(config.word_embedding_dim,
                config.output_dim,
                config.hidden_size,
                config.num_layers,
                config.bidirectional,
                config.dropout,
                ) 
    model._save_to_state_dict()
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
    data_dir = "/Users/kirill/Desktop/CP/digital-breakthrough/Utils/data/"
    cache_dir = "/Users/kirill/Desktop/CP/digital-breakthrough/cashe/"
    model_dir = "/Users/kirill/Desktop/CP/digital-breakthrough/save_models/TextRCNN/"
    log_dir = "log/"
    main(args.get_args(data_dir, cache_dir, model_dir, log_dir))

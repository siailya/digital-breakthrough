import torch
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import FastText
from tqdm import tqdm
from razdel import tokenize

class CustomTextDataset(Dataset):
    def __init__(self, x, y, labels):
        self.labels = labels
        self.x= x
        self.y = y
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        print(idx)
        label = self.labels[idx]
        address1 = self.x[idx]
        address2 = self.y[idx]
        sample = {"address1": address1, "address2": address2, "label": label}
        return sample



def address_emb(model,address1, address2, device):
    k = 10
    N = len(address1)
    vec1 = []
    vec2 = []
    for i in tqdm(range(N)):
        tokens = list(tokenize(address1[i]))
        tokens = [_.text for _ in tokens]
        predict = torch.tensor([model.wv.get_vector(token) for token in tokens])
        print(predict.shape)
        while predict.shape[0] <k:
            predict = torch.vstack([predict, torch.zeros((1,predict.shape[1]))])
            print(predict.shape)
        #predict = torch.mean(predict, axis=0).reshape(-1,1)
        #predict = predict / torch.linalg.norm(predict)
        vec1.append(predict)

    for i in tqdm(range(N)):
        tokens = list(tokenize(address2[i]))
        tokens = [_.text for _ in tokens]
        predict = torch.tensor([model.wv.get_vector(token) for token in tokens])
        while predict.shape[0] <k:
            predict = torch.vstack([predict, torch.zeros((1,predict.shape[1]))])

        #predict = torch.mean(predict, axis=0).reshape(-1,1)
        #predict = predict / torch.linalg.norm(predict)
        vec2.append(predict)
    #print(vec1[-1].shape, 'fsdhfbs')
    #vec1, vec2 = torch.vstack(vec1), torch.vstack(vec1)
    #print(vec1.shape)
    return vec1, vec2


def load_address_match_data(path, device, batch_size, save = False):

    print("********************data processing*************************")
    # all shenzhen data
    train_file_path = f'{path}train.csv'
    dev_file_path = f'{path}dev.csv'
    test_file_path = f'{path}test.csv'

    reader = lambda x: pd.read_csv(x)

    train = reader(train_file_path)
    dev = reader(dev_file_path)
    test = reader(test_file_path)

    model = FastText.load('/Users/kirill/Desktop/CP/digital-breakthrough/Utils/fasttext.model')

    emb = lambda x, y: address_emb(model,x, y, device)

    vec1_train, vec2_train =  emb(list(train['address1']), list(train['address2']))
    labels_train = torch.tensor(train['label'], device = device)
    vec1_dev, vec2_dev = emb(list(dev['address1']), list(dev['address2']))
    labels_dev = torch.tensor(dev['label'], device = device)
    vec1_test, vec2_test=  emb(list(test['address1']), list(test['address2']))
    labels_test =torch.tensor(test['label'], device = device)
    train_iter = CustomTextDataset(vec1_train, vec2_train, labels_train )
    dev_iter = CustomTextDataset(vec1_dev, vec2_dev, labels_dev )
    test_iter = CustomTextDataset(vec1_test, vec2_test, labels_test)

    train_dataloader = DataLoader(train_iter, batch_size = batch_size, shuffle= True)
    dev_dataloader = DataLoader(dev_iter, batch_size= batch_size, shuffle= True)
    test_dataloader = DataLoader(test_iter, batch_size= batch_size, shuffle= True)
    print("Successful Loading Dataset")
    if save == True:
        torch.save('train_iter' )
        torch.save('dev_iter')
        torch.save('test_iter')

    return train_dataloader, dev_dataloader, test_dataloader

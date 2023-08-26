import torch
from torchtext import data
from torchtext import vocab


def load_address_match_data(path, text_field, feature_field, feature_field2, label_field, batch_size, device, embedding_file, cache_dir):
    print("********************data processing*************************")
    # all shenzhen data
    train_file_path = ''
    dev_file_path = ''
    test_file_path = ''

    train, dev, test = data.TabularDataset.splits(
        path=path, train=train_file_path, validation=dev_file_path,test=test_file_path, format='tsv', skip_header=True,
        fields=[('address1', text_field), 
                ('address2', text_field),
                ('label', label_field)])

    print("the size of train: {}, dev:{}, test:{}".format(
        len(train.examples), len(dev.examples), len(test.examples))) 
    print('train data examples:')
    # print(train[0].__dict__.keys())
    for i in range(0, 1):
        print(train[i].address1, '\n')
        print(train[i].address2, '\n')
        print(train[i].label)

    print('embedding file path:', embedding_file)
    print('cache path:', cache_dir)
    print("train dataset path :{}\n"
          "dev dataset path :{}\n"
          "test dataset path:{}".format(train_file_path, dev_file_path, test_file_path))
    vectors = vocab.Vectors(embedding_file)  # cache=cache_dir vector.it

    text_field.build_vocab(
        train, dev, test, max_size=25000,
        vectors=vectors, unk_init=torch.Tensor.normal_)

    word_to_id = text_field.vocab.stoi

    label_field.build_vocab(train, dev, test)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(batch_size, batch_size, batch_size), sort_key=lambda x: len(x.address1 + x.address2),
        sort_within_batch=True, repeat=False, shuffle=True, device=device
    )

    return train_iter, dev_iter, test_iter, train, word_to_id
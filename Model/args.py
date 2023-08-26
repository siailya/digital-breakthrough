import argparse


def get_args(data_dir, cache_dir, embedding_folder, model_dir, log_dir):
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--model_name", default="TextMatchRCNN",
                        type=str, help="model name")
    parser.add_argument("--seed", default=1234, type=int, help="seed")

    # data_util
    parser.add_argument(
        "--data_path", default=data_dir, type=str, help="data path")
    parser.add_argument(
        "--cache_path", default=cache_dir, type=str, help="cache path"
    )
    parser.add_argument(
        "--sequence_length", default=50, type=int, help="sentence length"
    )

    # output file name
    parser.add_argument(
        "--model_dir", default=model_dir + "TextRCNN/", type=str, help="model save path"
    )
    parser.add_argument(
        "--log_dir", default=log_dir + "TextRCNN/", type=str, help="log path"
    )

    parser.add_argument("--do_train", default=False, type=bool, help="Whether to run training.")
    # parser.add_argument("--do_train",
    #                     action="store_true",
    #                     help="Run or not.")

    parser.add_argument("--print_step", default=100,
                        type=int, help="steps of print log")

    # hyper parameter
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch_num", default=25, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)

    # model
    parser.add_argument("--output_dim", default=2, type=int)

    # TextRNN
    parser.add_argument("--hidden_size", default=200, type=int, help="hidden layer dimension")
    parser.add_argument('--num_layers', default=2, type=int, help='hidden layer')
    parser.add_argument("--bidirectional", default=True, type=bool)

    # embedding
    parser.add_argument(
        '--word_embedding_file',
        default=embedding_folder + 'shenzhen_address_word2vec.bin',
        type=str,
        help='path of word embedding file')  # shenzhen data: default=embedding_folder + 'shenzhen_address_word2vec.bin'
    parser.add_argument(
        '--word_embedding_size',
        default=int(1873), type=int,
        help='word_embedding_size')  # shenzhen dataset:1873 jiangsu-hunna dataset:2030
    parser.add_argument(
        '--word_embedding_dim',
        default=300, type=int,
        help='word embedding size (default: 300)')

    config = parser.parse_args()

    return config

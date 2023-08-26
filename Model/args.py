import argparse


def get_args(data_dir, cache_dir, model_dir, log_dir):
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
        "--sequence_length", default=10, type=int, help="sentence length"
    )

    # output file name
    parser.add_argument(
        "--model_dir", default=model_dir + "TextRCNN/", type=str, help="model save path"
    )
    parser.add_argument(
        "--log_dir", default=log_dir + "TextRCNN/", type=str, help="log path"
    )

    parser.add_argument("--do_train", default=True, type=bool, help="Whether to run training.")
    # parser.add_argument("--do_train",
    #                     action="store_true",
    #                     help="Run or not.")

    parser.add_argument("--print_step", default=10,
                        type=int, help="steps of print log")

    # hyper parameter
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epoch_num", default=20, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)

    # model
    parser.add_argument("--output_dim", default=2, type=int)

    # TextRNN
    parser.add_argument("--hidden_size", default=400, type=int, help="hidden layer dimension")
    parser.add_argument('--num_layers', default=2, type=int, help='hidden layer')
    parser.add_argument("--bidirectional", default=True, type=bool)

    parser.add_argument(
        '--word_embedding_dim',
        default=128, type=int,
        help='word embedding size (default: 300)')

    config = parser.parse_args()

    return config

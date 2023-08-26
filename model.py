import numpy as np
from nltk.tokenize import WordPunctTokenizer
from subword_nmt.apply_bpe import BPE
from vocab import Vocab
from feature_extractor import FeatureExtractor

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

bpe = BPE(open('./bpe/bpe_rules', encoding='utf-8'))
data = np.array(open('./bpe/train.bpe', encoding='utf-8').read().split('\n'))[:-1]
inp_voc = Vocab.from_lines(data[:, 0])

tokenizer = WordPunctTokenizer()


def tokenize(x):
    return ' '.join(tokenizer.tokenize(x.lower()))


ext = FeatureExtractor(
    town_index_path="additional_data/town_20230808.csv",
    district_index_path="additional_data/district_20230808.csv",
    street_abbv_index_path="additional_data/geonimtype_20230808.csv",
    town_abbv_index_path="additional_data/subrf_20230808.csv"
)


class RNN_Model(nn.Module):
    def __init__(self, inp_voc, emb_size=64, hid_size=128, num_layers=5):
        """
        Базовая модель encoder-decoder архитектуры
        """
        super().__init__()

        self.inp_voc = inp_voc
        self.hid_size = hid_size

        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True, num_layers=num_layers)

        self.act = nn.ReLU()
        self.fc = nn.Linear(hid_size, hid_size)

        self.fc_with_features = nn.Linear(hid_size, hid_size)

    def forward(self, inp):
        """ Сначала примените  encode а затем decode"""
        ans = self.encode(inp)

        return ans

    def encode(self, inp, **flags):
        """
        Считаем скрытое состояние, которое будет начальным для decode
        :param inp: матрица входных токенов
        :returns: скрытое представление с которого будет начинаться decode
        """
        inp_emb = self.emb_inp(inp)
        batch_size = inp.shape[0]

        enc_seq, last_state_but_not_really = self.enc0(inp_emb)
        #         enc_seq, last_state_but_not_really = self.enc0(inp_emb)
        # enc_seq: [batch, time, hid_size], last_state: [batch, hid_size]

        # последний токен, не последние на самом деле, так как мы делали pading, чтобы тексты были
        # одинакового размер, поэтому подсчитать длину исходного предложения не так уж тривиально
        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        # ^-- shape: [batch_size, hid_size]
        out = self.fc(last_state)
        return last_state

    def sentence2emb(self, sentence: str) -> np.array:
        inp = inp_voc.to_matrix([bpe.process_line(
            tokenize(
                ext.resolve_abbv(sentence).lower().translate(
                    str.maketrans('', '', '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~'))
            )
        )]).to(device)
        emb = self.forward(inp).detach().cpu().flatten().numpy()
        return emb

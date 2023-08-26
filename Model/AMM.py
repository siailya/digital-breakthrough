import torch.nn as nn
import torch
from models.Linear import Linear


class AMM(nn.Module):

    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pretrained_embeddings):
        super(AMM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        self.W2 = Linear(2 * hidden_size + embedding_dim, hidden_size * 2)
        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.maxpool = nn.MaxPool1d(50)

        bidirect = 2 if bidirectional else 1
        self.input_dim = 5 * (hidden_size * bidirect)
        self.mid_dim = int(self.input_dim / 2)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.mid_dim),
            nn.Dropout(0.5),
            nn.Linear(self.mid_dim, 2)
        )

        self.linear_mid = int((hidden_size * bidirect) / 2)


    def forward(self, x, y):
        embedded = self.dropout(self.embedding(x))
        outputs, _ = self.rnn(embedded)  
        outputs = outputs.permute(1, 0, 2)
        embedded = embedded.permute(1, 0, 2)

        x = torch.cat((outputs, embedded), 2)
        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)

        features = self.maxpool(y2).squeeze()

        # RNN
        embedded2 = self.dropout(self.embedding(y))
        outputs2, _ = self.rnn(embedded2)

        outputs2 = outputs2.permute(1, 0, 2)
        embedded2 = embedded2.permute(1, 0, 2)

        # CNN
        y = torch.cat((outputs2, embedded2), 2)
        text2_y2 = torch.tanh(self.W2(y)).permute(0, 2, 1)
        features_y = self.maxpool(text2_y2).squeeze()


        features = torch.cat((features, torch.abs(features - features_y), features_y, features * features_y,
                              (features + features_y) / 2),1)
        output = self.classifier(features)

        return output


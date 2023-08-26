#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        """
        Args:
            input_size: x 的特征维度
            hidden_size: 隐层的特征维度
            num_layers: LSTM 层数
        """
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout
        )

        self.init_params()

    def init_params(self):  # 参考：https://zhuanlan.zhihu.com/p/64527432
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(
                    getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(
                    getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(
                    getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(
                    getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x, lengths):
        '''2种方式，其中pack_padded_sequence和pad_packed_sequence可能回导致x与y输出维度不匹配，所以使用方式2'''
        # 1采用pack_padded_sequence和pad_packed_sequence方式
        # x: [seq_len, batch_size, input_size]
        # lengths: [batch_size]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)  # 是否强制排序，最好设置为True，但是
        # 我的文本需要拆成两部分输入，所以没办法同时让两个输入都有序，一个输入的话可以，这一点自己考虑到还是不严谨
        # RuntimeError: `lengths` array must be sorted in decreasing order when `enforce_sorted` is True.
        # You can pass `enforce_sorted=False` to pack_padded_sequence and/or pack_sequence to sidestep this requirement if you do not need ONNX exportability.

        # packed_x， packed_output: PackedSequence 对象
        # hidden: [num_layers * bidirectional, batch_size, hidden_size]
        # cell: [num_layers * bidirectional, batch_size, hidden_size]
        packed_output, (hidden, cell) = self.rnn(packed_x)

        # output: [real_seq_len, batch_size, hidden_size * 2]
        # output_lengths: [batch_size]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)


        # # 2不采用pack_padded_sequence和pad_packed_sequence方式
        # output, (hidden, cell) = self.rnn(x)

        return hidden, output
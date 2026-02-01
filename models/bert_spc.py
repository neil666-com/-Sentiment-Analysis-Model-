# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        # 解析输入的索引和分段id，和你原代码解析方式保持一致
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # 适配transformers4.x：获取封装的输出对象，不再直接解包元组
        outputs = self.bert(input_ids=text_bert_indices, token_type_ids=bert_segments_ids)
        # 从输出对象中取池化张量，这是解决TypeError的核心
        pooled_output = outputs.pooler_output
        # 后续逻辑和你原代码完全一致，不用改
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits

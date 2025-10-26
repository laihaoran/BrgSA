import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import functools
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class GPTDecoder(nn.Module):
    def __init__(self, cfg):
        super(GPTDecoder, self).__init__()

        self.bert_type = cfg.model.text.bert_type
        self.last_n_layers = cfg.model.text.last_n_layers
        self.aggregate_method = cfg.model.text.aggregate_method
        self.norm = cfg.model.text.norm
        self.embedding_dim = cfg.model.text.embedding_dim
        self.freeze_bert = cfg.model.text.freeze_bert
        self.agg_tokens = cfg.model.text.agg_tokens

        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.emb_global, self.emb_local = None, None

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids, attn_mask, token_type):

        outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            word_embeddings, sent_embeddings = outputs[0], outputs[1]

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents



class EmbeddingFusingLayer(nn.Module):
    def __init__(self, in_features, feature_size):
        super(EmbeddingFusingLayer, self).__init__()
        self.query = nn.Linear(in_features, feature_size)
        self.key = nn.Linear(in_features, feature_size)
        self.value = nn.Linear(in_features, feature_size)

    def forward(self, features):
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)
        # ipdb.set_trace()
        sim = torch.einsum('bnd,bmd->bnm', q, k)
        weights = F.softmax(sim, dim=-1)
        fused = torch.einsum('bnd,bnm->bd', v, weights)
        return fused

class Linear_prob(nn.Module):
    def __init__(self, in_features, feature_size):
        super(Linear_prob, self).__init__()
        self.query = nn.Linear(in_features, feature_size)
        self.key = nn.Linear(in_features, feature_size)
        self.value = nn.Linear(in_features, feature_size)
    def forward(self, features):
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)
        return q, k, v

class EmbeddingFusing(nn.Module):
    def __init__(self, cfg):
        super(EmbeddingFusing, self).__init__()
        in_features = cfg.model.gpt.feature_size
        feature_size = cfg.model.gpt.embedding_dim
        # self.fc = nn.Linear(in_features, feature_size)
        # self.layernorm1 = nn.LayerNorm(feature_size)
        # self.linear_prob1 = Linear_prob(feature_size, feature_size)
        # self.multihead_attn1 = nn.MultiheadAttention(feature_size, num_heads=4, batch_first=True)
        # self.layernorm2 = nn.LayerNorm(feature_size)
        # self.fusion = EmbeddingFusingLayer(feature_size, feature_size)

        self.layernorm1 = nn.LayerNorm(in_features)
        self.linear_prob1 = Linear_prob(in_features, in_features)
        self.multihead_attn1 = nn.MultiheadAttention(in_features, num_heads=4, batch_first=True)
        self.layernorm2 = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, feature_size)
        self.layernorm3 = nn.LayerNorm(feature_size)
        self.fusion = EmbeddingFusingLayer(feature_size, feature_size)

    def forward(self, features):
        # ipdb.set_trace()
        # features = self.fc(features)
        features = self.layernorm1(features)
        q, k, v = self.linear_prob1(features)
        features = features + self.multihead_attn1(q, k, v)[0]
        features = self.layernorm2(features)
        features = self.fc(features)
        l_features = self.layernorm3(features)
        g_features = self.fusion(l_features)
        l_features = l_features.permute(0, 2, 1)
        return l_features, g_features
    
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout_rate):
        super(TransformerLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Self-attention
        attention_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)
        
        # Feedforward
        feedforward_output = self.feedforward(x)
        x = x + self.dropout(feedforward_output)
        x = self.norm2(x)       
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 计算位置编码矩阵
        self.position_encodings = torch.zeros(max_seq_len, embed_dim).half()
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.position_encodings[:, 0::2] = torch.sin(position * div_term).half()
        self.position_encodings[:, 1::2] = torch.cos(position * div_term).half()
        
        # 注册为模型参数
        # self.register_buffer('position_encodings', self.position_encodings)
    
    def forward(self, x):
        # 将位置编码加到输入序列张量上
        seq_len = x.size(1)
        position_encodings = self.position_encodings[:seq_len, :]
        position_encodings = position_encodings.unsqueeze(0)
        x = x + position_encodings.to(x.device)
        return x

class GPT(nn.Module):
    def __init__(self, embed_dim=1536, max_seq_len=59, num_layers=2, vocab_size=2304):
        super(GPT, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads=8, feedforward_dim=2048, dropout_rate=0.1) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        # 嵌入层
        # embedded = self.embedding(x)
        
        # 位置编码
        position_encoded = self.position_encoding(x)
        
        # Transformer层
        output = position_encoded
        for layer in self.transformer_layers:
            output = layer(output)
        
        # 全连接层，预测下一个token
        token_embeddings = self.fc(output[:, -1, :])
        
        return token_embeddings
    

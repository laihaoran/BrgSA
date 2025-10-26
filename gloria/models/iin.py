import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import ipdb

from transformers import AutoModel,BertConfig,AutoTokenizer

from ..models.transformer_decoder import *
# from transformer_decoder import *

class IIN_Model(nn.Module):
    def __init__(self, 
            cfg = None
            ):
        super().__init__()
        embed_dim = cfg.model.fusion.d_model
        class_num = cfg.model.fusion.class_num
        decoder_number_layer = cfg.model.fusion.decoder_number_layer

        self.d_model = embed_dim
        decoder_layer = TransformerDecoderLayer(self.d_model, 4, 1024,
                                        0.1, 'relu',normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, decoder_number_layer, self.decoder_norm,
                                return_intermediate=False)
        self.dropout_feas = nn.Dropout(0.1)

        self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
            nn.Linear(embed_dim, class_num)
        )
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(self, image_features, text_features, return_atten = False):
        #image_features (batch_size,dim)
        #text_features (batch_size,dim)
        batch_size = image_features.shape[0]
        image_features = image_features.unsqueeze(0) # (1, batch_size,dim)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1) # (batch_size,batch_size,dim)

        image_features = self.decoder_norm(image_features)
        text_features = self.decoder_norm(text_features)
        features,atten_map = self.decoder(text_features, image_features, 
                memory_key_padding_mask=None, pos=None, query_pos=None) 
        features = self.dropout_feas(features).transpose(0,1)  #b,embed_dim
        out = self.mlp_head(features)  #(batch_size, query_num)
        # ipdb.set_trace()
        # elif mode == 'local':
        #     #image_features (batch_size,patch_num,dim)
        #     #text_features (batch_size, word_number,dim)
        #     # batch_size = image_features.shape[0]
        #     image_features = image_features.transpose(0,1)  #(patch_num,batch_size,dim)
        #     # text_features = text_features.transpose(0,1)  #(word_number,batch_size,dim)
        #     # text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1) # (query_num,batch_size,dim)
        #     image_features = self.decoder_norm(image_features)
        #     text_features = self.decoder_norm(text_features)
        #     features,atten_map = self.decoder(text_features, image_features, 
        #             memory_key_padding_mask=None, pos=None, query_pos=None) 
        #     ipdb.set_trace()
        #     features = self.dropout_feas(features).transpose(0,1)  #b,embed_dim
        #     out = self.mlp_head(features)  #(batch_size, query_num)

        if return_atten:
            return out, atten_map
        else:
            return out
        


# if __name__ == '__main__':
#     net = IIN_Model(embed_dim=768,class_num=1, decoder_number_layer=2)

#     image_features = torch.randn(10, 768)
#     text_features = torch.randn(10,768)
#     out = net(image_features,text_features)
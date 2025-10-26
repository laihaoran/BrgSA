import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertConfig, BertModel
# from bert_model import BertCrossLayer
import ipdb
from ..models.bert_model import *

class DQN_M3AE(nn.Module):
    def __init__(self, 
            cfg = None
            ):
        super().__init__()
        embed_dim = cfg.model.fusion.d_model
        class_num = cfg.model.fusion.class_num
        decoder_number_layer = cfg.model.fusion.decoder_number_layer

        # embed_dim = 768
        # class_num = 1
        # decoder_number_layer = 6

        # add projection layer from M3AE
        self.multi_modal_language_proj = nn.Linear(embed_dim, embed_dim)
        self.multi_modal_vision_proj = nn.Linear(embed_dim, embed_dim)

        bert_config = BertConfig(
                vocab_size=28996,
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=768 * 4,
                max_position_embeddings=256,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )
        
        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(decoder_number_layer)])
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(decoder_number_layer)])
        
        self.dropout_feas = nn.Dropout(0.1)

        self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
            nn.Linear(embed_dim, class_num)
        )
        self._load_pretrained_model(cfg.model.fusion.pretrain_weight_path)

    def _load_pretrained_model(self, path_to_pretrained_model):
        """
        Private method to load the pretrained model weights.

        :param path_to_pretrained_model: Path to the pretrained model weights.
        """
        try:
            pretrained_weights = torch.load(path_to_pretrained_model, map_location='cpu')['state_dict']
            
            # Filtering out unnecessary keys
            model_dict = self.state_dict()

            pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict}
            # Overwriting entries in the existing state dict
            model_dict.update(pretrained_weights) 
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained weights from {path_to_pretrained_model}")
            
        except Exception as e:
            print(f"Error loading pretrained weights. Reason: {e}")
    
    def forward(self, batch, return_atten=False):
        
        global_img_feature, global_text_feature, local_img_feature, local_text_feature = batch[0], batch[1], batch[2], batch[3]

        local_img_feature = self.multi_modal_vision_proj(local_img_feature)
        local_text_feature = self.multi_modal_language_proj(local_text_feature)

        atten_map_text, atten_map_img = [], []
        for layer_idx, (text_layer, image_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                  self.multi_modal_vision_layers)):
            # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
            # == Begin: Co-Attention ==
            global_text_feature = text_layer(global_text_feature, local_img_feature, output_attentions=return_atten)
            global_img_feature = image_layer(global_img_feature, local_text_feature, output_attentions=return_atten)
            if return_atten:
                atten_map_text.append(global_text_feature[1])
                atten_map_img.append(global_img_feature[1])
            global_text_feature , global_img_feature = global_text_feature[0], global_img_feature[0]
      
        
        global_text_feature = self.dropout_feas(global_text_feature)  #b,embed_dim
        text2img_out = self.mlp_head(global_text_feature).squeeze(-1)  #(batch_size, query_num)

        global_img_feature = self.dropout_feas(global_img_feature)  #b,embed_dim
        img2text_out = self.mlp_head(global_img_feature).squeeze(-1)  #(batch_size, query_num)

        if return_atten:
            return text2img_out, img2text_out, atten_map_text, atten_map_img
        else:
            return text2img_out, img2text_out

    
if __name__ == '__main__':
    model = DQN_M3AE().cuda()
    global_img_feature = torch.ones((16, 16, 768)).cuda()
    global_text_feature =  torch.ones((16, 16, 768)).cuda()
    local_img_feature = torch.ones((16, 196, 768)).cuda()
    local_text_feature = torch.ones((16, 97, 768)).cuda()
    batch = [ global_img_feature, global_text_feature, local_img_feature, local_text_feature ]
    text2img_out, img2text_out = model(batch, return_atten=False)
   
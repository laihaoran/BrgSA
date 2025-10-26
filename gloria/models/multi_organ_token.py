import torch
import torch.nn as nn

class MultiOrganClsToken(nn.Module):
    def __init__(self, hidden_size, organ_types):
        super(MultiOrganClsToken, self).__init__()
        
        # 为每个器官创建独立的 cls_token 参数
        self.cls_tokens = nn.ParameterDict({
            organ: nn.Parameter(torch.zeros(1, 1, hidden_size)) for organ in organ_types
        })
    
    def initialize_from_pretrained(self, pretrained_state_dict):
        """
        从预训练的权重中提取单个 cls_token，并使用该权重初始化所有器官的 cls_token。
        """
        pretrained_cls_token = pretrained_state_dict['cls_token']
        
        # 将预训练的 cls_token 复制到每个器官的 cls_token
        for organ in self.cls_tokens.keys():
            self.cls_tokens[organ].data.copy_(pretrained_cls_token)
            # print(f"{organ} cls_token initialized with pretrained weights.")

# 使用示例
def create_multi_organ_cls_token(cfg):

    hidden_size = cfg.model.vision.hidden_size
    organ_types = cfg.model.vision.organ_types
    # 定义预训练模型路径
    pretrained_model_path = '/haoranlai/Project/M3D/VisionModel/vit_b16_3D_epoch_116.pth'
    # 加载预训练的权重文件
    pretrained_state_dict = torch.load(pretrained_model_path, map_location='cpu')
    
    # 创建 MultiOrganClsToken 实例
    multi_organ_cls_token = MultiOrganClsToken(hidden_size, organ_types)
    
    # 使用预训练的单个 cls_token 初始化每个器官的 cls_token
    multi_organ_cls_token.initialize_from_pretrained(pretrained_state_dict)
    
    return multi_organ_cls_token


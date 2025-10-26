import torch
import torch.nn as nn
from monai.networks.nets import ResNet as ResNet3D
from monai.networks.nets import vit
# import timm


def load_partial_state_dict(model, checkpoint_path):
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Get model weights
    model_dict = model.state_dict()

    # Filter out the weights that match the model
    matched_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

    # Print matched weights
    print("Matched keys:", len(matched_state_dict.keys()))

    # Update model weights
    model_dict.update(matched_state_dict)
    model.load_state_dict(model_dict)

    return model


class Identity(nn.Module):
    """Identity layer to replace the last fully connected layer"""
    def forward(self, x):
        return x


################################################################################
# ResNet Family for 3D
################################################################################


def resnet_18_3d(pretrained=True):
    model = ResNet3D(spatial_dims=3, n_input_channels=1, num_classes=18, conv1_t_stride=2, block='bottleneck', layers=[2, 2, 2, 2], block_inplanes = [64, 128, 256, 512]
)
    if pretrained:
        checkpoint = torch.load('/data2/haoranlai/Project/gloria/outputs_cls3d/last.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded resnet18 supervised model")
    feature_dims = model.fc.in_features
    model.fc = Identity()  # Replacing the final classification layer
    return model, feature_dims, 1024


def resnet_34_3d(pretrained=True):
    model = ResNet3D(spatial_dims=3, n_input_channels=1, num_classes=2, conv1_t_stride=2, block='bottleneck', layers=[3, 4, 6, 3], block_inplanes = [64, 128, 256, 512]
)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50_3d(pretrained=True):
    model = ResNet3D(spatial_dims=3, n_input_channels=1, num_classes=2, conv1_t_stride=2, block='bottleneck', layers=[3, 4, 6, 3], block_inplanes = [64, 128, 256, 512]
)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024



# import torch
# import torch.nn as nn
# import torchvision.models.video as tv_video
# from typing import Optional, Tuple


# def load_partial_state_dict(model: nn.Module, checkpoint_path: str) -> nn.Module:
#     """
#     从 checkpoint 部分加载匹配的权重（键名一致且 shape 一致），不严格。
#     兼容常见保存格式：{'model': state_dict} / {'state_dict': state_dict} / 直接 state_dict
#     """
#     state = torch.load(checkpoint_path, map_location="cpu")
#     if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
#         state_dict = state["model"]
#     elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
#         state_dict = state["state_dict"]
#     elif isinstance(state, dict):
#         state_dict = state
#     else:
#         raise ValueError(f"Unsupported checkpoint format: {type(state)}")

#     model_dict = model.state_dict()
#     matched = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
#     print(f"[load_partial_state_dict] Matched keys: {len(matched)}")
#     model_dict.update(matched)
#     model.load_state_dict(model_dict, strict=False)
#     return model


# class Identity(nn.Module):
#     """Identity layer to replace the last fully connected layer"""
#     def forward(self, x):
#         return x


# def _get_r3d_18(pretrained: bool) -> nn.Module:
#     """
#     兼容不同 torchvision 版本：
#     - 新版：weights="KINETICS400_V1"
#     - 旧版：pretrained=True/False
#     """
#     try:
#         # 新版 API：使用 weights 字符串
#         model = tv_video.r3d_18(weights="KINETICS400_V1" if pretrained else None)
#     except Exception:
#         # 兼容老版 API
#         model = tv_video.r3d_18(pretrained=pretrained)
#     return model


# ###############################################################################
# # 3D ResNet 家族（torchvision 版），保持原函数签名 & 返回值
# ###############################################################################
# def resnet_18_3d(pretrained: bool = True, custom_ckpt_path: Optional[str] = None) -> Tuple[nn.Module, int, int]:
#     """
#     用 torchvision 的 r3d_18 作为 3D ResNet18。
#     返回：(model, feature_dims, 1024)
#     - 输入必须是 3 通道：[B, 3, T, H, W]
#     - 如果提供 custom_ckpt_path，则在官方权重基础上做“部分匹配加载”
#     """
#     model = _get_r3d_18(pretrained=pretrained)

#     # 全连接输入维度
#     feature_dims = model.fc.in_features
#     # 替换分类头为 Identity，便于外部拿全局特征
#     model.fc = Identity()

#     # 可选：加载你自己的 checkpoint（部分匹配，不会因形状不符报错）
#     if custom_ckpt_path is not None:
#         try:
#             load_partial_state_dict(model, custom_ckpt_path)
#             print(f"[resnet_18_3d] Loaded custom checkpoint (partial match): {custom_ckpt_path}")
#         except FileNotFoundError:
#             print(f"[resnet_18_3d] Custom checkpoint not found: {custom_ckpt_path} (skipped)")
#         except Exception as e:
#             print(f"[resnet_18_3d] Failed to load custom checkpoint (skipped): {e}")
 
#     # 保持与原实现一致的第三个返回值 1024
#     return model, feature_dims, 256


# def resnet_34_3d(pretrained: bool = True, custom_ckpt_path: Optional[str] = None) -> Tuple[nn.Module, int, int]:
#     """
#     torchvision 暂无 r3d_34。为不破坏你现有调用，这里用 r3d_18 兜底（接口完全一致）。
#     如果后续需要真正的 34 层版本，我可以基于 VideoResNet 自定义构造。
#     """
#     print("[resnet_34_3d] torchvision 无现成 r3d_34，临时使用 r3d_18 作为替代骨干。")
#     return resnet_18_3d(pretrained=pretrained, custom_ckpt_path=custom_ckpt_path)


# def resnet_50_3d(pretrained: bool = True, custom_ckpt_path: Optional[str] = None) -> Tuple[nn.Module, int, int]:
#     """
#     torchvision 暂无 r3d_50。为不破坏你现有调用，这里用 r3d_18 兜底（接口完全一致）。
#     如果后续需要真正的 50 层 bottleneck 版本，我可以基于 VideoResNet 自定义构造。
#     """
#     print("[resnet_50_3d] torchvision 无现成 r3d_50，临时使用 r3d_18 作为替代骨干。")
#     return resnet_18_3d(pretrained=pretrained, custom_ckpt_path=custom_ckpt_path)







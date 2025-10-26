# # measure_clip3d_deploy.py
# import time
# from statistics import mean, pstdev
# import torch
# import torch.nn.functional as F
# import gloria
# # ----------------------------
# # 通用统计 & 实用函数
# # ----------------------------
# def _stats(ms_list):
#     ms_sorted = sorted(ms_list)
#     n = len(ms_sorted)
#     p50 = ms_sorted[n // 2]
#     p90 = ms_sorted[int(n * 0.9)]
#     return {
#         "mean_ms": mean(ms_sorted),
#         "std_ms": pstdev(ms_sorted) if n > 1 else 0.0,
#         "p50_ms": p50,
#         "p90_ms": p90,
#     }

# def _reset_peak(device):
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats(device)

# def _peak_mem(device):
#     alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
#     resvd = torch.cuda.max_memory_reserved(device) / (1024**3)
#     return {"peak_alloc_GB": alloc, "peak_reserved_GB": resvd}

# # ----------------------------
# # 1) 图像端 embedding 计时/显存
# # 要求: model.image_encode(x) -> [B, d]  或  你在下面改成你的接口
# # ----------------------------
# @torch.inference_mode()
# def measure_image_embed(model, x_vol, warmup=10, runs=100, amp=True):
#     """
#     x_vol: [1, C=1, D, H, W] 3D 体数据 (float tensor, on CPU)
#     返回: {"kernel": {...}, "e2e": {...}, "mem": {...}}
#     """
#     device = next(model.parameters()).device
#     x_vol = x_vol.to(device, non_blocking=True)

#     # 热身
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(warmup):
#             _ = model.img_encoder(x_vol)  # <- 如接口不同，请改成你的
#     torch.cuda.synchronize()

#     # kernel time (CUDA events)
#     start, end = torch.cuda.Event(True), torch.cuda.Event(True)
#     kernel_ms = []
#     _reset_peak(device)
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(runs):
#             start.record()
#             _ = model.img_encoder(x_vol)
#             end.record()
#             torch.cuda.synchronize()
#             kernel_ms.append(start.elapsed_time(end))
#     mem1 = _peak_mem(device)

#     # end-to-end（本阶段与kernel接近；若你有预/后处理，可在此包进去）
#     e2e_ms = []
#     _reset_peak(device)
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(runs):
#             t0 = time.perf_counter()
#             _ = model.img_encoder(x_vol)
#             torch.cuda.synchronize()
#             e2e_ms.append((time.perf_counter() - t0) * 1000)
#     mem2 = _peak_mem(device)

#     return {
#         "kernel": _stats(kernel_ms),
#         "e2e": _stats(e2e_ms),
#         "mem_kernel": mem1,
#         "mem_e2e": mem2,
#     }

# # ----------------------------
# # 2) 文本端 embedding 计时/显存
# # 要求: model.text_encode(input_ids, attention_mask) -> [B, d]
# # ----------------------------
# @torch.inference_mode()
# def measure_text_embed(model, input_ids, attention_mask, token_type_ids, warmup=20, runs=200, amp=True):
#     device = next(model.parameters()).device
#     input_ids = input_ids.to(device, non_blocking=True)
#     attention_mask = attention_mask.to(device, non_blocking=True)
#     if token_type_ids is not None:
#         token_type_ids = token_type_ids.to(device, non_blocking=True)


#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(warmup):
#             # _ = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
#             _ = model.text_encoder(ids=input_ids, attn_mask=attention_mask, token_type=token_type_ids)
#     torch.cuda.synchronize()

#     start, end = torch.cuda.Event(True), torch.cuda.Event(True)
#     kernel_ms = []
#     _reset_peak(device)
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(runs):
#             start.record()
#             # _ = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
#             _ = model.text_encoder(ids=input_ids, attn_mask=attention_mask, token_type=token_type_ids)
#             end.record()
#             torch.cuda.synchronize()
#             kernel_ms.append(start.elapsed_time(end))
#     mem1 = _peak_mem(device)

#     e2e_ms = []
#     _reset_peak(device)
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(runs):
#             t0 = time.perf_counter()
#             # _ = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
#             _ = model.text_encoder(ids=input_ids, attn_mask=attention_mask, token_type=token_type_ids)
#             torch.cuda.synchronize()
#             e2e_ms.append((time.perf_counter() - t0) * 1000)
#     mem2 = _peak_mem(device)

#     return {
#         "kernel": _stats(kernel_ms),
#         "e2e": _stats(e2e_ms),
#         "mem_kernel": mem1,
#         "mem_e2e": mem2,
#     }

# # ----------------------------
# # 3) 余弦相似度 + Top-K 计时/显存
# # 输入: z_txt [1,d], Z_img_gallery [N,d] (建议都已 L2 normalize, 同一设备)
# # ----------------------------
# @torch.inference_mode()
# def measure_similarity_topk(z_txt, Z_img_gallery, k=100, warmup=20, runs=200):
#     device = z_txt.device
#     if Z_img_gallery.device.type != z_txt.device.type:
#         Z_img_gallery = Z_img_gallery.to(device, non_blocking=True)

#     # 统一 L2 归一化，得到真正的余弦相似度
#     z_txt = F.normalize(z_txt, dim=-1)
#     Z_img_gallery = F.normalize(Z_img_gallery, dim=-1)

#     # 热身
#     for _ in range(warmup):
#         sims = z_txt @ Z_img_gallery.t()   # [1, N]
#         _ = torch.topk(sims, k, dim=-1)
#     torch.cuda.synchronize()

#     start, end = torch.cuda.Event(True), torch.cuda.Event(True)
#     kernel_ms = []
#     _reset_peak(device)
#     for _ in range(runs):
#         start.record()
#         sims = z_txt @ Z_img_gallery.t()
#         _ = torch.topk(sims, k, dim=-1)
#         end.record()
#         torch.cuda.synchronize()
#         kernel_ms.append(start.elapsed_time(end))
#     mem = _peak_mem(device)

#     return {"kernel": _stats(kernel_ms), "mem": mem}

# # ----------------------------
# # 4) 端到端 (E2E) 检索两种口径
# #  - Offline: 图库图像向量已离线预计算 (推荐部署口径)
# #  - Online: 端到端同时计算图像向量 (上界参考)
# # 要求: model.image_encode, model.text_encode 存在
# # ----------------------------
# @torch.inference_mode()
# def measure_e2e_offline(model, x_vol, input_ids, attention_mask, token_type_ids, Z_img_gallery, k=100,
#                         warmup=10, runs=100, amp=True):
#     """
#     E2E-Offline: 仅在线计算文本向量 + 相似度 + TopK（图库向量已在 GPU）
#     与 Gloria 接口对齐: model.text_encoder(ids=..., attn_mask=..., token_type=...)
#     """
#     device = next(model.parameters()).device

#     # ---- 准备输入，确保 dtype 正确 ----
#     input_ids      = input_ids.to(device, dtype=torch.long,  non_blocking=True)
#     attention_mask = attention_mask.to(device, dtype=torch.long, non_blocking=True)
#     # Gloria 的 text_encoder 这里没有用到 token_type（若你需要可加参数），保持与上游一致

#     # ---- 准备图库向量到 GPU，并一次性做 L2 归一化 / dtype 对齐 ----
#     if Z_img_gallery.device.type != "cuda":
#         Z_img_gallery = Z_img_gallery.to(device, non_blocking=True)
#     # 归一化只做一次，避免循环内重复
#     Z_img_gallery = F.normalize(Z_img_gallery, dim=-1)
#     gallery_dtype = Z_img_gallery.dtype  # 一般是 float16
#     N = Z_img_gallery.shape[0]
#     k_eff = min(int(k), int(N)) if N > 0 else 0

#     # ---- 一个小工具：抽取文本向量（兼容多种返回风格）----
#     def _encode_text_once():
#         out = model.text_encoder(ids=input_ids, attn_mask=attention_mask, token_type=token_type_ids)
#         # 可能是 tuple/list/dict/tensor
#         if isinstance(out, (tuple, list)):
#             # Gloria: 通常第 2 个返回是 text embedding
#             z = out[1]
#         elif isinstance(out, dict):
#             # 常见 key 尝试
#             for key in ("text_emb", "z_txt", "txt", "last_hidden_state", "pooler_output"):
#                 if key in out and torch.is_tensor(out[key]):
#                     z = out[key]
#                     break
#             else:
#                 # 兜底：取第一个 tensor 值
#                 tensors = [v for v in out.values() if torch.is_tensor(v)]
#                 if len(tensors) == 0:
#                     raise RuntimeError("text_encoder returned dict without tensor values.")
#                 z = tensors[0]
#         else:
#             z = out
#         if z.dim() == 1:
#             z = z.unsqueeze(0)  # -> [1, d]
#         return z

#     # ---- 热身 ----
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(warmup):
#             z_txt = _encode_text_once()
#             z_txt = F.normalize(z_txt, dim=-1).to(gallery_dtype)
#             if k_eff > 0:
#                 sims = z_txt @ Z_img_gallery.t()
#                 _ = torch.topk(sims, k_eff, dim=-1)
#     torch.cuda.synchronize()

#     # ---- 正式计时（端到端：文本→相似度→TopK）----
#     e2e_ms = []
#     _reset_peak(device)
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(runs):
#             t0 = time.perf_counter()
#             z_txt = _encode_text_once()
#             z_txt = F.normalize(z_txt, dim=-1).to(gallery_dtype)
#             if k_eff > 0:
#                 sims = z_txt @ Z_img_gallery.t()
#                 _ = torch.topk(sims, k_eff, dim=-1)
#             torch.cuda.synchronize()
#             e2e_ms.append((time.perf_counter() - t0) * 1000.0)
#     mem = _peak_mem(device)

#     return {"e2e": _stats(e2e_ms), "mem": mem}

# @torch.inference_mode()
# def measure_e2e_online(model, x_vol, input_ids, attention_mask, gallery_volumes,
#                        k=100, warmup=5, runs=20, amp=True):
#     """
#     E2E-Online: 同时在线计算 图像向量(查询+图库) + 文本向量 + 相似度 + TopK
#     注意：很慢，仅用于上界参考或小图库。
#     gallery_volumes: list/iterable of 3D volumes (each [1,1,D,H,W])，此函数将串行编码成矩阵
#     """
#     device = next(model.parameters()).device
#     x_vol = x_vol.to(device, non_blocking=True)
#     input_ids = input_ids.to(device, non_blocking=True)
#     attention_mask = attention_mask.to(device, non_blocking=True)

#     # 准备图库向量（热身时也做一次）
#     def build_gallery():
#         feats = []
#         with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#             for v in gallery_volumes:
#                 v = v.to(device, non_blocking=True)
#                 z = model.image_encode(v)
#                 feats.append(F.normalize(z, dim=-1))
#         return torch.cat(feats, dim=0)  # [N, d]

#     # 热身
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(warmup):
#             Z_img = build_gallery()
#             z_txt = model.text_encode(input_ids=input_ids, attention_mask=attention_mask)
#             sims = F.normalize(z_txt, dim=-1) @ Z_img.t()
#             _ = torch.topk(sims, k, dim=-1)
#     torch.cuda.synchronize()

#     # 正式计时
#     e2e_ms = []
#     _reset_peak(device)
#     with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
#         for _ in range(runs):
#             t0 = time.perf_counter()
#             Z_img = build_gallery()
#             z_txt = model.text_encode(input_ids=input_ids, attention_mask=attention_mask)
#             sims = F.normalize(z_txt, dim=-1) @ Z_img.t()
#             _ = torch.topk(sims, k, dim=-1)
#             torch.cuda.synchronize()
#             e2e_ms.append((time.perf_counter() - t0) * 1000)
#     mem = _peak_mem(device)
#     return {"e2e": _stats(e2e_ms), "mem": mem}

# # ----------------------------
# # 使用示例（按需替换）
# # ----------------------------
# if __name__ == "__main__":
#     """
#     假设:
#       - 你的 model 有方法: model.image_encode(x_vol), model.text_encode(input_ids, attention_mask)
#       - x_vol: torch.randn(1,1,D,H,W)
#       - input_ids/attention_mask: 来自你的 tokenizer
#       - Z_img_gallery: 预先算好的图库图像向量 [N, d] (可放 GPU)
#     """
#     # 伪代码示例（请替换为你的真实对象）
#     class DummyModel(torch.nn.Module):
#         def __init__(self, d=1024):
#             super().__init__()
#             self.conv = torch.nn.Conv3d(1, 8, 3, padding=1)
#             self.pool = torch.nn.AdaptiveAvgPool3d(1)
#             self.proj_img = torch.nn.Linear(8, d)
#             self.proj_txt = torch.nn.Linear(16, d)
#         def image_encode(self, x):
#             y = self.pool(torch.relu(self.conv(x))).flatten(1)
#             return self.proj_img(y)
#         def text_encode(self, input_ids, attention_mask=None):
#             # 假设输入已经是 embedding（示例用），实际替换为你的文本编码器
#             return self.proj_txt(input_ids.float())
        
#     # brgsa
    
    
#     torch.cuda.init()
#     device = torch.device("cuda:0")
#     # model = DummyModel().to(device).eval()
#     model = gloria.load_gloria(name="gloria_resnet50", device=device).eval()



#     # 构造假数据
#     D, H, W = 112, 224, 224
#     # D, H, W = 96, 208, 208
#     D, H, W = 112, 112, 112
#     x_vol = torch.randn(1, 1, W, H, D)
#     input_ids = torch.randint(0,1000,(1,16))      # 伪 token ids
#     attention_mask = torch.ones_like(input_ids)
#     token_type_ids = torch.zeros_like(input_ids)
#     Z_img_gallery = torch.randn(1000, 768, device=device).half()  # 预计算好的图库向量 (示例)

#     input_ids      = input_ids.to(device, dtype=torch.long, non_blocking=True)
#     attention_mask = attention_mask.to(device, dtype=torch.long, non_blocking=True)  # 或 bool
#     token_type_ids = token_type_ids.to(device, dtype=torch.long, non_blocking=True)


#     # 1) 单阶段测量
#     img_stats = measure_image_embed(model, x_vol, amp=True)
#     txt_stats = measure_text_embed(model, input_ids, attention_mask, token_type_ids, amp=True)
#     # 假设先得到一个文本向量以便测 sim+topk
#     with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=True):
#         # z_txt = model.text_encode(input_ids.to(device), attention_mask.to(device))
#         _, z_txt, _ = model.text_encoder(input_ids, attention_mask, token_type_ids)
        
#     sim_stats = measure_similarity_topk(z_txt, Z_img_gallery, k=100)



#     # 2) 端到端（推荐：Offline）
#     e2e_off = measure_e2e_offline(model, x_vol, input_ids, attention_mask, token_type_ids,Z_img_gallery, k=100, amp=True)

#     # 3) 如需 Online（小图库/上界参考）
#     # e2e_on = measure_e2e_online(model, x_vol, input_ids, attention_mask, gallery_volumes=[x_vol]*64, k=50, amp=True)

#     # 打印结果（示例）
#     print("Image Embed:", img_stats)
#     print("Text  Embed:", txt_stats)
#     print("Sim+TopK   :", sim_stats)
#     print("E2E Offline:", e2e_off)

#     time_s = e2e_off["e2e"]["mean_ms"] / 1000.0
#     gpu_gb = e2e_off["mem"]["peak_alloc_GB"]
#     print(f"\n[For Table] Time (s/study): {time_s:.6f} | GPU Mem (GB): {gpu_gb:.6f}")
















# ======== Zero-shot Classification: 1 volume + K texts ========

import time
from statistics import mean, pstdev
import torch
import torch.nn.functional as F

# ---------- 通用统计 ----------
def _stats(ms_list):
    ms_sorted = sorted(float(x) for x in ms_list)
    n = len(ms_sorted)
    p50 = ms_sorted[n // 2]
    p90 = ms_sorted[int(n * 0.9)]
    return {
        "mean_ms": mean(ms_sorted),
        "std_ms": pstdev(ms_sorted) if n > 1 else 0.0,
        "p50_ms": p50,
        "p90_ms": p90,
    }

def _reset_peak(device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

def _peak_mem(device):
    alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
    resvd = torch.cuda.max_memory_reserved(device) / (1024**3)
    return {"peak_alloc_GB": alloc, "peak_reserved_GB": resvd}

# ---------- 抽取 text embedding（兼容不同返回风格） ----------
@torch.inference_mode()
def _encode_text_batch(model, input_ids, attention_mask, token_type_ids=None, amp=True):
    """
    输入: ids/mask/token_type 形状 [K, L]
    返回: z_txt [K, d]，未归一化（外面统一做 F.normalize）
    """
    dev = next(model.parameters()).device
    input_ids      = input_ids.to(dev, dtype=torch.long, non_blocking=True)
    attention_mask = attention_mask.to(dev, dtype=torch.long, non_blocking=True)
    token_type_ids = (token_type_ids.to(dev, dtype=torch.long, non_blocking=True)
                      if token_type_ids is not None else None)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        # 兼容关键字/位置参数两种写法
        try:
            out = model.text_encoder(ids=input_ids,
                                     attn_mask=attention_mask,
                                     token_type=token_type_ids)
        except TypeError:
            out = model.text_encoder(input_ids, attention_mask, token_type_ids)

    # 兼容 tuple/list/dict/tensor
    if isinstance(out, (tuple, list)):
        # 你的 gloria 返回通常是 (loss?, z_txt, extra?)，第二个是文本向量
        z = out[1]
    elif isinstance(out, dict):
        for key in ("text_emb", "z_txt", "txt", "last_hidden_state", "pooler_output"):
            if key in out and torch.is_tensor(out[key]):
                z = out[key]; break
        else:
            ts = [v for v in out.values() if torch.is_tensor(v)]
            if not ts:
                raise RuntimeError("text_encoder dict 返回不含张量")
            z = ts[0]
    else:
        z = out

    if z.dim() == 1:
        z = z.unsqueeze(0)  # [1, d]
    return z  # [K, d]

# ---------- 抽取 image embedding ----------
@torch.inference_mode()
def _encode_image_once(model, x_vol, amp=True):
    """
    x_vol: [1, 1, D, H, W] on CPU
    返回：z_img [1, d]
    """
    dev = next(model.parameters()).device
    x_vol = x_vol.to(dev, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        z_img = model.img_encoder(x_vol)
    if isinstance(z_img, (tuple, list)):
        z_img = z_img[0]
    if z_img.dim() == 1:
        z_img = z_img.unsqueeze(0)
    return z_img  # [1, d]

# ---------- 核心：零样本分类端到端计时（图像 + K 文本 → 余弦相似度） ----------
@torch.inference_mode()
def measure_zeroshot_cls_e2e(model, x_vol,
                             input_ids, attention_mask, token_type_ids=None,
                             warmup=10, runs=100, amp=True, return_pred=False):
    """
    端到端：一次前向做完
      1) 图像嵌入（1 次）
      2) 文本嵌入（K 条，批量）
      3) 归一化 + 余弦相似度（[1,d] @ [K,d]^T）
      4) 可选 Top-1
    返回: {"kernel": {...}, "e2e": {...}, "mem": {...}, "pred": (idx, conf)}（可选）
    """
    dev = next(model.parameters()).device
    # 预先把输入搬到 GPU 并保证 dtype
    x_vol = x_vol.to(dev, non_blocking=True)
    input_ids      = input_ids.to(dev, dtype=torch.long, non_blocking=True)
    attention_mask = attention_mask.to(dev, dtype=torch.long, non_blocking=True)
    token_type_ids = (token_type_ids.to(dev, dtype=torch.long, non_blocking=True)
                      if token_type_ids is not None else None)

    # ---- 热身（完整路径）----
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            z_img = _encode_image_once(model, x_vol, amp=amp)            # [1,d]
            z_txt = _encode_text_batch(model, input_ids, attention_mask, token_type_ids, amp=amp)  # [K,d]
            z_img = F.normalize(z_img, dim=-1)
            z_txt = F.normalize(z_txt, dim=-1)
            sims  = z_img @ z_txt.t()  # [1, K]
            _ = torch.argmax(sims, dim=-1)
    torch.cuda.synchronize()

    # ---- CUDA kernel 时间（GPU阶段）----
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    kernel_ms = []
    _reset_peak(dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        for _ in range(runs):
            start.record()
            z_img = _encode_image_once(model, x_vol, amp=amp)
            z_txt = _encode_text_batch(model, input_ids, attention_mask, token_type_ids, amp=amp)
            z_img = F.normalize(z_img, dim=-1)
            z_txt = F.normalize(z_txt, dim=-1)
            sims  = z_img @ z_txt.t()
            _ = torch.argmax(sims, dim=-1)
            end.record()
            torch.cuda.synchronize()
            kernel_ms.append(start.elapsed_time(end))
    mem_kernel = _peak_mem(dev)

    # ---- 端到端 e2e（含 CPU 微小开销；更贴近真实）----
    e2e_ms = []
    _reset_peak(dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        for _ in range(runs):
            t0 = time.perf_counter()
            z_img = _encode_image_once(model, x_vol, amp=amp)
            z_txt = _encode_text_batch(model, input_ids, attention_mask, token_type_ids, amp=amp)
            z_img = F.normalize(z_img, dim=-1)
            z_txt = F.normalize(z_txt, dim=-1)
            sims  = z_img @ z_txt.t()          # [1, K]
            pred_idx = torch.argmax(sims, dim=-1)  # [1]
            torch.cuda.synchronize()
            e2e_ms.append((time.perf_counter() - t0) * 1000.0)
    mem_e2e = _peak_mem(dev)

    out = {
        "kernel": _stats(kernel_ms),
        "e2e": _stats(e2e_ms),
        "mem_kernel": mem_kernel,
        "mem_e2e": mem_e2e,
    }

    if return_pred:
        # 同步拿一次预测与置信度
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            z_img = _encode_image_once(model, x_vol, amp=amp)
            z_txt = _encode_text_batch(model, input_ids, attention_mask, token_type_ids, amp=amp)
            z_img = F.normalize(z_img, dim=-1)
            z_txt = F.normalize(z_txt, dim=-1)
            sims  = z_img @ z_txt.t()
            probs = torch.softmax(sims, dim=-1)  # [1, K]
            idx   = int(torch.argmax(probs, dim=-1).item())
            conf  = float(probs[0, idx].item())
        out["pred"] = {"top1_idx": idx, "top1_conf": conf}

    return out

# ---------------------------- 使用示例 ----------------------------
if __name__ == "__main__":
    """
    假设:
      - model: 你的 GLORIA/CLIP3D 类模型，含：
          model.img_encoder(x_vol) -> [1, d]
          model.text_encoder(ids=..., attn_mask=..., token_type=...) -> [K, d] 或兼容返回
      - x_vol: torch.randn(1,1,D,H,W)  (CPU)
      - input_ids/attention_mask/token_type_ids: [K, L]，K=类别数（如 18）
    """
    import gloria
    torch.cuda.init()
    device = torch.device("cuda:0")

    # 你自己的模型
    model = gloria.load_gloria(name="gloria_resnet50", device=device).eval()

    # 构造假数据：1 张 CT + K 个疾病文本（ids 版）
    K, L = 18, 16
    # D, H, W = 112, 224, 224
    D, H, W = 112, 112, 112
    x_vol = torch.randn(1, 1, D, H, W)         # 注意顺序按你的模型所需调整

    input_ids = torch.randint(0, 1000, (K, L))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)

    # 计时 & 显存（端到端：图像 + K 文本 + 余弦 + argmax）
    zs_stats = measure_zeroshot_cls_e2e(
        model, x_vol,
        input_ids, attention_mask, token_type_ids,
        warmup=10, runs=100, amp=True, return_pred=True
    )

    print("Zero-shot Classification (E2E):", zs_stats)
    print(f"[For Table] Time (s/study): {zs_stats['e2e']['mean_ms']/1000:.6f} | "
          f"GPU Mem (GB): {zs_stats['mem_e2e']['peak_alloc_GB']:.3f}")

    if "pred" in zs_stats:
        print(f"Top-1 class idx: {zs_stats['pred']['top1_idx']}  "
              f"Conf: {zs_stats['pred']['top1_conf']:.4f}")


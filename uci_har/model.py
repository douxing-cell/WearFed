import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class CNN_V3_V4(nn.Module):
    def __init__(self, out_1=32, out_2=64, out_3=128, number_of_classes=100, p=0.2):
        super(CNN_V3_V4, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.drop_conv = nn.Dropout(p=0.2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)

        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv3_bn = nn.BatchNorm2d(out_3)

        self.fc1 = nn.Linear(out_3 * 4 * 4, 2500)
        self.drop = nn.Dropout(p=p)
        self.fc1_bn = nn.BatchNorm1d(2500)

        self.fc2 = nn.Linear(2500, 2500)
        self.fc2_bn = nn.BatchNorm1d(2500)

        self.fc3 = nn.Linear(2500, 1000)
        self.fc3_bn = nn.BatchNorm1d(1000)

        self.fc4 = nn.Linear(1000, 1000)
        self.fc4_bn = nn.BatchNorm1d(1000)

        self.fc5 = nn.Linear(1000, 100)
        self.fc5_bn = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = self.maxpool1(x)
        x = self.drop_conv(x)

        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.drop_conv(x)

        x = self.cnn3(x)
        x = self.conv3_bn(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.drop_conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if x.shape[0] > 1:
            x = self.fc1_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.fc2_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc3(x)
        if x.shape[0] > 1:
            x = self.fc3_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc4(x)
        if x.shape[0] > 1:
            x = self.fc4_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc5(x)
        if x.shape[0] > 1:
            x = self.fc5_bn(x)

        return x


# =========================
# INT8 quantization support (Fake INT8, STE) - for compute simulation
# 仅用于 MSMALL：Quantized + Tiny
# =========================
class _FakeInt8QuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if x.numel() == 0:
            return x
        max_abs = x.detach().abs().max()
        scale = (max_abs / 127.0).clamp_min(eps)
        q = torch.clamp(torch.round(x / scale), -127.0, 127.0)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def fake_int8_quant(x: torch.Tensor) -> torch.Tensor:
    return _FakeInt8QuantSTE.apply(x)


# =========================
# True INT8 payload (for communication/storage)
# - Used by MSMALL to generate real int8 state_dict + per-tensor scales
# - Aggregation can dequantize back to float on server side
# =========================
@torch.no_grad()
def quantize_tensor_int8(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor int8 quantization: x ~= (q.int8) * scale"""
    if x.numel() == 0:
        return x.to(torch.int8), torch.tensor(1.0, device=x.device, dtype=torch.float32)
    max_abs = x.detach().abs().max()
    scale = (max_abs / 127.0).clamp_min(eps).to(torch.float32)
    q = torch.clamp(torch.round(x / scale), -127.0, 127.0).to(torch.int8)
    return q, scale


@torch.no_grad()
def dequantize_tensor_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale.to(torch.float32)


class UCIHARModel(nn.Module):
    """
    关键保证（用于 4.3）：
      - forward(x) 仍返回 logits（不影响 server.evaluate / 原有训练逻辑）
      - extract_features(x) 返回固定维度 d_full (= hidden_dims[-1]) 的特征向量
        即：无论 profile 是 FULL/PRUNED/MSMALL，都 pad 到 full dim，确保 prototypes 维度一致。
    """

    def __init__(self, input_dim=561, hidden_dims=[256, 128, 64], number_of_classes=6, p=0.2):
        super(UCIHARModel, self).__init__()

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(d) for d in hidden_dims]
        self.number_of_classes = int(number_of_classes)

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=p)

        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, self.number_of_classes)

        # 文档三档：
        #   FULL    : full model
        #   PRUNED  : structured pruning（这里用 hidden dim 比例裁剪来等价实现）
        #   MSMALL  : Quantized + Tiny（25% 容量 + INT8 量化）
        self._profile_ratios = {
            "FULL": 1.0,
            "PRUNED": 0.5,
            "MSMALL": 0.25,
        }
        self.active_profile = "FULL"
        self.active_ratio = 1.0

        # 4.3: full feature dim (fixed prototype dimension)
        self.full_feat_dim = int(self.hidden_dims[-1])

    def set_profile(self, profile: str) -> None:
        if profile is None:
            profile = "FULL"
        profile = str(profile).upper()
        if profile not in self._profile_ratios:
            raise ValueError(f"Unknown profile: {profile}. Supported: {list(self._profile_ratios.keys())}")
        self.active_profile = profile
        self.active_ratio = float(self._profile_ratios[profile])

    def _active_dims(self) -> List[int]:
        ratio = float(self.active_ratio)
        return [max(1, int(d * ratio)) for d in self.hidden_dims]

    @staticmethod
    def _bn_forward_slice(bn: nn.BatchNorm1d, x: torch.Tensor, d: int) -> torch.Tensor:
        return F.batch_norm(
            x[:, :d],
            bn.running_mean[:d],
            bn.running_var[:d],
            bn.weight[:d] if bn.affine else None,
            bn.bias[:d] if bn.affine else None,
            bn.training,
            bn.momentum,
            bn.eps,
        )

    def _forward_features_active(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 active 最后层特征，维度随 profile 变化（FULL=64/PRUNED=32/MSMALL=16...）
        """
        active_dims = self._active_dims()
        use_fake_int8 = (self.active_profile == "MSMALL")

        if use_fake_int8:
            x = fake_int8_quant(x)

        for i, (linear, bn) in enumerate(zip(self.layers, self.bn_layers)):
            out_dim = active_dims[i]
            in_dim = x.shape[1]

            w = linear.weight[:out_dim, :in_dim]
            b = linear.bias[:out_dim] if linear.bias is not None else None

            if use_fake_int8:
                w = fake_int8_quant(w)
                x_in = fake_int8_quant(x)
            else:
                x_in = x

            x = F.linear(x_in, w, b)

            if x.shape[0] > 1:
                x = self._bn_forward_slice(bn, x, out_dim)

            x = F.relu(x)
            x = self.dropout(x)

            if use_fake_int8:
                x = fake_int8_quant(x)

        return x  # (B, d_active_last)

    @staticmethod
    def _pad_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[1] == dim:
            return x
        if x.shape[1] > dim:
            return x[:, :dim]
        pad = dim - x.shape[1]
        return F.pad(x, (0, pad), mode="constant", value=0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats_active = self._forward_features_active(x)   # (B, d_active)
        feats_full = self._pad_to_dim(feats_active, self.full_feat_dim)  # (B, d_full)
        return feats_full

    # -------------------------
    # MSMALL: real INT8 payload export (for comm/storage)
    # -------------------------
    @torch.no_grad()
    def export_int8_state_dict(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Export a *real* int8 payload of parameters for MSMALL.
        Returns:
          q_state:  {param_name: int8 tensor}
          scales:   {param_name: float32 scalar tensor}
        Notes:
          - This does NOT change model parameters in-place.
          - Server can dequantize with: dequantize_tensor_int8(q_state[k], scales[k]).
        """
        q_state: Dict[str, torch.Tensor] = {}
        scales: Dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            q, s = quantize_tensor_int8(p.detach().cpu())
            q_state[name] = q.contiguous()
            scales[name] = s.detach().cpu().view(1)
        return q_state, scales

    @staticmethod
    @torch.no_grad()
    def estimate_int8_payload_bytes(
        q_state: Dict[str, torch.Tensor],
        scales: Dict[str, torch.Tensor],
    ) -> int:
        total = 0
        for k, v in q_state.items():
            total += int(v.numel())  # int8 -> 1 byte
            s = scales.get(k, None)
            if s is not None:
                total += int(s.numel()) * 4  # float32 scale
        return int(total)

    def forward(self, x):
        feats_active = self._forward_features_active(x)
        last_dim = feats_active.shape[1]

        w_out = self.output_layer.weight[:, :last_dim]
        b_out = self.output_layer.bias

        if self.active_profile == "MSMALL":
            w_out = fake_int8_quant(w_out)
            feats_in = fake_int8_quant(feats_active)
        else:
            feats_in = feats_active

        logits = F.linear(feats_in, w_out, b_out)
        return logits


if __name__ == "__main__":
    model_mmtv5 = CNN_V3_V4(out_1=32, out_2=64, out_3=128, number_of_classes=100, p=0.2)

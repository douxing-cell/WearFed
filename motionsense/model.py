
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F





class _FakeInt8QuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            max_val = torch.max(torch.abs(x)).clamp_min(1e-12)
            scale = max_val / 127.0
            q = torch.round(x / scale).clamp(-127, 127)
            return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def fake_int8_quant(x: torch.Tensor) -> torch.Tensor:
    return _FakeInt8QuantSTE.apply(x)





@torch.no_grad()
def quantize_int8_tensor(x: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.detach().to(torch.float32)
    max_val = torch.max(torch.abs(x_fp32)).clamp_min(eps)
    scale = (max_val / 127.0).to(torch.float32)
    q = torch.round(x_fp32 / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


class CNN_V3_V4(nn.Module):
    








    def __init__(
        self,
        out_1: int = 32,
        out_2: int = 64,
        out_3: int = 128,
        number_of_classes: int = 6,
        p: float = 0.2,
        window_size: int = 128,
        num_features: int = 12,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.num_features = int(num_features)
        self.number_of_classes = int(number_of_classes)

        
        self.out_1 = int(out_1)
        self.out_2 = int(out_2)
        self.out_3 = int(out_3)

        self.cnn1 = nn.Conv1d(in_channels=num_features, out_channels=out_1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.cnn2 = nn.Conv1d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.cnn3 = nn.Conv1d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(out_3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.drop_conv = nn.Dropout(p=0.2)
        self.drop = nn.Dropout(p=p)

        
        L = self.window_size // 8
        self.fc_in_full = int(out_3 * L)

        self.fc1 = nn.Linear(self.fc_in_full, 512)
        self.fc1_bn = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.fc3_bn = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, number_of_classes)

        self._profile_ratios = {
            "FULL": 1.0,
            "PRUNED": 0.5,
            "MSMALL": 0.25,
        }
        self.active_profile = "FULL"
        self.active_ratio = 1.0

        
        self.d_full = 128

    def set_profile(self, profile: str) -> None:
        if profile is None:
            profile = "FULL"
        profile = str(profile).upper()
        if profile not in self._profile_ratios:
            raise ValueError(f"Unknown profile: {profile}. Supported: {list(self._profile_ratios.keys())}")
        self.active_profile = profile
        self.active_ratio = float(self._profile_ratios[profile])

    def _active_dims(self) -> Dict[str, int]:
        r = float(self.active_ratio)
        c1 = max(1, int(self.out_1 * r))
        c2 = max(1, int(self.out_2 * r))
        c3 = max(1, int(self.out_3 * r))
        h1 = max(1, int(512 * r))
        h2 = max(1, int(256 * r))
        h3 = max(1, int(128 * r))
        return {"c1": c1, "c2": c2, "c3": c3, "h1": h1, "h2": h2, "h3": h3}

    @staticmethod
    def _bn1d_slice(bn: nn.BatchNorm1d, x: torch.Tensor, c: int) -> torch.Tensor:
        
        bs = int(x.size(0))
        training_flag = bool(bn.training and bs > 1)
        return F.batch_norm(
            x[:, :c, :],
            bn.running_mean[:c],
            bn.running_var[:c],
            bn.weight[:c] if bn.weight is not None else None,
            bn.bias[:c] if bn.bias is not None else None,
            training_flag,
            bn.momentum,
            bn.eps,
        )

    @staticmethod
    def _bn1d_vec_slice(bn: nn.BatchNorm1d, x: torch.Tensor, d: int) -> torch.Tensor:
        
        bs = int(x.size(0))
        training_flag = bool(bn.training and bs > 1)
        return F.batch_norm(
            x[:, :d],
            bn.running_mean[:d],
            bn.running_var[:d],
            bn.weight[:d] if bn.weight is not None else None,
            bn.bias[:d] if bn.bias is not None else None,
            training_flag,
            bn.momentum,
            bn.eps,
        )

    def _forward_features_active(self, x: torch.Tensor) -> torch.Tensor:
        dims = self._active_dims()
        c1, c2, c3 = dims["c1"], dims["c2"], dims["c3"]
        h1, h2, h3 = dims["h1"], dims["h2"], dims["h3"]

        
        w1 = self.cnn1.weight[:c1, : self.num_features, :]
        b1 = self.cnn1.bias[:c1] if self.cnn1.bias is not None else None
        x = F.conv1d(x, w1, b1, stride=self.cnn1.stride, padding=self.cnn1.padding, dilation=self.cnn1.dilation, groups=1)
        x = self._bn1d_slice(self.bn1, x, c1)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop_conv(x)

        
        w2 = self.cnn2.weight[:c2, :c1, :]
        b2 = self.cnn2.bias[:c2] if self.cnn2.bias is not None else None
        x = F.conv1d(x[:, :c1, :], w2, b2, stride=self.cnn2.stride, padding=self.cnn2.padding, dilation=self.cnn2.dilation, groups=1)
        x = self._bn1d_slice(self.bn2, x, c2)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop_conv(x)

        
        w3 = self.cnn3.weight[:c3, :c2, :]
        b3 = self.cnn3.bias[:c3] if self.cnn3.bias is not None else None
        x = F.conv1d(x[:, :c2, :], w3, b3, stride=self.cnn3.stride, padding=self.cnn3.padding, dilation=self.cnn3.dilation, groups=1)
        x = self._bn1d_slice(self.bn3, x, c3)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop_conv(x)

        
        x = x[:, :c3, :]
        x = x.reshape(x.size(0), -1)

        L = self.window_size // 8
        fc_in = int(c3 * L)

        
        x = F.linear(x[:, :fc_in], self.fc1.weight[:h1, :fc_in], self.fc1.bias[:h1])
        x = self._bn1d_vec_slice(self.fc1_bn, x, h1)
        x = F.relu(self.drop(x))

        
        x = F.linear(x[:, :h1], self.fc2.weight[:h2, :h1], self.fc2.bias[:h2])
        x = self._bn1d_vec_slice(self.fc2_bn, x, h2)
        x = F.relu(self.drop(x))

        
        x = F.linear(x[:, :h2], self.fc3.weight[:h3, :h2], self.fc3.bias[:h3])
        x = self._bn1d_vec_slice(self.fc3_bn, x, h3)
        x = F.relu(self.drop(x))

        return x[:, :h3]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats_active = self._forward_features_active(x)
        d_act = int(feats_active.shape[1])
        if d_act == self.d_full:
            return feats_active
        if d_act > self.d_full:
            return feats_active[:, : self.d_full]
        pad = self.d_full - d_act
        return torch.cat([feats_active, torch.zeros((feats_active.shape[0], pad), device=feats_active.device)], dim=1)

    @torch.no_grad()
    def export_int8_state_dict(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        



        q_state: Dict[str, torch.Tensor] = {}
        scales: Dict[str, torch.Tensor] = {}

        dims = self._active_dims()
        c1, c2, c3 = dims["c1"], dims["c2"], dims["c3"]
        h1, h2, h3 = dims["h1"], dims["h2"], dims["h3"]

        
        qw, sw = quantize_int8_tensor(self.cnn1.weight[:c1, : self.num_features, :])
        q_state["cnn1.weight"] = qw.cpu(); scales["cnn1.weight"] = sw.cpu()
        if self.cnn1.bias is not None:
            qb, sb = quantize_int8_tensor(self.cnn1.bias[:c1])
            q_state["cnn1.bias"] = qb.cpu(); scales["cnn1.bias"] = sb.cpu()

        for name, t in [("weight", self.bn1.weight), ("bias", self.bn1.bias)]:
            if t is not None:
                qt, st = quantize_int8_tensor(t[:c1])
                q_state[f"bn1.{name}"] = qt.cpu(); scales[f"bn1.{name}"] = st.cpu()

        qw, sw = quantize_int8_tensor(self.cnn2.weight[:c2, :c1, :])
        q_state["cnn2.weight"] = qw.cpu(); scales["cnn2.weight"] = sw.cpu()
        if self.cnn2.bias is not None:
            qb, sb = quantize_int8_tensor(self.cnn2.bias[:c2])
            q_state["cnn2.bias"] = qb.cpu(); scales["cnn2.bias"] = sb.cpu()

        for name, t in [("weight", self.bn2.weight), ("bias", self.bn2.bias)]:
            if t is not None:
                qt, st = quantize_int8_tensor(t[:c2])
                q_state[f"bn2.{name}"] = qt.cpu(); scales[f"bn2.{name}"] = st.cpu()

        qw, sw = quantize_int8_tensor(self.cnn3.weight[:c3, :c2, :])
        q_state["cnn3.weight"] = qw.cpu(); scales["cnn3.weight"] = sw.cpu()
        if self.cnn3.bias is not None:
            qb, sb = quantize_int8_tensor(self.cnn3.bias[:c3])
            q_state["cnn3.bias"] = qb.cpu(); scales["cnn3.bias"] = sb.cpu()

        for name, t in [("weight", self.bn3.weight), ("bias", self.bn3.bias)]:
            if t is not None:
                qt, st = quantize_int8_tensor(t[:c3])
                q_state[f"bn3.{name}"] = qt.cpu(); scales[f"bn3.{name}"] = st.cpu()

        
        L = self.window_size // 8
        fc_in = int(c3 * L)

        qw, sw = quantize_int8_tensor(self.fc1.weight[:h1, :fc_in])
        qb, sb = quantize_int8_tensor(self.fc1.bias[:h1])
        q_state["fc1.weight"] = qw.cpu(); scales["fc1.weight"] = sw.cpu()
        q_state["fc1.bias"] = qb.cpu(); scales["fc1.bias"] = sb.cpu()
        for name, t in [("weight", self.fc1_bn.weight), ("bias", self.fc1_bn.bias)]:
            if t is not None:
                qt, st = quantize_int8_tensor(t[:h1])
                q_state[f"fc1_bn.{name}"] = qt.cpu(); scales[f"fc1_bn.{name}"] = st.cpu()

        qw, sw = quantize_int8_tensor(self.fc2.weight[:h2, :h1])
        qb, sb = quantize_int8_tensor(self.fc2.bias[:h2])
        q_state["fc2.weight"] = qw.cpu(); scales["fc2.weight"] = sw.cpu()
        q_state["fc2.bias"] = qb.cpu(); scales["fc2.bias"] = sb.cpu()
        for name, t in [("weight", self.fc2_bn.weight), ("bias", self.fc2_bn.bias)]:
            if t is not None:
                qt, st = quantize_int8_tensor(t[:h2])
                q_state[f"fc2_bn.{name}"] = qt.cpu(); scales[f"fc2_bn.{name}"] = st.cpu()

        qw, sw = quantize_int8_tensor(self.fc3.weight[:h3, :h2])
        qb, sb = quantize_int8_tensor(self.fc3.bias[:h3])
        q_state["fc3.weight"] = qw.cpu(); scales["fc3.weight"] = sw.cpu()
        q_state["fc3.bias"] = qb.cpu(); scales["fc3.bias"] = sb.cpu()
        for name, t in [("weight", self.fc3_bn.weight), ("bias", self.fc3_bn.bias)]:
            if t is not None:
                qt, st = quantize_int8_tensor(t[:h3])
                q_state[f"fc3_bn.{name}"] = qt.cpu(); scales[f"fc3_bn.{name}"] = st.cpu()

        qw, sw = quantize_int8_tensor(self.fc4.weight[:, :h3])
        qb, sb = quantize_int8_tensor(self.fc4.bias)
        q_state["fc4.weight"] = qw.cpu(); scales["fc4.weight"] = sw.cpu()
        q_state["fc4.bias"] = qb.cpu(); scales["fc4.bias"] = sb.cpu()

        return q_state, scales

    @staticmethod
    def estimate_int8_payload_bytes(q_state: Dict[str, torch.Tensor], scales: Dict[str, torch.Tensor]) -> int:
        n_int8 = 0
        for v in q_state.values():
            if torch.is_tensor(v):
                n_int8 += int(v.numel())
        n_scales = len(scales)
        return int(n_int8 * 1 + n_scales * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats_active = self._forward_features_active(x)
        h3 = int(self._active_dims()["h3"])

        w_out = self.fc4.weight[:, :h3]
        b_out = self.fc4.bias

        if self.active_profile == "MSMALL":
            w_out = fake_int8_quant(w_out)
            feats_in = fake_int8_quant(feats_active)
        else:
            feats_in = feats_active

        logits = F.linear(feats_in[:, :h3], w_out, b_out)
        return logits


import copy
from dataclasses import dataclass,field
from enum import IntEnum, Enum
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm


class Client:
    
















    def __init__(
        self,
        model,
        dataloader,
        mps_device,
        profile: Optional[str] = None,
        global_prototypes: Optional[Dict[int, torch.Tensor]] = None,
        lambda_pa: float = 0.0,
    ):
        self.device = mps_device
        self.model = copy.deepcopy(model)

        
        self.profile: Optional[str] = None
        if profile is not None and hasattr(self.model, "set_profile"):
            try:
                self.model.set_profile(profile)
                self.profile = str(profile).upper()
            except Exception:
                self.profile = None

        self.dataloader = dataloader
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.5)
        self.criterion = nn.CrossEntropyLoss()

        
        self.global_prototypes: Optional[Dict[int, torch.Tensor]] = global_prototypes
        self.lambda_pa: float = float(lambda_pa)

        
        self.proto_dict: Dict[int, torch.Tensor] = {}
        self.count_dict: Dict[int, int] = {}

        
        self.int8_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.int8_scales: Optional[Dict[str, torch.Tensor]] = None
        self.int8_num_bytes: int = 0

        
        self.samples_seen: int = 0

    def _prototype_alignment_loss(
        self,
        feats: torch.Tensor,          
        targets: torch.Tensor,        
        global_prototypes: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        if feats.numel() == 0:
            return feats.sum() * 0.0

        proto_list = []
        valid_mask = []
        for y in targets.tolist():
            if int(y) in global_prototypes:
                proto_list.append(global_prototypes[int(y)].to(feats.device).view(1, -1))
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not any(valid_mask):
            return feats.sum() * 0.0

        feats_valid = feats[torch.tensor(valid_mask, device=feats.device)]
        protos = torch.cat(proto_list, dim=0)

        dist_sq = torch.sum((feats_valid - protos) ** 2, dim=1)
        return (0.5 * dist_sq).mean()

    @torch.no_grad()
    def compute_prototypes(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
        


        if not hasattr(self.model, "extract_features"):
            return {}, {}

        self.model.eval()
        sums: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}

        for data, target in self.dataloader:
            data = data.to(self.device)
            target = target.to(self.device)

            feats = self.model.extract_features(data)
            for i in range(feats.shape[0]):
                y = int(target[i].item())
                f = feats[i].detach()
                if y not in sums:
                    sums[y] = f.clone()
                    counts[y] = 1
                else:
                    sums[y] += f
                    counts[y] += 1

        proto: Dict[int, torch.Tensor] = {}
        for y, s in sums.items():
            n = max(1, int(counts.get(y, 0)))
            proto[y] = (s / float(n)).detach().cpu()

        return proto, counts

    def train(self, epochs=1):
        self.model.train()
        self.samples_seen = 0

        for _ in range(epochs):
            for data, target in tqdm(self.dataloader):
                self.optimizer.zero_grad()

                data = data.to(self.device)
                target = target.to(self.device)
                self.samples_seen += int(target.shape[0])

                logits = self.model(data)
                loss = self.criterion(logits, target)

                
                if (
                    self.lambda_pa > 0.0
                    and self.global_prototypes is not None
                    and hasattr(self.model, "extract_features")
                    and (self.profile is not None and self.profile.upper() in ("MSMALL",))
                ):
                    feats = self.model.extract_features(data)
                    pa = self._prototype_alignment_loss(feats, target, self.global_prototypes)
                    loss = loss + self.lambda_pa * pa

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                self.optimizer.step()

        self.proto_dict, self.count_dict = self.compute_prototypes()

        
        if hasattr(self.model, "export_int8_state_dict") and (self.profile is not None and self.profile.upper() in ("MSMALL",)):
            try:
                q_state, scales = self.model.export_int8_state_dict()
                self.int8_state_dict = q_state
                self.int8_scales = scales
                if hasattr(self.model, "estimate_int8_payload_bytes"):
                    self.int8_num_bytes = int(self.model.estimate_int8_payload_bytes(q_state, scales))
            except Exception:
                self.int8_state_dict, self.int8_scales, self.int8_num_bytes = None, None, 0

        return self.model






class UserState(IntEnum):
    IDLE = 0
    ACTIVE = 1
    CHARGING = 2


class ActivityLabel(str, Enum):
    SLEEP = "sleep"
    COMMUTE = "commute"
    EXERCISE = "exercise"
    WORK = "work"
    IDLE = "idle"   


@dataclass
class HardwareEnergyModel:
    battery_capacity_mAh: float = 300.0
    battery_voltage_V: float = 3.7
    energy_per_flop_J: float = 1.2e-9
    tx_power_W: float = 60e-3
    ble_rate_bps: float = 1e6  

    def battery_energy_J(self) -> float:
        return (self.battery_capacity_mAh / 1000.0) * self.battery_voltage_V * 3600.0

    def compute_energy_J(self, flops: float) -> float:
        return float(flops) * float(self.energy_per_flop_J)

    def comm_energy_J(self, bits: float) -> float:
        t = float(bits) / float(self.ble_rate_bps)
        return float(self.tx_power_W) * t


@dataclass
class EnergyParams:
    base_power_W: Dict[ActivityLabel, float]
    charge_power_W: float

    @staticmethod
    def default() -> "EnergyParams":
        return EnergyParams(
            base_power_W={
                ActivityLabel.SLEEP: 3e-3,     
                ActivityLabel.WORK: 8e-3,      
                ActivityLabel.COMMUTE: 12e-3,  
                ActivityLabel.EXERCISE: 12e-3, 
                ActivityLabel.IDLE: 5e-3,      
            },
            charge_power_W=0.5,  
        )


@dataclass
class SyntheticUserBehaviorTraces:
    rng: np.random.Generator
    day_topup_prob: float = 0.25
    topup_min_hours: int = 1
    topup_max_hours: int = 2

    topup_start: Optional[int] = None
    topup_end: Optional[int] = None

    def __post_init__(self):
        if self.rng.uniform() < self.day_topup_prob:
            start = int(self.rng.integers(10, 18))
            dur = int(self.rng.integers(self.topup_min_hours, self.topup_max_hours + 1))
            self.topup_start = start
            self.topup_end = min(23, start + dur)

    @staticmethod
    def _is_night_charging(hour: int) -> bool:
        return (hour >= 22) or (hour <= 7)

    def is_charging(self, hour: int) -> bool:
        if self._is_night_charging(hour):
            return True
        if self.topup_start is not None and self.topup_end is not None:
            return self.topup_start <= hour <= self.topup_end
        return False

    def activity_at(self, hour: int) -> ActivityLabel:
        if 0 <= hour <= 7:
            return ActivityLabel.SLEEP
        if 7 < hour <= 9:
            return ActivityLabel.COMMUTE
        if 9 < hour <= 12:
            return ActivityLabel.WORK
        if 12 < hour <= 13:
            return ActivityLabel.IDLE
        if 13 < hour <= 17:
            return ActivityLabel.WORK
        if 17 < hour <= 19:
            return ActivityLabel.COMMUTE
        if 19 < hour <= 20:
            return ActivityLabel.EXERCISE
        return ActivityLabel.WORK


class ModelProfile(str, Enum):
    FULL = "FULL"
    PRUNED = "PRUNED"
    MSMALL = "MSMALL"


@dataclass
class ProfileLookupParams:
    eas_high: float = 0.35
    eas_mid: float = 0.20
    cpu_low: float = 0.35


def lookup_profile(
    eas_value: float,
    cpu_util: float,
    s_t: UserState,
    params: ProfileLookupParams = ProfileLookupParams(),
) -> ModelProfile:
    if s_t == UserState.CHARGING:
        return ModelProfile.FULL

    if eas_value >= params.eas_high and cpu_util <= params.cpu_low:
        return ModelProfile.FULL
    if eas_value >= params.eas_mid:
        return ModelProfile.PRUNED

    return ModelProfile.MSMALL


class BACSDecisionEngine:
    def __init__(
        self,
        state_transition_P: np.ndarray,
        energy_params: EnergyParams,
        tau: int = 6,
        theta_eas: float = 0.0,   
        B_min: float = 0.15,      
        alpha_eas: float = 0.5,
        beta_eas: float = 0.5,
        consumption_mode: str = "mean",
        rng: Optional[np.random.Generator] = None,
        hw: Optional[HardwareEnergyModel] = None,
        step_seconds: int = 3600,
        pred_error: float = 0.03,   
    ):
        assert state_transition_P.shape == (3, 3), "P must be 3x3 for {Idle, Active, Charging}"
        assert consumption_mode in ("mean", "last"), "consumption_mode must be 'mean' or 'last'"

        self.P = state_transition_P.astype(np.float64)
        self.energy = energy_params
        self.tau = int(tau)
        self.theta_eas = float(theta_eas)
        self.B_min = float(B_min)
        self.alpha_eas = float(alpha_eas)
        self.beta_eas = float(beta_eas)
        self.consumption_mode = consumption_mode
        self.rng = rng if rng is not None else np.random.default_rng()

        self.hw = hw if hw is not None else HardwareEnergyModel()
        self.step_seconds = int(step_seconds)

        self.pred_error = float(pred_error)
        self._E_batt = float(self.hw.battery_energy_J())

    def _state_dist_k(self, s_t: UserState, k: int) -> np.ndarray:
        Pk = np.linalg.matrix_power(self.P, k)
        return Pk[int(s_t), :]

    def expected_gain_and_consumption_series(self, s_t: UserState) -> Tuple[float, List[float]]:
        gain_sum_frac = 0.0
        cons_list_frac: List[float] = []

        p_idle_W = float(self.energy.base_power_W.get(ActivityLabel.WORK, 8e-3))
        p_active_W = float(self.energy.base_power_W.get(ActivityLabel.COMMUTE, 12e-3))
        p_chg_base_W = float(self.energy.base_power_W.get(ActivityLabel.IDLE, 5e-3))
        p_charge_W = float(self.energy.charge_power_W)

        for k in range(1, self.tau + 1):
            dist = self._state_dist_k(s_t, k)

            exp_base_W = (
                dist[UserState.IDLE] * p_idle_W
                + dist[UserState.ACTIVE] * p_active_W
                + dist[UserState.CHARGING] * p_chg_base_W
            )
            base_E_J = float(exp_base_W) * float(self.step_seconds)
            cons_list_frac.append(base_E_J / self._E_batt)

            charge_E_J = float(dist[UserState.CHARGING]) * float(p_charge_W) * float(self.step_seconds)
            gain_sum_frac += (charge_E_J / self._E_batt)

        return float(gain_sum_frac), cons_list_frac

    def eas(self, B_cur: float, s_t: UserState) -> float:
        gain_sum, cons_list = self.expected_gain_and_consumption_series(s_t)
        cons_term = float(sum(cons_list)) if cons_list else 0.0
        return float(B_cur + self.alpha_eas * gain_sum - self.beta_eas * cons_term)

    def availability_flag(self, B_cur: float, s_t: UserState) -> int:
        if s_t == UserState.CHARGING:
            return 1
        if float(B_cur) < self.B_min:
            return 0
        return 1 if self.eas(B_cur, s_t) >= self.theta_eas else 0

    
    def availability_flag_formula8(self, B_cur: float, s_t: UserState, fl_energy_J: float) -> int:
        if s_t == UserState.CHARGING:
            return 1

        B_cur = float(B_cur)
        if B_cur <= 0.0:
            return 0

        fl_cost_frac = float(fl_energy_J) / float(self._E_batt) if float(self._E_batt) > 0 else 1.0
        theta_t = float(self.B_min) + float(self.pred_error) + float(fl_cost_frac)

        eas_val = float(self.eas(B_cur, s_t))
        return 1 if eas_val >= theta_t else 0

    def sample_next_state(self, s_t: UserState) -> UserState:
        probs = self.P[int(s_t), :]
        nxt = int(self.rng.choice([0, 1, 2], p=probs))
        return UserState(nxt)

    def step_battery(
        self,
        B_t: float,
        s_t: UserState,
        activity: ActivityLabel,
        fl_selected: bool,
        fl_energy_J: float,
        charging: bool,
    ) -> float:
        B_next = float(B_t)

        p_base = float(self.energy.base_power_W.get(activity, self.energy.base_power_W.get(ActivityLabel.IDLE, 5e-3)))
        if activity == ActivityLabel.EXERCISE:
            p_base = p_base * 3.0

        base_E = p_base * float(self.step_seconds)
        B_next -= base_E / self._E_batt

        if charging:
            charge_E = float(self.energy.charge_power_W) * float(self.step_seconds)
            B_next += charge_E / self._E_batt

        if fl_selected and fl_energy_J > 0.0:
            B_next -= float(fl_energy_J) / self._E_batt

        return float(np.clip(B_next, 0.0, 1.0))



@dataclass
class WearableClientInfo:
    loader: Any
    battery: float
    state: UserState
    bacs: BACSDecisionEngine
    cpu_util: float = 0.0
    
    profile_params: ProfileLookupParams = field(default_factory=ProfileLookupParams)

    trace: Optional[SyntheticUserBehaviorTraces] = None
    hour_of_day: int = 0
    activity: ActivityLabel = ActivityLabel.IDLE

    def __post_init__(self):
        if self.trace is None:
            self.trace = SyntheticUserBehaviorTraces(rng=self.bacs.rng)
        self.hour_of_day = int(self.bacs.rng.integers(0, 24))
        self._refresh_activity_and_state()

    def _refresh_activity_and_state(self):
        assert self.trace is not None
        self.activity = self.trace.activity_at(self.hour_of_day)
        charging = self.trace.is_charging(self.hour_of_day)

        if charging:
            self.state = UserState.CHARGING
        else:
            if self.activity in (ActivityLabel.COMMUTE, ActivityLabel.EXERCISE):
                self.state = UserState.ACTIVE
            else:
                self.state = UserState.IDLE

        self.cpu_util = self._sample_cpu_util_by_activity(self.activity, charging)

    def _sample_cpu_util_by_activity(self, activity: ActivityLabel, charging: bool) -> float:
        if charging:
            return float(self.bacs.rng.uniform(0.0, 0.25))
        if activity == ActivityLabel.EXERCISE:
            return float(self.bacs.rng.uniform(0.6, 0.95))
        if activity == ActivityLabel.COMMUTE:
            return float(self.bacs.rng.uniform(0.4, 0.8))
        if activity == ActivityLabel.WORK:
            return float(self.bacs.rng.uniform(0.1, 0.4))
        if activity == ActivityLabel.SLEEP:
            return float(self.bacs.rng.uniform(0.0, 0.15))
        return float(self.bacs.rng.uniform(0.05, 0.3))

    def availability(self) -> int:
        return self.bacs.availability_flag(self.battery, self.state)

    def select_profile(self) -> ModelProfile:
        eas_val = self.bacs.eas(self.battery, self.state)
        return lookup_profile(eas_val, float(self.cpu_util), self.state, self.profile_params)

    def env_step(self, fl_selected: bool, fl_energy_J: float) -> None:
        assert self.trace is not None
        charging = self.trace.is_charging(self.hour_of_day)

        self.battery = self.bacs.step_battery(
            B_t=self.battery,
            s_t=self.state,
            activity=self.activity,
            fl_selected=fl_selected,
            fl_energy_J=float(fl_energy_J),
            charging=charging,
        )

        self.hour_of_day = int((self.hour_of_day + 1) % 24)
        self._refresh_activity_and_state()

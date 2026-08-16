"""记忆公式纯函数（无模型、无 IO，可直接单元测试）。

S(m) = α·R(m,q) + β·T(m) + γ·F(m) + δ
  R(m,q): 语义相关性（检索余弦相似度，取正部分）
  T(m)  : exp(-(Δt/τ)^c)              —— 艾宾浩斯时间衰减
  F(m)  : clamp(w0 + w1·Σ(v_i·I_type(i)) + w2·(1-exp(-refresh·k)), 0, 1)  —— 固有重要性
  M(Δt) : 1 - exp(-(Δt/τ_m)^c_m)      —— 记忆成熟度（切片触发用）
"""
import math

from .config import TYPE_SCORE_MAP


def clamp(value: float, min_val: float, max_val: float) -> float:
    if min_val >= max_val:
        raise ValueError(f'不允许出现min_val >= max_val的情况')
    return max(min_val, min(value, max_val))


def maturity(dt: float, tau_m: float, c_m: float) -> float:
    """记忆成熟度 M(Δt)=1-exp(-(Δt/τ_m)^c_m)：Δt 越长越接近 1"""
    return 1 - math.exp(-((dt / tau_m) ** c_m))


def time_decay(dt: float, tau: float, c: float) -> float:
    """时间衰减 T(m)=exp(-(Δt/τ)^c)：dt=0 时为 1，越久越接近 0"""
    return math.exp(-((dt / tau) ** c))


def importance(w0: float, w1: float, w2: float, type_score_sum: float,
               refresh_count: float, k: float = 0.5) -> float:
    """固有重要性 F(m)（clamp 到 [0,1] 防止权重溢出）"""
    return clamp(
        w0 + w1 * type_score_sum + w2 * (1 - math.exp(-refresh_count * k)), 0.0, 1.0)


def score(alpha: float, r: float, beta: float, t: float, gamma: float,
          f: float, delta: float) -> float:
    """记忆片段综合得分 S(m)"""
    return alpha * r + beta * t + gamma * f + delta


def type_score_summary(types: list[str], type_score_map: dict[str, float] | None = None) -> float:
    """Σ(v_i·I_type(i))：片段类型重要性的总和（未知名类型按 0 计）"""
    m = type_score_map or TYPE_SCORE_MAP
    return sum(m.get(t, 0.0) * 1.0 for t in types)


def validate_param_constraints(params) -> None:
    """校验 α+β+γ=1、w0+w1+w2=1（不满足时抛 ValueError）"""
    w_sum = params.w0 + params.w1 + params.w2
    abg_sum = params.alpha + params.beta + params.gamma
    if abs(w_sum - 1.0) > 1e-6:
        raise ValueError(f'w0+w1+w2 必须等于 1，当前为 {w_sum}')
    if abs(abg_sum - 1.0) > 1e-6:
        raise ValueError(f'α+β+γ 必须等于 1，当前为 {abg_sum}')

"""记忆公式纯函数（无模型、无 IO，可直接单元测试）。

S(m) = α·R(m,q) + β·T(m) + γ·F(m) + δ
  R(m,q): 语义相关性——**候选内相对刻度**（见 relevance_scores）
  T(m)  : exp(-(Δt/τ)^c)              —— 艾宾浩斯时间衰减
  F(m)  : clamp(w0 + w1·u + w2·(1-exp(-refresh·k)), 0, 1)  —— 固有重要性
          u = 1 - Π(1 - v_i)          —— 类型分用**概率并集**合成（见 type_score_union）
  M(Δt) : 1 - exp(-(Δt/τ_m)^c_m)      —— 记忆成熟度（切片触发用）

2026-10-05 两处标定调整（依据：`benchmarks/analyze_scoring_log.py` 对真跑打分表的复算，
详见项目根目录《记忆中间件修改记录_2026-10-05.md》§9 与下面对每个函数的说明）：
  ① R 由绝对值 `max(0, cos − 0.5)` 改为**候选内相对刻度**——实测 αR 的加权极差只有 0~0.119
     （10 次注入事件里 3 次全候选为 0），而 LLM 却有 8/9 次把最大权重给 α：最大值给了最不区分候选的一维。
  ② F 的类型分由 Σ 改为概率并集——实测 Σ 无上界（identity+preference=1.75、再加 decision=2.4），
     乘 w1 后撞 clamp：6 个候选里 5 个 F 精确等于 1.000，类型重要性退化成二值，
     且巩固项（str 0→6）被 clamp 吃掉、完全不可见。
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
    """Σ(v_i·I_type(i))：片段类型重要性的总和（未知名类型按 0 计）。

    ⚠️ 已被 `type_score_union` 取代（F 的类型分合成改用并集）：本函数无上界，喂给
    `importance()` 后会被 clamp 砍平。保留仅为向后兼容与对照实验，不要再用于打分。
    """
    m = type_score_map or TYPE_SCORE_MAP
    return sum(m.get(t, 0.0) * 1.0 for t in types)


def type_score_union(types: list[str], type_score_map: dict[str, float] | None = None) -> float:
    """多类型重要度合成：概率并集 `u = 1 − Π(1 − v_i)`（未知名类型按 0 计，不影响结果）。

    为什么不用求和（原做法）：Σ 无上界——identity 0.95 + preference 0.80 = 1.75、
    再加 decision 0.85 = 2.6；乘 w1（LLM 调到 0.75）后 `w0 + w1·Σ ≥ 1.41`，会被 clamp 砍成 1.000。
    实测后果（2026-10-05 真跑，见修改记录 §9）：6 个候选里 5 个 F **精确等于 1.000**——
    类型重要性退化成"是否含 identity/preference"的二值特征，且 `w2·(1−e^(−str/2))` 这一项
    被 clamp 吃掉：某个片段巩固次数 0→6，F 全程 1.000，**巩固完全不可见**。
    想靠调小 w1 绕开也不行：要让 1.75 不撞顶需 w1 ≤ 0.51、要让 2.4 不撞顶需 w1 ≤ 0.375——
    类型越多要求权重越小，等于把类型分整体压扁；而且类型数量在原理上没有上界。

    为什么是并集（对照其他常见做法，四条要求：有界 / 越多越大 / 不饱和 / 可解释）：
      - 求和 + clamp：不饱和 ✗（现状）
      - 取平均：违反"越多越大"且**方向相反**——identity+preference 均值 0.875 < identity 单独 0.95，
        多标一个高价值类型反而降分
      - 取最大：多类型完全不叠加，identity+preference 与 identity 同分，类型标注白做
      - tanh / Σ/(1+Σ)：也有界单调，但数值要靠拍参数、解释性差
      - **概率并集**：有界 [0,1)、严格递增、边际递减（第二个类型加得比第一个少），
        可解释为"至少命中一个类型的置信度"；配合 w0+w1+w2=1 使 F 天然落在 [0,1]，无需 clamp。

    注意：`importance()` 里的 clamp 保留作为兜底（LLM 给出的 w 理论上满足和为 1，但不保证）。
    """
    m = type_score_map or TYPE_SCORE_MAP
    u = 1.0
    for t in types:
        u *= (1.0 - m.get(t, 0.0))
    return 1.0 - u


def relevance_scores(cosines: list[float]) -> list[float]:
    """把一批候选片段的余弦，转成 [0,1] 的语义相关度 R（**候选内相对刻度**）。

    做法：以本批余弦的**中位数**为锚——低于中位数的记 0，最高者为 1，中间线性插值：
        R = max(0, (cos − median) / (max − median))
    锚点选中位数而不是最小值，是为了保住原来那句 `cos − 0.5` 的**意图**：
    "只有明显更相关的候选才算数"（因此仍有一半候选在这一维得 0），
    同时让够格的候选之间拉开 0~1 的差距，α 的权重才有落点。

    为什么不继续用绝对值 `max(0, cos − 0.5)`：
      · 绝对值不可跨主题/跨嵌入模型比较——换一个主题词或换一个嵌入模型，整批余弦会整体平移，
        0.5 这个数就不再"对应高相关"。
      · 而 top-K 注入只关心**同一批候选内部的相对次序**，相对刻度与目的对齐，且不必拍任何数字。
      · 实测（10 次注入事件）：旧做法下 αR 的加权极差仅 0~0.119（3 次全候选为 0），
        而 LLM 8/9 次把最大权重给 α —— 最大的权重落在最不区分候选的一维上。

    为什么不用 Z-score（同样能把这一维"救活"）：
      · **同一批候选内，Z-score 与本函数的排序完全一致**（都是同一列的单调变换），
        因此 top-K 名单一模一样——它不是"更准"，只是另一种刻度。
      · 差别在语义：Z-score 有正有负、无上界，会把 S 推出 [0,1]，破坏 α+β+γ=1 的权重语义
        （δ 也要重新标定）；本函数保 [0,1]，S 的取值区间与含义不变。
      · 若更看重"保留候选间距离结构/离群点"，可换 Z-score，但需同时接受上面那条语义变化。

    退化情形：整批余弦相同（`max == median`）→ 全部记 0，表示"这批候选在这一维没有区分度"，
    不做除零、不制造虚假差异。已知取舍：候选余弦挤得很近时（如全在 0.50~0.52），
    本函数会把这点微小差异放大到满量程——验收时需同时看注入体积与召回是否退化。
    """
    if not cosines:
        return []
    ordered = sorted(cosines)
    anchor = ordered[len(ordered) // 2]
    span = ordered[-1] - anchor
    if span <= 1e-9:
        return [0.0 for _ in cosines]
    return [max(0.0, (c - anchor) / span) for c in cosines]


def validate_param_constraints(params) -> None:
    """校验 α+β+γ=1、w0+w1+w2=1（不满足时抛 ValueError）"""
    w_sum = params.w0 + params.w1 + params.w2
    abg_sum = params.alpha + params.beta + params.gamma
    if abs(w_sum - 1.0) > 1e-6:
        raise ValueError(f'w0+w1+w2 必须等于 1，当前为 {w_sum}')
    if abs(abg_sum - 1.0) > 1e-6:
        raise ValueError(f'α+β+γ 必须等于 1，当前为 {abg_sum}')

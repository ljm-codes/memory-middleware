# -*- coding: utf-8 -*-
"""Layer 0：记忆公式纯函数测试"""
import math

import pytest

from memory_middleware.config import TYPE_SCORE_MAP
from memory_middleware.formula import (
    clamp,
    importance,
    maturity,
    relevance_scores,
    score,
    time_decay,
    type_score_summary,
    type_score_union,
    validate_param_constraints,
)
from memory_middleware.models import TimeMemoryFormulaParam


class TestClamp:
    def test_clamp_inside_range(self):
        assert clamp(0.5, 0, 1) == 0.5

    def test_clamp_above_max(self):
        assert clamp(1.5, 0, 1) == 1.0

    def test_clamp_below_min(self):
        assert clamp(-0.5, 0, 1) == 0.0

    def test_clamp_invalid_bounds(self):
        with pytest.raises(ValueError):
            clamp(0.5, 1, 0)


class TestMaturity:
    def test_zero_dt(self):
        # Δt=0 → M=0
        assert maturity(0, 600, 0.5) == 0.0

    def test_bounds(self):
        # M 恒在 [0, 1)
        assert 0 <= maturity(10_000, 600, 0.5) < 1.0

    def test_monotonic(self):
        # 时间越久成熟度越高
        assert maturity(500, 600, 0.5) < maturity(1000, 600, 0.5) < maturity(5000, 600, 0.5)

    def test_tau_scales_time(self):
        # τ 越大，同等 Δt 成熟度越低
        assert maturity(600, 600, 0.5) > maturity(600, 6000, 0.5)


class TestTimeDecay:
    def test_zero_dt_is_one(self):
        assert time_decay(0, 43200, 0.5) == 1.0

    def test_decays_to_zero(self):
        assert time_decay(10 ** 9, 43200, 0.5) < 1e-9

    def test_bounds_and_monotonic(self):
        assert 0 <= time_decay(1000, 43200, 0.5) <= 1
        assert time_decay(100, 43200, 0.5) > time_decay(1000, 43200, 0.5)


class TestImportance:
    def test_clamped_to_unit(self):
        # 极端权重也不越界
        assert importance(0.15, 0.8, 0.05, 1.0, 10 ** 6) == 1.0
        assert importance(-1, -1, -1, 0, 0) == 0.0

    def test_increases_with_refresh(self):
        # 巩固次数越多，F 越高
        base = importance(0.1, 0.8, 0.1, 0.5, 0)
        strengthened = importance(0.1, 0.8, 0.1, 0.5, 5)
        assert strengthened > base

    def test_increases_with_type_score(self):
        assert importance(0.1, 0.8, 0.1, 0.9, 0) > importance(0.1, 0.8, 0.1, 0.1, 0)


class TestScore:
    def test_linear_combination(self):
        assert score(0.33, 1.0, 0.33, 0.5, 0.34, 0.0, 0) == pytest.approx(0.33 + 0.165, abs=1e-9)

    def test_delta_offset(self):
        assert score(1, 0, 0, 0, 0, 0, 0.1) == pytest.approx(0.1)


class TestTypeScoreSummary:
    def test_known_types(self):
        assert type_score_summary(['identity']) == TYPE_SCORE_MAP['identity']

    def test_multiple_types_sum(self):
        s = type_score_summary(['identity', 'chat'])
        assert s == pytest.approx(0.95 + 0.15)

    def test_unknown_type_ignored(self):
        assert type_score_summary(['不存在的类型']) == 0.0

    def test_type_score_map_values_in_unit_interval(self):
        assert all(0 < v <= 1 for v in TYPE_SCORE_MAP.values())


class TestParamConstraints:
    def test_valid_params(self):
        validate_param_constraints(TimeMemoryFormulaParam())  # 默认值 0.33+0.33+0.34=1

    def test_w_sum_must_be_one(self):
        p = TimeMemoryFormulaParam(w0=0.5, w1=0.5, w2=0.5)
        with pytest.raises(ValueError):
            validate_param_constraints(p)

    def test_alpha_beta_gamma_must_be_one(self):
        p = TimeMemoryFormulaParam(alpha=1.0, beta=1.0, gamma=1.0)
        with pytest.raises(ValueError):
            validate_param_constraints(p)


class TestMathEbbinghausReference:
    def test_tau_is_half_life_scale(self):
        # τ 时间尺度下衰减到 ~37%（1/e）——艾宾浩斯曲线的定义点
        assert time_decay(43200, 43200, 1.0) == pytest.approx(math.e ** -1, rel=1e-6)


class TestTypeScoreUnion:
    """类型分合成由 Σ 改为概率并集（见 formula.type_score_union 的取舍说明）"""

    def test_identity_plus_preference_is_0_99(self):
        assert type_score_union(['identity', 'preference']) == pytest.approx(1 - 0.05 * 0.20)

    def test_bounded_never_reaches_one(self):
        """有界：所有已知类型全标上也不撞 1 —— 这正是取代 Σ 的理由"""
        u = type_score_union(list(TYPE_SCORE_MAP))
        assert 0 < u < 1

    def test_monotone_increasing(self):
        a = type_score_union(['identity'])
        b = type_score_union(['identity', 'preference'])
        c = type_score_union(['identity', 'preference', 'fact'])
        assert a < b < c

    def test_marginal_diminishing(self):
        """边际递减：第二个类型的增量小于第一个（信息冗余）"""
        first = type_score_union(['identity']) - type_score_union([])
        second = type_score_union(['identity', 'preference']) - type_score_union(['identity'])
        assert second < first

    def test_unknown_type_ignored(self):
        assert type_score_union(['不存在的类型']) == 0.0
        assert type_score_union(['identity', '不存在的类型']) == pytest.approx(
            type_score_union(['identity']))

    def test_does_not_saturate_importance(self):
        """F 不再撞顶（w0=0.1 w1=0.8 w2=0.1 时 identity+preference 应 < 1）"""
        f0 = importance(0.1, 0.8, 0.1, type_score_union(['identity', 'preference']), 0)
        f6 = importance(0.1, 0.8, 0.1, type_score_union(['identity', 'preference']), 6)
        assert f0 < 1.0, '旧做法这里是 1.000（Σ=1.75 × 0.8 → clamp）'
        assert f6 > f0 + 0.05, '巩固必须可见（旧做法被 clamp 吃掉）'


class TestRelevanceScores:
    """语义相关度：候选内相对刻度（中位数锚定），取代 max(0, cos − 0.5)"""

    def test_range_and_endpoints(self):
        r = relevance_scores([0.50, 0.55, 0.60, 0.62])
        assert all(0.0 <= x <= 1.0 for x in r)
        assert max(r) == 1.0, '最高者应当拿满'
        assert r[-1] == 1.0

    def test_below_median_is_zero(self):
        """保住原意图：中位数以下不再贡献相关度（仍有一半候选得 0）"""
        r = relevance_scores([0.50, 0.51, 0.60, 0.62])
        assert r[0] == 0.0 and r[1] == 0.0

    def test_monotone(self):
        cs = [0.4, 0.5, 0.55, 0.6, 0.7]
        r = relevance_scores(cs)
        assert r == sorted(r), '余弦越大，相关度不得更小'

    def test_degenerate_all_equal_is_zero(self):
        """整批余弦相同 → 本维无区分度，全 0（不除零、不制造虚假差异）"""
        assert relevance_scores([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]

    def test_empty(self):
        assert relevance_scores([]) == []

    def test_top_candidate_still_top_after_transform(self):
        """单调变换不改变排序——所以它与 Z-score 在入选名单上等价（仅刻度不同）"""
        cs = [0.45, 0.52, 0.58, 0.61]
        r = relevance_scores(cs)
        assert cs.index(max(cs)) == r.index(max(r))

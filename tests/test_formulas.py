# -*- coding: utf-8 -*-
"""Layer 0：记忆公式纯函数测试"""
import math

import pytest

from memory_middleware.config import TYPE_SCORE_MAP
from memory_middleware.formula import (
    clamp,
    importance,
    maturity,
    score,
    time_decay,
    type_score_summary,
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

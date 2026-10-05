# -*- coding: utf-8 -*-
"""Layer 0：增量画像合并（merge_user_profile）测试"""
from helpers import DictEmbeddings, ExplodingEmbeddings, SYNONYM_MAP
from memory_middleware.profile import field_default_value, merge_user_profile


class TestListMerge:
    def test_dedup_new_first(self):
        merged = merge_user_profile({'user_hobby': ['游泳']}, {'user_hobby': ['游泳', '读书']})
        assert merged['user_hobby'] == ['游泳', '读书']

    def test_empty_old(self):
        merged = merge_user_profile({}, {'user_hobby': ['读书']})
        assert merged['user_hobby'] == ['读书']

    def test_old_items_kept(self):
        merged = merge_user_profile({'user_hobby': ['游泳']}, {'user_hobby': []})
        assert merged['user_hobby'] == ['游泳']


class TestDictMerge:
    def test_recursive_merge_new_wins(self):
        old = {'key_relationships': {'a': '朋友'}}
        new = {'key_relationships': {'b': '同事'}}
        merged = merge_user_profile(old, new)
        assert merged['key_relationships'] == {'a': '朋友', 'b': '同事'}


class TestScalarMerge:
    def test_non_empty_overrides(self):
        merged = merge_user_profile({'user_name': '旧名'}, {'user_name': '新名'})
        assert merged['user_name'] == '新名'

    def test_empty_keeps_old(self):
        # 增量总结未提及的字段（空值/默认值）不得覆盖旧画像
        merged = merge_user_profile({'user_name': '旧名'}, {'user_name': ''})
        assert merged['user_name'] == '旧名'

    def test_default_value_keeps_old(self):
        # '保密' 是 user_sex 的默认值：新总结输出默认值视为"未提及"
        merged = merge_user_profile({'user_sex': '女'}, {'user_sex': '保密'})
        assert merged['user_sex'] == '女'

    def test_none_keeps_old(self):
        merged = merge_user_profile({'user_occupation': '工程师'}, {'user_occupation': None})
        assert merged['user_occupation'] == '工程师'


class TestExtras:
    def test_non_profile_metadata_preserved(self):
        # last_summarized_id 等归纳游标元数据不属于 UserProfile 字段，但必须保留
        merged = merge_user_profile({'last_summarized_id': 5}, {'user_name': '张三'})
        assert merged['last_summarized_id'] == 5
        assert merged['user_name'] == '张三'

    def test_does_not_mutate_inputs(self):
        old = {'user_hobby': ['游泳']}
        new = {'user_hobby': ['读书']}
        merge_user_profile(old, new)
        assert old['user_hobby'] == ['游泳']
        assert new['user_hobby'] == ['读书']

    def test_field_default_value_helper(self):
        assert field_default_value('user_sex') == '保密'
        assert field_default_value('user_hobby') == []
        assert field_default_value('user_occupation') is None


class TestSemanticDedup:
    """语义去重（注入 embedding）：同义变体只保留新值，膨胀被抑制"""

    def _merge(self, old, new, **kw):
        return merge_user_profile(
            old, new, embedding=DictEmbeddings(SYNONYM_MAP),
            semantic_min_items=1, **kw)

    def test_synonym_new_replaces_old(self):
        # 膨胀根源场景："不吃香菜" vs "忌香菜"——新措辞刷新旧措辞（新值优先）
        merged = self._merge(
            {'user_constraints': ['不吃香菜']},
            {'user_constraints': ['忌香菜']},
        )
        assert merged['user_constraints'] == ['忌香菜']

    def test_synonym_replaced_keeps_distinct_ones(self):
        # "少糖"被"不要太甜"替换；"喜欢拍照"是新增；"不吃香菜"与两者无关被保留
        merged = self._merge(
            {'user_constraints': ['少糖', '不吃香菜']},
            {'user_constraints': ['不要太甜', '喜欢拍照']},
        )
        assert merged['user_constraints'] == ['不要太甜', '喜欢拍照', '不吃香菜']

    def test_new_internal_dedup(self):
        # 增量总结内部两两同义：保留先出现的项
        merged = self._merge(
            {'user_constraints': []},
            {'user_constraints': ['忌香菜', '不吃香菜', '不要香菜']},
        )
        assert merged['user_constraints'] == ['忌香菜']

    def test_below_threshold_keeps_both(self):
        # cos=0.8 < 0.85 阈值：宁漏不去，两条都保留
        merged = self._merge(
            {'user_constraints': ['不吃香菜']},
            {'user_constraints': ['少放盐']},
        )
        assert merged['user_constraints'] == ['少放盐', '不吃香菜']

    def test_does_not_mutate_inputs(self):
        old = {'user_constraints': ['不吃香菜']}
        new = {'user_constraints': ['忌香菜']}
        self._merge(old, new)
        assert old['user_constraints'] == ['不吃香菜']
        assert new['user_constraints'] == ['忌香菜']

    def test_embedding_none_keeps_exact_dedup(self):
        # 未注入 embedding（离线默认）：退化为精确去重，同义变体不删
        merged = merge_user_profile(
            {'user_constraints': ['不吃香菜']},
            {'user_constraints': ['忌香菜']},
        )
        assert merged['user_constraints'] == ['忌香菜', '不吃香菜']


class TestSemanticMinItems:
    """膨胀阈值：小列表不调嵌入 API（成本控制）"""

    def test_small_list_skips_api(self):
        # 2 old + 1 new = 3 条 < 5：嵌入不得被调用，走精确去重
        emb = ExplodingEmbeddings()
        merged = merge_user_profile(
            {'user_constraints': ['不吃香菜', '少糖']},
            {'user_constraints': ['喜欢拍照']},
            embedding=emb, semantic_min_items=5,
        )
        assert merged['user_constraints'] == ['喜欢拍照', '不吃香菜', '少糖']

    def test_triggers_at_min_items(self):
        # 精确去重后恰好 = semantic_min_items：开始调嵌入
        emb = DictEmbeddings(SYNONYM_MAP)
        merge_user_profile(
            {'user_constraints': ['不吃香菜', '少糖']},
            {'user_constraints': ['喜欢拍照', '周末爬山', '少放盐']},
            embedding=emb, semantic_min_items=5,
        )
        assert emb.calls, "达到膨胀阈值应调用嵌入 API"


class TestListCap:
    """列表上限：超出保留新值截断"""

    def test_truncate_keeps_newest(self):
        merged = merge_user_profile(
            {'user_hobby': ['a1', 'a2', 'a3', 'a4', 'a5']},
            {'user_hobby': ['b1', 'b2']},
            max_items=3,
        )
        # 新值在前，截掉最旧的尾部
        assert merged['user_hobby'] == ['b1', 'b2', 'a1']

    def test_cap_applies_after_semantic_dedup(self):
        merged = merge_user_profile(
            {'user_constraints': ['不吃香菜', '少糖', '喜欢拍照', '周末爬山', '少放盐']},
            {'user_constraints': ['忌香菜', '不要香菜']},
            embedding=DictEmbeddings(SYNONYM_MAP),
            semantic_min_items=1, max_items=3,
        )
        # 语义去重后剩 ['忌香菜', '少糖', '喜欢拍照', '周末爬山']
        # （'不要香菜'/'不吃香菜'/'少放盐' 与 '忌香菜' 相似被删），再截断前 3
        assert merged['user_constraints'] == ['忌香菜', '少糖', '喜欢拍照']

# -*- coding: utf-8 -*-
"""Layer 0：增量画像合并（merge_user_profile）测试"""
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

# -*- coding: utf-8 -*-
"""Layer 0：SystemPrompt 拼接顺序（缓存前缀规则）与记忆片段注入"""
from langchain_core.documents import Document

from memory_middleware.prompt import SystemPrompt, SystemPromptOperation


class TestSystemPromptOrder:
    def test_core_profile_fragments_order(self):
        """静态核心 → 用户画像 → 记忆片段（缓存前缀友好：易变内容放最后）"""
        prompt = SystemPrompt(core='【核心】', profile='画像内容', memory_fragments=[
            Document(page_content='片段A'),
            Document(page_content='片段B'),
        ])
        s = str(prompt)
        assert s.index('【核心】') < s.index('【当前用户画像】') < s.index('以下为相关片段：') < s.index('片段A')

    def test_empty_profile_no_block(self):
        prompt = SystemPrompt(core='【核心】', memory_fragments=[Document(page_content='片段A')])
        s = str(prompt)
        assert '【当前用户画像】' not in s
        assert '片段A' in s

    def test_empty_fragments_no_header(self):
        prompt = SystemPrompt(core='【核心】', profile='画像')
        s = str(prompt)
        assert '以下为相关片段' not in s
        assert '画像' in s


class TestSystemPromptOperation:
    def test_add_memorys_increments_strengthen(self):
        doc = Document(page_content='内容', metadata={'strengthen_num': 0, 'theme': 't'}, id='u1-1')
        op = SystemPromptOperation(initial_prompt='核心')
        new_docs, old_ids = op.add_memorys([doc])
        assert old_ids == ['u1-1']
        # 注入的片段巩固次数 +1（成功注入后由 update_config_strengthen 落库）
        assert new_docs[0].metadata['strengthen_num'] == 1

    def test_add_memorys_replaces_previous(self):
        op = SystemPromptOperation(initial_prompt='核心')
        d1 = Document(page_content='A', metadata={'strengthen_num': 0}, id='a')
        d2 = Document(page_content='B', metadata={'strengthen_num': 0}, id='b')
        op.add_memorys([d1])
        op.add_memorys([d2])
        assert [f.page_content for f in op.prompt.memory_fragments] == ['B']

    def test_prompt_contains_injected_fragments(self):
        op = SystemPromptOperation(initial_prompt='核心提示词')
        op.add_memorys([Document(page_content='记忆片段内容', metadata={'strengthen_num': 0}, id='x')])
        s = op.get_prompt()
        assert s.startswith('核心提示词')
        assert '记忆片段内容' in s

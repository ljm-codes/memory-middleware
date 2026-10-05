# -*- coding: utf-8 -*-
"""注入选片（_select_top_k）：单主题名额上限与补齐逻辑

动机：细节类信息常集中在少数片段里，而注入只有 top_k 个名额。若同一主题的片段占满名额，
其它主题（往往载着关键细节）就整块进不了上下文——表现为召回在 4/8 与 8/8 之间跳。
"""
import asyncio

from langchain_core.documents import Document

from helpers import make_fragment


def _doc(text: str, theme: str, score: float):
    return (Document(page_content=text, metadata={'theme': theme}), score)


class TestSelectTopK:
    def test_max_per_theme_caps_then_fills(self, build_middleware):
        mw, _, _ = build_middleware(top_k=4, top_k_max_per_theme=2)
        scored = [_doc('a1', 'A', 0.9), _doc('a2', 'A', 0.8), _doc('a3', 'A', 0.7),
                  _doc('b1', 'B', 0.6), _doc('c1', 'C', 0.5)]
        picked = mw._select_top_k(scored)
        # A 主题最多 2 席（a1/a2），第 3 个 A 被跳过 → b1、c1 进来
        assert [d.page_content for d, _ in picked] == ['a1', 'a2', 'b1', 'c1']

    def test_fill_up_when_theme_concentrated(self, build_middleware):
        mw, _, _ = build_middleware(top_k=3, top_k_max_per_theme=1)
        scored = [_doc(f'a{i}', 'A', 1 - i * 0.1) for i in range(5)]
        picked = mw._select_top_k(scored)
        # 全同主题：上限只限制到填不满为止，随后放开补齐到 top_k（不至于注入不足）
        assert [d.page_content for d, _ in picked] == ['a0', 'a1', 'a2']

    def test_no_cap_keeps_plain_top_k(self, build_middleware):
        mw, _, _ = build_middleware(top_k=2)
        scored = [_doc(f'a{i}', 'A', 1 - i * 0.1) for i in range(4)]
        assert [d.page_content for d, _ in mw._select_top_k(scored)] == ['a0', 'a1']

    def test_priority_types_jump_the_queue(self, build_middleware):
        """保底类型优先占席：主题偏了也不至于让 identity/preference 片段整块落选"""
        from langchain_core.documents import Document
        mw, _, _ = build_middleware(top_k=2, always_inject_types=('identity', 'preference'))
        scored = [(Document(page_content='闲聊1', metadata={'theme': 'A', 'type': ['chat']}), 0.9),
                  (Document(page_content='闲聊2', metadata={'theme': 'B', 'type': ['chat']}), 0.8),
                  (Document(page_content='身份', metadata={'theme': 'C', 'type': ['identity']}), 0.1),
                  (Document(page_content='偏好', metadata={'theme': 'D', 'type': ['preference']}), 0.05)]
        picked = mw._select_top_k(scored)
        assert [d.page_content for d, _ in picked] == ['身份', '偏好']

    def test_priority_none_keeps_score_order(self, build_middleware):
        mw, _, _ = build_middleware(top_k=1, always_inject_types=())   # 显式关闭保底 → 纯按分数
        scored = [(Document(page_content='闲聊', metadata={'theme': 'A', 'type': ['chat']}), 0.9),
                  (Document(page_content='身份', metadata={'theme': 'C', 'type': ['identity']}), 0.1)]
        assert [d.page_content for d, _ in mw._select_top_k(scored)] == ['闲聊']

    def test_default_config_protects_identity(self, build_middleware):
        """默认配置就带保底：identity 片段即使分数最低也优先占席（默认值即修复）"""
        mw, _, _ = build_middleware(top_k=1)   # 不传 always_inject_types → 走默认
        assert 'identity' in tuple(mw.config.always_inject_types)
        assert mw.config.retrieve_k is None    # 默认不预筛
        scored = [(Document(page_content='闲聊', metadata={'theme': 'A', 'type': ['chat']}), 0.9),
                  (Document(page_content='身份', metadata={'theme': 'C', 'type': ['identity']}), 0.1)]
        assert [d.page_content for d, _ in mw._select_top_k(scored)] == ['身份']


class TestScoringLog:
    """打分明细（诊断）：把"多维评分里哪一维在决定入选"变成可读数据。

    这份日志是对外结论（每次事件的 R/T/F 与各维加权贡献）的唯一数据源，
    字段名与主项目 `time_memory.py` 保持一致——同一套分析脚本两边通用。
    """

    def test_records_full_candidate_table(self, build_middleware):
        mw, vec_store, _ = build_middleware(top_k=2, always_inject_types=())
        mw.memory_spliter.dialogue_theme_by_user['u1'] = '测试主题'

        async def _add():
            await vec_store.aadd_documents([
                make_fragment('u1', 1, '片段一', theme='测试主题', types=['chat'], age=10),
                make_fragment('u1', 2, '片段二', theme='测试主题', types=['identity'], age=20),
                make_fragment('u1', 3, '片段三', theme='测试主题', types=['chat'], age=30),
            ])
        asyncio.run(_add())

        picked = asyncio.run(mw._extract_fragments_by_time('u1'))

        assert len(mw._scoring_log) == 1, '一次注入事件记一条'
        ev = mw._scoring_log[0]
        assert set(ev['params']) == {'alpha', 'beta', 'gamma', 'delta', 'w0', 'w1', 'w2'}

        rows = ev['candidates']
        assert len(rows) == 3, '不预筛时候选池全部入表'
        for row in rows:
            assert {'id', 'theme', 'type', 'strengthen_num', 'age_s',
                    'r', 't', 'f', 'alpha_r', 'beta_t', 'gamma_f', 's', 'picked'} <= set(row)
            # 记录值自洽：S = αR + βT + γF + δ（各维贡献是四舍五入后的值，留 0.01 容差）
            total = row['alpha_r'] + row['beta_t'] + row['gamma_f'] + ev['params']['delta']
            assert abs(row['s'] - total) < 0.01

        # picked 标记必须与真正返回的选片一致（否则诊断会骗人）
        marked = {row['id'] for row in rows if row['picked']}
        assert marked == {(d.metadata or {}).get('id') for d, _ in picked}
        assert len(marked) == 2

    def test_log_is_bounded(self, build_middleware):
        """有界保留：长会话下不会无界增长（最多 20 条事件）"""
        mw, _, _ = build_middleware()
        assert mw._scoring_log == []
        for i in range(25):
            mw._record_scoring(type('P', (), dict(alpha=0.4, beta=0.3, gamma=0.3, delta=0.0,
                                                  w0=0.1, w1=0.5, w2=0.4))(),
                               [{'id': i}], set())
        assert len(mw._scoring_log) == 20

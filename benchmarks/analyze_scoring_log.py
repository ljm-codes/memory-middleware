# -*- coding: utf-8 -*-
"""读 recall_results.json 的 scoring_log，回答"多维评分里哪一维在决定入选"。

三件事：
1) 每次事件里 αR / βT / γF 各自的**极差**——极差大的维度才在区分候选（其余是近似常数）；
2) τ 消融：用记录的 (age_s, T) 反解 c，再按不同 τ 重算 T 与 S、重排，看入选是否翻转；
3) 巩固：同一片段的 strengthen_num 跨事件是否累积（F 的 refresh 项是否在动）。

用法：
    python analyze_scoring_log.py <key> [<key> ...]      # 逐事件明细
    python analyze_scoring_log.py --pair <旧key> <新key>  # 公式改动的前后验收对比

`--pair` 输出三条验收判据 + 各旋钮的决策权对比：
  A. F 是否还撞顶（旧公式：含 identity/preference 的片段一律被 clamp 成 1.000）
  B. 巩固是否可见（同一片段 str 递增时 F 是否跟着涨）
  C. 召回 / 注入正文体积是否退化
  D. 决策权："关掉某一维/旋钮后入选集合变化的事件数"（口径：集合，非顺序）
"""
import json
import math
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent / 'output' / 'recall_results.json'


def dims(ev):
    """每次事件：各维加权贡献的极差 + 逐维导出的 c（用于校验与消融）"""
    cs = ev['candidates']
    if not cs:
        return None
    spread = {d: max(c[d] for c in cs) - min(c[d] for c in cs)
              for d in ('alpha_r', 'beta_t', 'gamma_f')}
    s_spread = max(c['s'] for c in cs) - min(c['s'] for c in cs)
    return spread, s_spread


def derive_c(cs, tau):
    """由 T = exp(-(age/τ)^c) 反解 c：c = ln(-ln T) / ln(age/τ)（age>0, 0<T<1）"""
    vals = []
    for c in cs:
        age, T = c['age_s'], c['t']
        if age > 0 and 0 < T < 1:
            vals.append(math.log(-math.log(T)) / math.log(age / tau))
    return vals


def rerank(cs, params, tau, c_t, top_k=4, priority=('identity', 'preference')):
    """按给定 τ 重算 T 与 S → 重排 → 返回前 top_k 个 id（保底类型优先，与生产选片一致）"""
    scored = []
    for c in cs:
        t = math.exp(-((c['age_s'] / tau) ** c_t))
        s = (c['alpha_r'] + params['beta'] * t + c['gamma_f'] + params['delta'])
        scored.append((c, s, t))
    scored.sort(key=lambda x: (not (set(priority) & set(x[0]['type'] or [])), -x[1]))
    return scored[:top_k]


def show(key, full):
    """full = 该 key 的整条记录（bmdm 与 _meta 是兄弟字段）"""
    rec = full.get('bmdm') or {}
    log = rec.get('scoring_log') or []
    meta = full.get('_meta', {})
    # 本次跑分实际使用的 τ：从 _meta 的口径戳里读回（A=生产 43200；B 无覆盖 → 基准默认 3600）
    tau0 = float((meta.get('bmdm_config') or {}).get('tau', 3600.0))
    print(f"\n{'='*78}\n{key}  （{meta.get('version')} @ {meta.get('run_at') or meta.get('run_date')}"
          f" | 记录口径 τ={tau0:.0f}s）")
    print(f"事件数 {len(log)}  末次召回 {rec.get('recall')}/{len(rec.get('hits') or [])}")
    if not log:
        print('（无打分数据）')
        return
    for i, ev in enumerate(log, 1):
        p = ev['params']
        cs = ev['candidates']
        sp, s_spread = dims(ev)
        print(f"\n--- 事件 {i}：候选 {len(cs)} | α={p['alpha']} β={p['beta']} γ={p['gamma']} δ={p['delta']}"
              f" | S 极差 {s_spread:.3f}")
        print(f"    各维极差：αR={sp['alpha_r']:.3f}  βT={sp['beta_t']:.3f}  γF={sp['gamma_f']:.3f}"
              f"   → 决定排序的维度：{max(sp, key=sp.get)}")
        cs_ = derive_c(cs, tau=tau0)
        if cs_:
            print(f"    反解 c（按记录 τ={tau0:.0f}）：min={min(cs_):.3f} max={max(cs_):.3f} "
                  f"→ {'一致，消融可信' if max(cs_) - min(cs_) < 0.02 else '不一致，检查记录'}")
        # τ 消融
        c_val = (sum(cs_) / len(cs_)) if cs_ else 0.5
        base = [c['id'] for c, _, _ in rerank(cs, p, tau0, c_val)]
        pure = [c['id'] for c, _, _ in rerank(cs, p, tau0, c_val, priority=())]
        print(f"    入选（生产：保底 identity/preference 优先）τ={tau0:.0f}：{base}")
        print(f"    入选（关掉保底，纯按 S 排序）           τ={tau0:.0f}：{pure}"
              f"{'   ← 保底改变了结果' if pure != base else ''}")
        for tau in (43200.0, 3600.0, 600.0, 120.0):
            ts = [math.exp(-((c['age_s'] / tau) ** c_val)) for c in cs]
            alt = [c['id'] for c, _, _ in rerank(cs, p, tau, c_val)]
            alt_pure = [c['id'] for c, _, _ in rerank(cs, p, tau, c_val, priority=())]
            flip = '一致' if alt == base else f'**翻转** → {alt}'
            flip_pure = '一致' if alt_pure == pure else f'**翻转** → {alt_pure}'
            print(f"    τ={tau:>7.0f}s：T 区间 [{min(ts):.2f}, {max(ts):.2f}] | 生产：{flip}"
                  f" | 无保底：{flip_pure}")
        if i == 1 or i == len(log):
            for c in sorted(cs, key=lambda x: -x['s']):
                mark = '*' if c['picked'] else ' '
                print(f"      {mark} id={c['id']} age={c['age_s']:>7.1f}s str={c['strengthen_num']} "
                      f"| R={c['r']:.3f} T={c['t']:.3f} F={c['f']:.3f} "
                      f"| αR={c['alpha_r']:.3f} βT={c['beta_t']:.3f} γF={c['gamma_f']:.3f} → S={c['s']:.3f}")
    # 巩固：跨事件的 strengthen_num
    seen = {}
    for i, ev in enumerate(log, 1):
        for c in ev['candidates']:
            seen.setdefault(c['id'], []).append((i, c['strengthen_num'], round(c['f'], 3), c['picked']))
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    if multi:
        print("\n--- 巩固（同一片段跨事件）：id → [(事件, strengthen_num, F, 是否入选)]")
        for k, v in multi.items():
            print(f"    id={k}: {v}")


PRIORITY = {'identity', 'preference'}


def _picked_set(cs, params, floor=True, top_k=4):
    rows = [(c, params['alpha'] * c['r'] + params['beta'] * c['t']
             + params['gamma'] * c['f'] + params['delta']) for c in cs]
    key = ((lambda x: (not (PRIORITY & set(x[0]['type'] or [])), -x[1])) if floor
           else (lambda x: -x[1]))
    rows.sort(key=key)
    return frozenset(c['id'] for c, _ in rows[:top_k])


def _newest_set(cs, top_k=4):
    return frozenset(c['id'] for c in
                     sorted(cs, key=lambda x: (not (PRIORITY & set(x['type'] or [])), x['age_s']))[:top_k])


def _tau_set(cs, params, tau, c_t=0.5, top_k=4):
    rows = []
    for x in cs:
        t = math.exp(-((x['age_s'] / tau) ** c_t))
        rows.append((x, params['alpha'] * x['r'] + params['beta'] * t
                     + params['gamma'] * x['f'] + params['delta']))
    rows.sort(key=lambda x: (not (PRIORITY & set(x[0]['type'] or [])), -x[1]))
    return frozenset(x['id'] for x, _ in rows[:top_k])


def decision_power(full):
    """决策权：把某一维/旋钮去掉后，入选**集合**变化的事件数（10 次事件为分母）"""
    log = (full.get('bmdm') or {}).get('scoring_log') or []
    out = {k: 0 for k in ('关掉语义相关(α=0)', '关掉时间衰减(β=0)', '固定参数0.4/0.3/0.3',
                          '关掉类型保底', 'τ压到120s', '朴素:保底+取最新')}
    for ev in log:
        p, cs = ev['params'], ev['candidates']
        base = _picked_set(cs, p)
        if _picked_set(cs, {**p, 'alpha': 0}) != base:
            out['关掉语义相关(α=0)'] += 1
        if _picked_set(cs, {**p, 'beta': 0}) != base:
            out['关掉时间衰减(β=0)'] += 1
        if _picked_set(cs, {**p, 'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3}) != base:
            out['固定参数0.4/0.3/0.3'] += 1
        if _picked_set(cs, p, floor=False) != base:
            out['关掉类型保底'] += 1
        if _tau_set(cs, p, 120.0) != base:
            out['τ压到120s'] += 1
        if _newest_set(cs) != base:
            out['朴素:保底+取最新'] += 1
    return out, len(log)


def acceptance(key, full):
    """把一个 key 压成验收指标"""
    rec = full.get('bmdm') or {}
    log = rec.get('scoring_log') or []
    payload = rec.get('memory_payload') or {}
    cands = [c for ev in log for c in ev['candidates']]
    f_vals = [c['f'] for c in cands]
    r_vals = [c['r'] for c in cands]
    seq = {}
    for i, ev in enumerate(log, 1):
        for c in ev['candidates']:
            seq.setdefault(c['id'], []).append((i, c['strengthen_num'], c['f']))
    cons, cons_flat = [], 0
    for rows in seq.values():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            if b[1] > a[1]:                       # 巩固次数变多了
                cons.append((b[1] - a[1], round(b[2] - a[2], 4)))
                if b[2] - a[2] <= 1e-9:
                    cons_flat += 1
    power, n_ev = decision_power(full)
    return {
        'key': key, 'events': len(log), 'recall': rec.get('recall'),
        'hits': ''.join('1' if h else '0' for h in (rec.get('hits') or [])),
        'inj_tok': payload.get('injected_tokens'), 'cands': len(cands),
        'f_sat': sum(1 for f in f_vals if f >= 0.999), 'f_distinct': len(set(round(f, 4) for f in f_vals)),
        'r_zero': sum(1 for r in r_vals if r == 0), 'r_max': max(r_vals) if r_vals else 0,
        'cons': cons, 'cons_flat': cons_flat, 'power': power, 'n_ev': n_ev,
        'meta': (full.get('_meta') or {}),
    }


def pair(old_key, new_key):
    data = json.loads(OUT.read_text(encoding='utf-8'))
    rows = [acceptance(k, data.get(k) or {}) for k in (old_key, new_key)]
    o, n = rows
    print(f"\n{'='*80}\n验收对比：{old_key}  →  {new_key}")
    for r in rows:
        m = r['meta']
        print(f"  {r['key']:<24} {m.get('version')} @ {m.get('run_at') or m.get('run_date')}")
    print(f"\nA. F 是否还撞顶（含 identity/preference 的片段曾被 clamp 成 1.000）")
    print(f"   旧: {o['f_sat']}/{o['cands']} 个候选 F≥0.999，不同取值 {o['f_distinct']} 种")
    print(f"   新: {n['f_sat']}/{n['cands']} 个候选 F≥0.999，不同取值 {n['f_distinct']} 种")
    print(f"\nB. 巩固是否可见（同一片段 str 变多时 F 的变化）")
    for r in rows:
        pairs = r['cons']
        flat = r['cons_flat']
        desc = '、'.join(f'+{d}str→{df:+.3f}' for d, df in pairs[:4]) or '（无跨事件样本）'
        print(f"   {r['key']:<24} {len(pairs)} 次巩固；其中 {flat} 次 F 没动 | {desc}")
    print(f"\nC. 召回 / 注入体积")
    for r in rows:
        print(f"   {r['key']:<24} 召回 {r['recall']}/{len(r['hits'])} ({r['hits']}) | "
              f"注入正文 {r['inj_tok']} tok")
    print(f"\nD. 决策权（关掉它后入选集合变化的事件数，分母 {o['n_ev']}）")
    print(f"   {'动作':<22}{'旧':>8}{'新':>8}")
    for k in o['power']:
        print(f"   {k:<22}{o['power'][k]:>6}/{o['n_ev']}{n['power'][k]:>6}/{n['n_ev']}")
    print(f"\nE. 语义相关度 R 的刻度")
    for r in rows:
        print(f"   {r['key']:<24} R=0 的候选 {r['r_zero']}/{r['cands']}，最大 R {r['r_max']:.3f}")
    print()


def main():
    if '--pair' in sys.argv:
        i = sys.argv.index('--pair')
        pair(sys.argv[i + 1], sys.argv[i + 2])
        return
    keys = sys.argv[1:]
    if not keys:
        print(__doc__)
        return
    data = json.loads(OUT.read_text(encoding='utf-8'))
    for key in keys:
        full = data.get(key) or {}
        if not full.get('bmdm'):
            print(f'{key}: 没有 bmdm 记录')
            continue
        show(key, full)


if __name__ == '__main__':
    main()

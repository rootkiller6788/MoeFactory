"""语料：**真值与数据一起造，否则无法核对读数**。

这一层测的不是「能不能造出样本」，而是三条让「专家分化成为必需」的性质是否真的成立：

1. 内容词表在域之间**共享** ⇒ 同一个内容 token 在不同域有不同后继；
2. 每个域的规则是**错排**（无不动点）⇒ 规则非平凡，一个专家不能靠「原样复制」通吃；
3. **可计算下界** ``domain_blind_floor`` 能从构造里推出来 ⇒ 有了一条不靠拍阈值的判据。

第 3 条是本项目的验收核心，所以它被单独测：算出来的值必须等于手算的值。
"""

from __future__ import annotations

import json
import math
import os

import pytest

from moefactory.corpus import (
    Corpus,
    Sample,
    build_corpus,
    iter_batches,
    make_batch,
    read_corpus,
    write_corpus,
)


# ----------------------------------------------------------------------
# 构造：确定性与规则性质
# ----------------------------------------------------------------------
def test_corpus_is_deterministic(cfg: dict) -> None:
    a, b = build_corpus(cfg), build_corpus(cfg)
    assert a.digest == b.digest
    assert a.permutations == b.permutations
    assert [s.sample_id for s in a.train] == [s.sample_id for s in b.train]


def test_different_seed_gives_different_corpus(cfg: dict) -> None:
    import copy

    c = copy.deepcopy(cfg)
    c["seed"] = int(cfg["seed"]) + 1
    assert build_corpus(c).digest != build_corpus(cfg).digest


def test_permutations_are_derangements(corpus: Corpus) -> None:
    """错排：``perm[i] != i``。有不动点的话规则就退化成「原样输出」。"""
    assert len(corpus.permutations) == corpus.n_domains
    for perm in corpus.permutations:
        assert len(perm) == corpus.n_content
        assert sorted(perm) == list(range(corpus.n_content))
        assert all(perm[i] != i for i in range(corpus.n_content))


def test_domains_have_distinct_rules(corpus: Corpus) -> None:
    assert len(set(corpus.permutations)) == corpus.n_domains


def test_vocab_layout(corpus: Corpus) -> None:
    assert corpus.vocab_size == corpus.n_content + corpus.n_domains
    for d in range(corpus.n_domains):
        assert corpus.tag_of(d) == corpus.n_content + d
        assert corpus.tag_of(d) >= corpus.n_content  # 标签不落在内容区间里


def test_rule_matches_permutation(corpus: Corpus) -> None:
    for d in range(corpus.n_domains):
        for c in range(corpus.n_content):
            assert corpus.rule(d, c) == corpus.permutations[d][c]


def test_samples_follow_their_domain_rule(corpus: Corpus) -> None:
    for s in corpus.train + corpus.test:
        assert s.tag == corpus.tag_of(s.domain)
        for a, b in zip(s.content, s.content[1:]):
            assert b == corpus.rule(s.domain, a)


def test_sample_ids_are_unique(corpus: Corpus) -> None:
    ids = corpus.sample_ids()
    assert len(ids) == len(set(ids))


def test_split_sizes_follow_config(cfg: dict, corpus: Corpus) -> None:
    assert len(corpus.train) == cfg["task"]["n_train"]
    assert len(corpus.test) == cfg["task"]["n_test"]
    assert all(len(s.content) == cfg["task"]["seq_len"] for s in corpus.train)


def test_domains_are_balanced_across_samples(corpus: Corpus) -> None:
    """域是轮转分配的，所以「每域等权」这个前提成立 —— 下界公式依赖它。"""
    counts = [0] * corpus.n_domains
    for s in corpus.train:
        counts[s.domain] += 1
    assert max(counts) - min(counts) <= 1


# ----------------------------------------------------------------------
# 可计算下界（本项目的验收判据）
# ----------------------------------------------------------------------
def test_shared_vocabulary_makes_specialisation_necessary(corpus: Corpus) -> None:
    """同一个内容 token 在 4 个域里几乎总有 3~4 个不同后继。

    **不需要每个 token 都恰好 4 个不同后继。** 实测默认语料是 3.25/4：随机错排之间
    偶尔会撞到同一个值（例如某个 token 的两个域都指向同一个后继）。只要「一个 token
    平均有多个后继」成立，看不见域的路由器就不可能完美 —— 这才是分化的存在理由。
    """
    counts = [len(set(corpus.successors_of(c))) for c in range(corpus.n_content)]
    assert min(counts) >= 2, "存在后继唯一的 token，那部分退化成了「无歧义」"
    assert corpus.distinct_successor_ratio() > 0.75 * corpus.n_domains
    assert sum(1 for n in counts if n == corpus.n_domains) >= corpus.n_content // 4


def test_domain_blind_floor_matches_hand_computation(corpus: Corpus) -> None:
    """下界必须等于「手算的 E_c[H(后继|c)]」—— 它是推导出来的，不是标定出来的。"""
    total = 0.0
    for c in range(corpus.n_content):
        succ = [corpus.permutations[d][c] for d in range(corpus.n_domains)]
        counts: dict[int, int] = {}
        for s in succ:
            counts[s] = counts.get(s, 0) + 1
        n = len(succ)
        total += -math.fsum((v / n) * math.log(v / n) for v in counts.values())
    assert corpus.domain_blind_floor() == pytest.approx(total / corpus.n_content, abs=1e-12)


def test_domain_blind_floor_is_nontrivial(corpus: Corpus) -> None:
    """下界必须显著大于 0，否则「突破下界」这件事没有信息量。"""
    floor = corpus.domain_blind_floor()
    assert floor > 0.5
    assert floor <= math.log(corpus.n_domains) + 1e-9


def test_domain_blind_floor_is_the_minimum_cross_entropy(corpus: Corpus) -> None:
    """下界的语义检验：**它是「看不见域」这一族模型的最小交叉熵。**

    对每个内容 token ``c``，后继分布 ``p_c`` 是「各域等权」的混合。任何只看到 ``c``
    的预测分布 ``q_c`` 都有

        CE(p_c, q_c) = H(p_c) + KL(p_c ‖ q_c) ≥ H(p_c)

    等号只在 ``q_c == p_c`` 时成立。这条不等式就是「loss < 下界 ⇒ 一定用上了域信息」
    的全部依据 —— 所以它必须被单独测，而不是靠直觉。
    """
    def cross_entropy(qs: list[list[float]]) -> float:
        total = 0.0
        for c in range(corpus.n_content):
            succ = corpus.successors_of(c)
            n = len(succ)
            p: dict[int, float] = {}
            for s in succ:
                p[s] = p.get(s, 0.0) + 1.0 / n
            total += -math.fsum(pi * math.log(qs[c][s]) for s, pi in p.items())
        return total / corpus.n_content

    n = corpus.n_content
    uniform = [[1.0 / n] * n for _ in range(n)]
    # 每个 token 押一个「最常见后继」的尖峰分布（留一点质量，避免 log(0)）
    spike = []
    for c in range(n):
        row = [1e-9] * n
        counts: dict[int, int] = {}
        for s in corpus.successors_of(c):
            counts[s] = counts.get(s, 0) + 1
        row[max(counts, key=lambda k: counts[k])] = 1.0
        spike.append([v / sum(row) for v in row])
    exact = []
    for c in range(n):
        row = [0.0] * n
        for s in corpus.successors_of(c):
            row[s] += 1.0 / corpus.n_domains
        exact.append(row)

    floor = corpus.domain_blind_floor()
    assert cross_entropy(uniform) >= floor - 1e-9
    assert cross_entropy(spike) >= floor - 1e-9
    assert cross_entropy(exact) == pytest.approx(floor, abs=1e-9), "取等号当且仅当 q = p"


def test_corpus_meta_records_the_floor(cfg: dict) -> None:
    meta = build_corpus(cfg).to_dict(with_samples=False)
    assert meta["domain_blind_floor"] == pytest.approx(build_corpus(cfg).domain_blind_floor())
    assert "train" not in meta  # with_samples=False 不塞样本


# ----------------------------------------------------------------------
# batch
# ----------------------------------------------------------------------
def test_make_batch_flattens_to_token_level(corpus: Corpus) -> None:
    samples = corpus.train[:5]
    b = make_batch(corpus, samples)
    per = corpus.seq_len - 1
    assert len(b) == 5 * per
    assert len(b.tags) == len(b.inputs) == len(b.targets) == len(b.domains) == len(b)
    assert len(b.sample_ids) == 5


def test_make_batch_supervision_is_next_token(corpus: Corpus) -> None:
    b = make_batch(corpus, corpus.test)
    for i in range(len(b)):
        assert b.targets[i] == corpus.rule(b.domains[i], b.inputs[i])
        assert b.tags[i] == corpus.tag_of(b.domains[i])


def test_iter_batches_is_deterministic_and_covers_everything(corpus: Corpus) -> None:
    first = [b.inputs for b in iter_batches(corpus, corpus.train, 7, 1, 0)]
    again = [b.inputs for b in iter_batches(corpus, corpus.train, 7, 1, 0)]
    assert first == again
    assert sum(len(x) for x in first) == len(corpus.train) * (corpus.seq_len - 1)


def test_iter_batches_epoch_changes_order(corpus: Corpus) -> None:
    e0 = [b.inputs for b in iter_batches(corpus, corpus.train, 7, 1, 0)]
    e1 = [b.inputs for b in iter_batches(corpus, corpus.train, 7, 1, 1)]
    assert e0 != e1


# ----------------------------------------------------------------------
# 落盘 / 读回
# ----------------------------------------------------------------------
def test_write_read_corpus_round_trip(corpus: Corpus, tmp_path) -> None:
    path = str(tmp_path / "corpus.jsonl")
    write_corpus(corpus, path)
    back = read_corpus(path)
    assert back.digest == corpus.digest
    assert back.permutations == corpus.permutations
    assert [s.sample_id for s in back.train] == [s.sample_id for s in corpus.train]
    assert [s.sample_id for s in back.test] == [s.sample_id for s in corpus.test]
    assert back.domain_blind_floor() == pytest.approx(corpus.domain_blind_floor())


def test_corpus_jsonl_has_meta_line(corpus: Corpus, tmp_path) -> None:
    path = str(tmp_path / "corpus.jsonl")
    write_corpus(corpus, path)
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert "_meta" in rows[0]
    assert {r["split"] for r in rows[1:]} == {"train", "test"}
    assert len(rows) - 1 == len(corpus.train) + len(corpus.test)


def test_read_corpus_without_meta_raises(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sample_id": "x", "domain": 0, "tag": 12, "content": [1]}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError):
        read_corpus(str(path))


def test_sample_round_trip() -> None:
    s = Sample(sample_id="abc", domain=2, tag=14, content=[1, 2, 3])
    assert Sample.from_dict(s.to_dict()) == s


def test_corpus_from_dict_needs_samples(corpus: Corpus) -> None:
    d = corpus.to_dict(with_samples=True)
    back = Corpus.from_dict(d)
    assert back.digest == corpus.digest
    assert len(back.train) == len(corpus.train)

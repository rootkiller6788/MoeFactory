"""合成语料：**带域真值**的多任务 next-token 任务。

## 任务构造（为什么必须是这个样子）

```
vocab = [内容 token 0..C-1] + [任务标签 n_content+0 .. n_content+D-1]
域 d 的规则 = 一个随机错排 perm_d : 内容 token → 内容 token
一条样本   = [tag_d, c0, c1, c2, ...]，其中 c_{i+1} = perm_d(c_i)
监督       = 下一个内容 token（即 perm_d(c_i)）
```

三个关键点，每一个都是为了**让「专家分化」成为必需而不是装饰**：

1. **内容词表在域之间共享。** 于是同一个内容 token ``c`` 在 4 个域里有 4 个
   **不同的**后继 —— 只看 ``c`` 无法知道该预测什么。这一条是分化的存在理由。
2. **标签进路由器、不进专家。** 路由器看到 ``emb[c] + emb[tag]``，专家只看到
   ``emb[c]``。所以专家的信息**只能**来自「路由器把它派给了哪个任务」。
   若专家也能看到标签，一个专家就能通吃四个域，分化又会退化成装饰。
3. **规则用随机错排，不用循环移位。** 循环移位是线性的，一个专家能轻松通吃。

## 可计算的基线（本项目的「外部已知真值」）

由第 1 条直接得到一个**算出来的**（不是拟合的）下界：

    domain_blind_floor = E_c[ H(后继 | c) ]

一个**看不见域**的路由器，对每个内容 token 只能押一个后继，其最优损失就是
「给定 c 时后继的熵」。默认配置下它约为 ``ln(3~4) ≈ 1.28``。

于是有一条极硬的判据：

    **loss 低于 domain_blind_floor ⇒ 路由一定用上了域信息（即发生了分化）。**

这条比「NMI 大于某个阈值」强得多：NMI 的阈值是拍的，而这个下界是从任务
构造里**推**出来的。报告里两者都会出现，结论以这条为准。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

from .rng import make_rng, sha256_text


@dataclass(frozen=True)
class Sample:
    """一条样本：任务标签 + 内容序列 + 域真值 + 内容寻址 id。"""

    sample_id: str
    domain: int
    tag: int
    content: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Sample":
        return Sample(
            sample_id=d["sample_id"],
            domain=d["domain"],
            tag=d["tag"],
            content=list(d["content"]),
        )


def _derangement(width: int, rng: Any) -> list[int]:
    """造一个随机**错排**（``perm[i] != i``）。无不动点，规则才非平凡。"""
    if width < 2:
        raise ValueError("width 至少为 2 才能造错排")
    while True:
        idx = list(range(width))
        rng.shuffle(idx)
        if all(idx[i] != i for i in range(width)):
            return idx


@dataclass(frozen=True)
class Corpus:
    """带真值的合成语料。**真值与数据一起造**，否则无法核对读数。"""

    name: str
    n_domains: int
    n_content: int
    seq_len: int
    seed: int
    train: tuple[Sample, ...]
    test: tuple[Sample, ...]
    permutations: tuple[tuple[int, ...], ...]
    digest: str

    # -- 真值 --------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.n_content + self.n_domains

    def tag_of(self, domain: int) -> int:
        """域 d 的标签 token id。"""
        return self.n_content + domain

    def rule(self, domain: int, token: int) -> int:
        """域 ``domain`` 的转移规则：``token → perm(token)``（token 是内容 token）。"""
        return self.permutations[domain][token]

    def sample_ids(self) -> list[str]:
        return [s.sample_id for s in self.train] + [s.sample_id for s in self.test]

    def successors_of(self, token: int) -> list[int]:
        """内容 token ``c`` 在各域下的后继（可能重复）。"""
        return [self.permutations[d][token] for d in range(self.n_domains)]

    def domain_blind_floor(self) -> float:
        """**算出来的**下界：看不见域的路由器能达到的最优损失。

        对每个内容 token，把「各域后继」当经验分布求熵，再对 token 取平均。
        每个域等权、每个内容 token 等权 —— 语料就是按这个造的。
        """
        total = 0.0
        for c in range(self.n_content):
            succ = self.successors_of(c)
            counts: dict[int, int] = {}
            for s in succ:
                counts[s] = counts.get(s, 0) + 1
            n = len(succ)
            h = -math.fsum((v / n) * math.log(v / n) for v in counts.values())
            total += h
        return total / self.n_content

    def distinct_successor_ratio(self) -> float:
        """平均每个内容 token 有几个不同的后继（上限 ``n_domains``）。"""
        vals = [len(set(self.successors_of(c))) for c in range(self.n_content)]
        return sum(vals) / len(vals)

    # -- 序列化 ------------------------------------------------------
    def to_dict(self, with_samples: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "n_domains": self.n_domains,
            "n_content": self.n_content,
            "seq_len": self.seq_len,
            "seed": self.seed,
            "digest": self.digest,
            "vocab_size": self.vocab_size,
            "permutations": [list(p) for p in self.permutations],
            "n_train": len(self.train),
            "n_test": len(self.test),
            "domain_blind_floor": self.domain_blind_floor(),
        }
        if with_samples:
            d["train"] = [s.to_dict() for s in self.train]
            d["test"] = [s.to_dict() for s in self.test]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Corpus":
        return Corpus(
            name=d["name"],
            n_domains=d["n_domains"],
            n_content=d["n_content"],
            seq_len=d["seq_len"],
            seed=d["seed"],
            train=tuple(Sample.from_dict(s) for s in d["train"]),
            test=tuple(Sample.from_dict(s) for s in d["test"]),
            permutations=tuple(tuple(p) for p in d["permutations"]),
            digest=d["digest"],
        )


def build_corpus(cfg: dict) -> Corpus:
    """按配置造语料。**同配置必得同语料**（真值也必须逐位一致）。"""
    t = cfg["task"]
    name = t.get("name", "domain-recall")
    n_domains = int(t.get("n_domains", 4))
    n_content = int(t.get("n_content", 12))
    seq_len = int(t.get("seq_len", 10))
    n_train = int(t.get("n_train", 48))
    n_test = int(t.get("n_test", 16))
    seed = int(cfg["seed"])

    rng = make_rng(seed, "corpus", name, n_domains, n_content, seq_len)
    perms = tuple(tuple(_derangement(n_content, rng)) for _ in range(n_domains))

    def make_split(count: int, split: str) -> tuple[Sample, ...]:
        out: list[Sample] = []
        for i in range(count):
            domain = i % n_domains
            r = make_rng(seed, "sample", split, i, name)
            cur = r.randrange(n_content)
            content = [cur]
            for _ in range(seq_len - 1):
                cur = perms[domain][cur]
                content.append(cur)
            sid = sha256_text(
                f"{name}|{split}|{i}|{domain}|{','.join(map(str, content))}"
            )[:16]
            out.append(
                Sample(
                    sample_id=sid,
                    domain=domain,
                    tag=n_content + domain,
                    content=content,
                )
            )
        return tuple(out)

    train = make_split(n_train, "train")
    test = make_split(n_test, "test")
    digest = hashlib.sha256(
        "\n".join(sorted(s.sample_id for s in train + test)).encode("utf-8")
    ).hexdigest()[:12]
    return Corpus(
        name=name,
        n_domains=n_domains,
        n_content=n_content,
        seq_len=seq_len,
        seed=seed,
        train=train,
        test=test,
        permutations=perms,
        digest=digest,
    )


# ----------------------------------------------------------------------
# batch
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    """展平后的一个 batch：内容 token、任务标签、目标、域真值，都是 ``[T]``。"""

    inputs: list[int]      # 内容 token（专家看得到的）
    tags: list[int]        # 任务标签（只有路由器看得到）
    targets: list[int]     # 下一个内容 token
    domains: list[int]     # 域真值 —— NMI / purity 的原料
    sample_ids: list[str]

    def __len__(self) -> int:
        return len(self.inputs)


def make_batch(corpus: Corpus, samples: Sequence[Sample]) -> Batch:
    """把若干条样本**展平**成逐位置监督。

    展平（而不是按序列喂）是刻意的：MoE 的路由是**逐 token** 的，展平后每个
    token 就是一个独立的路由决策点，统计口径最直接。
    """
    inputs: list[int] = []
    tags: list[int] = []
    targets: list[int] = []
    domains: list[int] = []
    ids: list[str] = []
    for s in samples:
        c = s.content
        inputs.extend(c[:-1])
        targets.extend(c[1:])
        tags.extend([s.tag] * (len(c) - 1))
        domains.extend([s.domain] * (len(c) - 1))
        ids.append(s.sample_id)
    return Batch(inputs=inputs, tags=tags, targets=targets, domains=domains, sample_ids=ids)


def iter_batches(
    corpus: Corpus, samples: Sequence[Sample], batch_size: int, seed: int, epoch: int
) -> Iterator[Batch]:
    """确定性打乱后切批。同一 ``(seed, epoch)`` 必得同一顺序。"""
    order = list(range(len(samples)))
    make_rng(seed, "shuffle", epoch, corpus.digest).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [samples[i] for i in order[start : start + batch_size]]
        yield make_batch(corpus, chunk)


def write_corpus(corpus: Corpus, path: str) -> None:
    """把语料连同真值落盘（jsonl：一行元信息 + 一行一条样本）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": corpus.to_dict(with_samples=False)}, ensure_ascii=False) + "\n")
        for split, samples in (("train", corpus.train), ("test", corpus.test)):
            for s in samples:
                row = s.to_dict()
                row["split"] = split
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_corpus(path: str) -> Corpus:
    """读回语料（与 :func:`write_corpus` 互逆）。"""
    meta: dict[str, Any] | None = None
    train: list[Sample] = []
    test: list[Sample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            s = Sample.from_dict(row)
            (train if row.get("split") == "train" else test).append(s)
    if meta is None:
        raise ValueError(f"{path} 缺少元信息行")
    return Corpus(
        name=meta["name"],
        n_domains=meta["n_domains"],
        n_content=meta["n_content"],
        seq_len=meta["seq_len"],
        seed=meta["seed"],
        train=tuple(train),
        test=tuple(test),
        permutations=tuple(tuple(p) for p in meta["permutations"]),
        digest=meta["digest"],
    )

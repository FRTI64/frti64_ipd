# frti64_ipd — Axelrod IPD 策略

本仓库包含两个可直接用于 [Axelrod](https://github.com/Axelrod-Python/Axelrod) 的独立单文件迭代囚徒困境（IPD）策略。两个文件都只依赖 `axelrod` 和 Python 标准库，不依赖其他仓库代码或外部权重文件。

## 策略一览

| 文件 | 类 | 类型 | 外部依赖 |
|---|---|---|---|
| [`rl_switch.py`](rl_switch.py) | `RLSwitch` | RL-Wide160 + RL-Repair 窄切换 | 无；两份 MLP 权重已内嵌 |
| [`triple_v4.py`](triple_v4.py) | `TripleV4` | Final + Responsive3 + BayesPlus 多数投票 | 无；纯规则/贝叶斯逻辑 |

两个策略都按 **100 回合**设计：第 1 天合作，第 100 天背叛，训练和统计口径均基于 100 回合。

## 环境

- Python 3.10+
- `axelrod==4.14.0`（开发与验证版本）

```bash
python -m pip install axelrod==4.14.0
```

## 快速开始

```python
import axelrod as axl

from rl_switch import RLSwitch
from triple_v4 import TripleV4

players = [RLSwitch(), TripleV4(), axl.TitForTat()]

results = axl.Tournament(
    players,
    turns=100,
    repetitions=5,
    seed=1,
).play()

print(results.ranked_names)
```

单场对局：

```python
match = axl.Match((RLSwitch(), axl.TitForTat()), turns=100)
match.play()
print(match.final_score())
```

两个文件都可以直接运行自检：

```bash
python rl_switch.py
python triple_v4.py
```

## RL-Switch

`RLSwitch` 是 pri_dile `rl-switch` 候选的 Axelrod 接口移植：

- 默认使用 **RL-Wide160**（160/128 MLP）；
- 前 6 回合双方都至少背叛 2 次、且对手最新一手为合作时，整场剩余时间切换到 **RL-Repair**（64/32 MLP）；
- 两份 `ipd-neural-policy-v1` 权重以 zlib + base85 压缩形式内嵌在源文件中；
- 对手从未背叛时保持合作；第 100 天背叛。

```python
player = RLSwitch()                    # 默认 seed=20261024
player = RLSwitch(seed=20261024, name="My RL-Switch")
```

### 模式口径

活跃文件只保留 **`sample`** 模式，这是 pri_dile 统计和测试协议使用的模式：

```python
RLSwitch()                 # sample
RLSwitch(mode="sample")    # 显式写法，兼容候选 build 接口
```

`mode="argmax"` 会抛出 `ValueError`。旧的确定性 argmax 版本已归档到：

```text
archive/rl_switch_argmax.py
```

需要复现 argmax 行为时：

```python
from archive.rl_switch_argmax import RLSwitch as RLSwitchArgmax

player = RLSwitchArgmax(mode="argmax")
```

## Triple-v4

`TripleV4` 是 pri_dile `triple-v4` 候选的 Axelrod 接口移植：

- Final、Responsive3、BayesPlus 三个组件各自给出 C/D，按多数投票融合；
- Final 组件额外包含一条窄的“噪声互惠者”分支：对手被我们背叛后强烈报复、但又偶尔在合作后背叛时，降低惩罚升级的概率；
- 不增加主动 probe；
- 内部随机种子固定为与原候选一致的值，动作在相同历史下可复现。

```python
player = TripleV4()                 # seed 仅保留 API 兼容
player = TripleV4(name="My Triple-v4")
```

## 复现与验证

- 活跃 `RLSwitch` 的 `sample` 模式与原始 `ipd` 实现在 224 个非 long-run Axelrod 策略、100 回合对局中逐动作一致；
- archived `RLSwitch(mode="argmax")` 同样通过逐动作 parity；
- `TripleV4` 与原始 `triple-v4` 在相同历史下逐动作一致，并通过随机历史 fuzz；
- 随机策略在每场 `reset()` 时重建内部 RNG，因此同一组对局可重复。

## 限制

- 两个策略的 endgame、late-harvester 和特征归一化都基于 100 回合；请使用 `turns=100`，不要直接用于其他回合长度。
- `RLSwitch` 的 `sample` 模式是随机策略；如果不希望随机性，请使用归档中的 argmax 版本。
- `TripleV4(seed=...)` 的 `seed` 参数不改变三个组件的固定种子，仅用于接口兼容。
- 本仓库只包含策略代码，不包含 pri_dile 的训练、评估和权重导出工具。

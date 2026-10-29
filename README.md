# frti64_ipd — Axelrod IPD 策略集

本仓库包含 24 个可直接用于 [Axelrod](https://github.com/Axelrod-Python/Axelrod) 的独立单文件迭代囚徒困境（IPD）策略。每个文件只依赖 `axelrod` 和 Python 标准库，不依赖其他仓库代码；学习型策略的权重已内嵌。

> **重要：所有策略都在 `T=10, R=6, P=1, S=0` 的收益矩阵下开发和测试。**
> Axelrod 的默认 `Game()` 是 `T=5, R=3, P=1, S=0`，直接使用默认矩阵会得到不可比的分数。
> 请使用下面的 `GAME`：
>
> ```python
> import axelrod as axl
>
> GAME = axl.Game(r=6, s=0, t=10, p=1)  # T=10, R=6, P=1, S=0
> ```
>
> 对局长度同样固定为 **100 回合**（`turns=100`）。所有策略按第 1 天合作、第 100 天背叛的口径实现。

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

GAME = axl.Game(r=6, s=0, t=10, p=1)

players = [RLSwitch(), TripleV4(), axl.TitForTat()]

results = axl.Tournament(
    players,
    turns=100,
    repetitions=5,
    seed=1,
    game=GAME,
).play()

print(results.ranked_names)
```

单场对局：

```python
match = axl.Match(
    (RLSwitch(), axl.TitForTat()),
    turns=100,
    game=GAME,
)
match.play()
print(match.final_score())
```

每个文件都可以直接运行自检（自检已经使用 `GAME`）：

```bash
python rl_switch.py
python triple_v4.py
python final_tuned.py
python selfplay.py
```

## 文件与类

### 规则 / 贝叶斯策略

| 文件 | 类 | 原候选 | 说明 |
|---|---|---|---|
| `final_base.py` | `FinalBase` | final-base | 默认 Final 状态机 |
| `final_tuned.py` | `FinalTuned` | final-tuned | 内嵌 tuned Final 参数 |
| `resp3_base.py` | `Resp3Base` | resp3-base | 默认 Responsive3 |
| `resp3_tuned.py` | `Resp3Tuned` | resp3-tuned | 内嵌 tuned Responsive3 参数 |
| `bayes_base.py` | `BayesBase` | bayes-base | 默认 BayesPlus 模型库 |
| `bayes_tuned.py` | `BayesTuned` | bayes-tuned | 内嵌 tuned BayesPlus 先验/阈值 |
| `hybrid.py` | `Hybrid` | hybrid | tuned Final + 默认 Responsive3 |
| `triple_majority.py` | `TripleMajority` | triple-majority | Final + Resp3 + BayesPlus 多数投票 |
| `triple_v4.py` | `TripleV4` | triple-v4 | Triple-majority 的噪声互惠者分支版本 |

### 学习型 / 元策略

| 文件 | 类 | 原候选 | 内嵌内容 |
|---|---|---|---|
| `neural.py` | `Neural` | neural | 85 维 MLP |
| `gru.py` | `GRU` | gru | 85 维 GRU-Net |
| `rl.py` | `RL` | rl | RL-PPO actor-critic checkpoint（导出为部署 MLP） |
| `rl_hard60.py` | `RLHard60` | rl-hard60 | RL-PPO-Hard60 MLP |
| `rl_repair.py` | `RLRepair` | rl-repair | RL-Repair MLP |
| `rl_pattern.py` | `RLPattern` | rl-pattern | RL-Pattern MLP |
| `rl_wide.py` | `RLWide` | rl-wide | RL-Wide 128/128 MLP |
| `rl_wide160.py` | `RLWide160` | rl-wide160 | RL-Wide160 160/128 MLP |
| `rl_switch.py` | `RLSwitch` | rl-switch | RL-Wide160 + RL-Repair 双 MLP + 切换规则 |
| `rl_dagger_gru.py` | `GRUDagger` | rl-dagger-gru | GRU-DAgger 64 隐层 GRU |
| `rl_gate.py` | `RLGate` | rl-gate | 双 MLP + 25 维 logistic gate |
| `adv_mlp.py` | `AdvMLP` | adv-mlp | Neural 基座 + 约束利用规则 |
| `adv_gru.py` | `AdvGRU` | adv-gru | GRU 基座 + 约束利用规则 |
| `adv_rl.py` | `AdvRL` | adv-rl | BC/PPO MLP 基座 + 激进利用规则 |
| `selfplay.py` | `SelfPlayGRU` | selfplay | 两层 GRU + 21 维 self-play observation |

## 收益矩阵与计分

开发/测试矩阵：

```text
               对手 C    对手 D
我方 C           R=6       S=0
我方 D           T=10      P=1
```

对应 Axelrod：

```python
GAME = axl.Game(r=6, s=0, t=10, p=1)
```

`axelrod.Match` 和 `axelrod.Tournament` 都必须显式传入 `game=GAME`。策略本身只读取历史动作，因此即使误用默认矩阵也能运行，但分数、对手的收益敏感策略行为以及复现结果都不再是原始开发口径。

## 模式口径

- 所有学习型导出文件只支持 `sample` 模式；这是 pri_dile 统计和测试协议使用的模式。
- `mode="sample"` 可以显式传入，兼容原始候选的 `build` 接口。
- 其他模式会抛出 `ValueError`。
- `rl-switch` 的确定性 argmax 版本单独归档在 `archive/rl_switch_argmax.py`：

```python
from archive.rl_switch_argmax import RLSwitch as RLSwitchArgmax

player = RLSwitchArgmax(mode="argmax")
```

## 复现与验证

- 每个导出文件都与原始 pri_dile 候选做过逐动作 parity；
- 标准 3/0/5/1 矩阵下覆盖 224 个非 long-run Axelrod 策略、100 回合对局；
- `T=10, R=6, P=1, S=0` 矩阵下覆盖 213 个可构造的 Axelrod 策略；其余 11 个是 Axelrod 内置零行列式策略，它们的 four-vector 按标准 3/0/5/1 校准，在自定义矩阵下会直接拒绝构造，这是 Axelrod 对手库的限制；
- `GRU`、`GRU-DAgger`、`Adv-GRU`、`SelfPlay-GRU` 会在每场 `reset()` 时重建隐状态；
- `RL-Gate` 在 `t=10` 时读取前 10 回合并锁定一个专家；
- `RL-Switch` 在前 6 回合检测到“双方都背叛过至少 2 次、对手最新一手合作”后切换到 RL-Repair。

## 限制

- 策略按 **100 回合**设计；请使用 `turns=100`，不要直接用于其他长度。
- 开发/测试收益矩阵固定为 `T=10, R=6, P=1, S=0`；不要用 Axelrod 默认矩阵比较分数。
- `sample` 模式是随机策略；同一 seed 下可复现，不同 seed 会有动作差异。
- `SelfPlay-GRU` 使用独立的 21 维 observation，不共享 85 维 neural features。
- 本仓库只包含推理代码和权重，不包含 pri_dile 的训练/评估工具。

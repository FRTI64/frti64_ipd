# frti64_ipd — Axelrod IPD 策略集

本仓库包含 30 个可直接用于 [Axelrod](https://github.com/Axelrod-Python/Axelrod) 的独立单文件迭代囚徒困境（IPD）策略。每个文件只依赖 `axelrod` 和 Python 标准库，不依赖其他仓库代码；学习型策略的权重已内嵌。

## 参赛策略

最终参赛提交为以下两个策略：

| 文件 | 类 | 名称 | 类型 |
|---|---|---|---|
| **`rl_full_guard_c.py`** | `RLFullGuardC` | **RL-FullGuard-C** | RL-V5-FT-Complex + probe/late 反应式防护 |
| **`triple_v4.py`** | `TripleV4` | **Triple-v4** | Final + Responsive3 + BayesPlus 多数投票 |

上一版学习型参赛策略 `rl_switch.py` / `RLSwitch` 保留为 fallback / 对照，不再作为首选。

其余文件为候选、历史或对照策略，不参与最终参赛提交。

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

from rl_full_guard_c import RLFullGuardC
from triple_v4 import TripleV4

GAME = axl.Game(r=6, s=0, t=10, p=1)

players = [RLFullGuardC(), TripleV4(), axl.TitForTat()]

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
    (RLFullGuardC(), axl.TitForTat()),
    turns=100,
    game=GAME,
)
match.play()
print(match.final_score())
```

每个文件都可以直接运行自检（自检已经使用 `GAME`）：

```bash
python rl_full_guard_c.py
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
| **`triple_v4.py`** | **`TripleV4`** | triple-v4 | **参赛策略**；Triple-majority 的噪声互惠者分支版本 |

### 学习型 / 元策略（含 2026-10-30/31 新增）

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
| `rl_v5_ft.py` | `RLV5FT` | rl-v5-ft | V5-FT 160/128 MLP（train_v5 fine-tune） |
| `rl_v5_ft_complex.py` | `RLV5FTComplex` | rl-v5-ft-complex | V5-FT-Complex 160/128 MLP（train_v5 + train_complex） |
| `rl_probe_guard.py` | `RLProbeGuard` | rl-probe-guard | V5-FT + 早期 probe 反应式 guard |
| `rl_defection_guard.py` | `RLDefectionGuard` | rl-defection-guard | V5-FT + late 反应式 guard |
| `rl_full_guard.py` | `RLFullGuard` | rl-full-guard | V5-FT + probe/late 反应式 guard |
| **`rl_full_guard_c.py`** | **`RLFullGuardC`** | rl-full-guard-c | **参赛策略**；V5-FT-Complex 160/128 MLP + probe/late 反应式 guard |
| `rl_switch.py` | `RLSwitch` | rl-switch | 上一版学习型参赛策略 / fallback；RL-Wide160 + RL-Repair 双 MLP + 切换规则 |
| `rl_dagger_gru.py` | `GRUDagger` | rl-dagger-gru | GRU-DAgger 64 隐层 GRU |
| `rl_gate.py` | `RLGate` | rl-gate | 双 MLP + 25 维 logistic gate |
| `adv_mlp.py` | `AdvMLP` | adv-mlp | Neural 基座 + 约束利用规则 |
| `adv_gru.py` | `AdvGRU` | adv-gru | GRU 基座 + 约束利用规则 |
| `adv_rl.py` | `AdvRL` | adv-rl | BC/PPO MLP 基座 + 激进利用规则 |
| `selfplay.py` | `SelfPlayGRU` | selfplay | 两层 GRU + 21 维 self-play observation |

> 2026-10-30/31 新增的 `rl_v5_ft` / `rl_v5_ft_complex` /
> `rl_probe_guard` / `rl_defection_guard` / `rl_full_guard` /
> `rl_full_guard_c` 均只支持 `sample` 模式，并已与 pri_dile 对应候选
> 在 3 个 seed × 4 组固定对手序列上做逐动作 parity。

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

- 每个导出文件都与原始 pri_dile 候选做过逐动作 parity；`rl_full_guard_c.py`
  在 3 个 seed × 4 组固定对手序列上与 pri_dile `rl-full-guard-c` 逐动作一致；
- 标准 3/0/5/1 矩阵下覆盖 224 个非 long-run Axelrod 策略、100 回合对局；
- `T=10, R=6, P=1, S=0` 矩阵下覆盖 213 个可构造的 Axelrod 策略；其余 11 个是 Axelrod 内置零行列式策略，它们的 four-vector 按标准 3/0/5/1 校准，在自定义矩阵下会直接拒绝构造，这是 Axelrod 对手库的限制；
- `GRU`、`GRU-DAgger`、`Adv-GRU`、`SelfPlay-GRU` 会在每场 `reset()` 时重建隐状态；
- `RL-Gate` 在 `t=10` 时读取前 10 回合并锁定一个专家；
- `RL-Switch` 在前 6 回合检测到“双方都背叛过至少 2 次、对手最新一手合作”后切换到 RL-Repair；
- `RL-FullGuard-C` 在对手首次 D 于零基 5/7（probe）或 ≥9（late）、连续 3 回合 D 且我方未主动 D 时触发；probe 响应 `C,C,C`，late 响应 `D,C,C`，对手回 C 立即释放；
- `RL-ProbeGuard` 只看 probe 窗口（两回合确认）；`RL-DefectionGuard` 只看 late 窗口（三回合确认）；`RL-FullGuard` 是两者的 V5-FT 基座版本。

## 限制

- 策略按 **100 回合**设计；请使用 `turns=100`，不要直接用于其他长度。
- `RL-FullGuard-C` 的 probe/late 窗口按零基回合计；改为其他回合数会改变触发位置。
- 开发/测试收益矩阵固定为 `T=10, R=6, P=1, S=0`；不要用 Axelrod 默认矩阵比较分数。
- `sample` 模式是随机策略；同一 seed 下可复现，不同 seed 会有动作差异。
- `SelfPlay-GRU` 使用独立的 21 维 observation，不共享 85 维 neural features。
- 本仓库只包含推理代码和权重，不包含 pri_dile 的训练/评估工具。

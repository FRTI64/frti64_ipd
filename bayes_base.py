#!/usr/bin/env python3
"""Bayes-base as a standalone Axelrod strategy.

Faithful single-file port of the pri_dile ``bayes-base`` candidate and the
BayesPlus model bank.  The strategy is defined for 100-round matches.

Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
Use the module-level ``GAME`` with axelrod Match/Tournament.
"""
from __future__ import annotations

import math
import random

from axelrod import Action, Game, Player

# Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
GAME = Game(r=6, s=0, t=10, p=1)

# Internal action convention used by the original pri_dile policies.
C = 0
D = 1

# Axelrod action objects.
A_C = Action.C
A_D = Action.D


def _histories(player, opponent):
    mine = [D if move == A_D else C for move in player.history]
    theirs = [D if move == A_D else C for move in opponent.history]
    if len(mine) != len(theirs):
        raise RuntimeError("history lengths differ")
    return mine, theirs


def _require_sample(mode, name):
    if mode != "sample":
        raise ValueError(f"{name} supports only 'sample' mode")

def _payoff(me: int, them: int) -> int:
    """pri_dile payoff matrix: CC=6, CD=0, DC=10, DD=1."""
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1
def p_allc(t, mine, theirs):
    return 0.0


def p_alld(t, mine, theirs):
    return 1.0


def p_tft(t, mine, theirs):
    return 0.0 if t == 0 else float(mine[t - 1] == D)


def p_stft(t, mine, theirs):
    return 1.0 if t == 0 else float(mine[t - 1] == D)


def p_grim(t, mine, theirs):
    return float(D in mine[:t])


def p_tft2(t, mine, theirs):
    if t == 0:
        return 0.0
    return float(t >= 2 and mine[t - 1] == D and mine[t - 2] == D)


def p_gtft(t, mine, theirs, g=0.1):
    if t == 0:
        return 0.0
    if mine[t - 1] == C:
        return 0.0
    return 1.0 - g


def p_joss(t, mine, theirs, p=0.05):
    if t == 0:
        return 0.0
    if mine[t - 1] == D:
        return 1.0
    return p


def p_random(t, mine, theirs, p=0.5):
    return p


def p_alternator(t, mine, theirs):
    return float(t % 2 == 1)


def p_pavlov(t, mine, theirs):
    if t == 0:
        return 0.0
    last_opp = theirs[t - 1]
    last_mine = mine[t - 1]
    opp_payoff = _payoff(last_opp, last_mine)
    return float(last_opp if opp_payoff >= 3 else 1 - last_opp)


def p_lated(t, mine, theirs, k=10):
    return float(t >= 100 - k)


def p_noisy_tft(t, mine, theirs, p=0.05):
    if t == 0:
        return 0.0
    base = mine[t - 1]
    return p if base == C else 1.0 - p


def p_tft_delay(t, mine, theirs, delay=2):
    if t < delay:
        return 0.0
    return float(mine[t - delay] == D)


def p_tft_lastd(t, mine, theirs, k=1):
    if t >= 100 - k:
        return 1.0
    return p_tft(t, mine, theirs)


def p_grudger(t, mine, theirs, k=1):
    return float(mine[:t].count(D) >= k)


def p_periodic(t, mine, theirs, pattern=(C, C, D)):
    return float(pattern[t % len(pattern)] == D)


def p_earlyd(t, mine, theirs, k=10):
    return float(t < k)


def p_allc_lastd(t, mine, theirs):
    return float(t >= 99)


def p_tester(t, mine, theirs):
    if t == 0:
        return 1.0
    if mine[0] == D:
        return float(mine[t - 1] == D)
    return float((t - 1) % 4 < 2)


def p_prober(t, mine, theirs):
    if t < 3:
        return (0.0, 0.0, 1.0)[t]
    if D in mine[:3]:
        return float(mine[t - 1] == D)
    return float((t - 3) % 4 < 2)


def p_remorseful_prober(t, mine, theirs):
    if t == 0:
        return 1.0
    mode = "tft" if mine[0] == D else "exploit"
    action = C
    for s in range(1, t + 1):
        if mode == "exploit":
            if mine[s - 1] == D:
                mode = "tft"
                action = C
            else:
                action = D
        else:
            action = mine[s - 1]
    return float(action == D)


class BayesPlusMachine:
    name = "BayesPlus"
    NONRESP_THRESHOLD = 0.40
    PUNITIVE_THRESHOLD = 0.55

    def __init__(self, seed: int = 20261024):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.mode = "nice"
        self.tests = 0
        self.last_test = -100
        self.pattern = False

        self.models = []
        self._add("AllC", 8.0, p_allc)
        self._add("AllD", 4.0, p_alld)
        self._add("TFT", 18.0, p_tft)
        self._add("sTFT", 3.0, p_stft)
        self._add("Grim", 5.0, p_grim)
        self._add("TFT2", 8.0, p_tft2)
        self._add("GTFT.1", 8.0, lambda t, m, o: p_gtft(t, m, o, .1))
        self._add("GTFT.25", 3.0, lambda t, m, o: p_gtft(t, m, o, .25))
        self._add("Joss.05", 4.0, lambda t, m, o: p_joss(t, m, o, .05))
        self._add("Joss.15", 2.0, lambda t, m, o: p_joss(t, m, o, .15))
        self._add("Random.2", 3.0, lambda t, m, o: p_random(t, m, o, .2))
        self._add("Random.5", 3.0, lambda t, m, o: p_random(t, m, o, .5))
        self._add("Random.8", 1.0, lambda t, m, o: p_random(t, m, o, .8))
        self._add("Pavlov", 4.0, p_pavlov)
        self._add("Alternator", 1.0, p_alternator)
        for k in (5, 10, 15, 20, 25, 30, 40, 50):
            self._add(f"LateD{k}", 3.0,
                      lambda t, m, o, k=k: p_lated(t, m, o, k))
        self._names = [x[0] for x in self.models]
        self._priors = [x[1] for x in self.models]
        self._fns = [x[2] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0

        # BayesPlusMachine extensions.
        for k in range(1, 100):
            self._add(f"LateDx{k}", 0.16,
                      lambda t, m, o, k=k: p_lated(t, m, o, k))
        for k in (5, 10, 20, 33, 50):
            self._add(f"GTFTx{k}", 1.5,
                      lambda t, m, o, k=k: p_gtft(t, m, o, k / 100.0))
        for p in (0.02, 0.10, 0.20):
            self._add(f"NoisyTFTx{p}", 1.5,
                      lambda t, m, o, p=p: p_noisy_tft(t, m, o, p))
        for delay in (1, 2, 3):
            self._add(f"TFTdelay{delay}", 1.5,
                      lambda t, m, o, d=delay: p_tft_delay(t, m, o, d))
        for k in (1, 2, 3, 5, 10):
            self._add(f"TFTlastD{k}", 1.0,
                      lambda t, m, o, k=k: p_tft_lastd(t, m, o, k))
        for k in (1, 2, 3):
            self._add(f"Grudger{k}", 2.0,
                      lambda t, m, o, k=k: p_grudger(t, m, o, k))
        self._add("Mistrust", 2.0, lambda t, m, o: p_grudger(t, m, o, 2))
        self._add("Periodic3", 1.0,
                  lambda t, m, o: p_periodic(t, m, o, (C, C, D)))
        self._add("Periodic4", 1.0,
                  lambda t, m, o: p_periodic(t, m, o, (C, D, D, C)))
        self._add("AlternatorD", 1.0,
                  lambda t, m, o: p_periodic(t, m, o, (D, C)))
        self._add("Tester", 2.0, p_tester)
        self._add("Prober", 2.0, p_prober)
        self._add("RemorsefulProber", 1.5, p_remorseful_prober)
        self._add("AllC-lastD", 1.0, p_allc_lastd)
        self._add("EarlyD10", 1.0,
                  lambda t, m, o: p_earlyd(t, m, o, 10))
        self._add("EarlyD25", 0.5,
                  lambda t, m, o: p_earlyd(t, m, o, 25))
        self._names = [x[0] for x in self.models]
        self._priors = [x[1] for x in self.models]
        self._fns = [x[2] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0

    def _add(self, name, prior, fn):
        self.models.append((name, prior, fn))

    def _posterior(self, mine, theirs):
        for t in range(self._updated, len(theirs)):
            y = theirs[t]
            for i, fn in enumerate(self._fns):
                p = max(1e-6, min(1 - 1e-6, fn(t, mine, theirs)))
                self._logw[i] += math.log(p if y == D else 1.0 - p)
        self._updated = len(theirs)
        mx = max(self._logw)
        ws = [math.exp(x - mx) for x in self._logw]
        z = sum(ws)
        return {name: w / z for name, w in zip(self._names, ws)}

    def act(self, t, mine, theirs):
        return self.move(t + 1, mine, theirs)

    def move(self, day, mine, theirs):
        if day >= 100:
            return D
        if day == 1:
            return C
        if D not in theirs:
            return C

        post = self._posterior(mine, theirs)
        resp = 0.0
        nonresp = 0.0
        punitive = 0.0
        for name, p in post.items():
            if name.startswith(("TFT", "sTFT", "GTFT", "Joss",
                                "NoisyTFT", "Pavlov")):
                resp += p
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                punitive += p
            else:
                nonresp += p

        if (nonresp >= self.NONRESP_THRESHOLD
                or punitive >= self.PUNITIVE_THRESHOLD):
            return D
        if resp >= 0.50:
            return C
        if theirs[-1] == D and len(theirs) >= 2 and theirs[-2] == D:
            return D
        return C

class BayesBase(Player):
    """Axelrod Player around the base BayesPlus model bank."""

    name = "Bayes-base"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, seed: int = 20261024, mode: str = "sample",
                 name: str | None = None):
        super().__init__()
        _require_sample(mode, "Bayes-base")
        self.seed = int(seed)
        self.mode = mode
        if name is not None:
            self.name = str(name)
        self.policy = BayesPlusMachine(seed=self.seed)

    def strategy(self, opponent: Player) -> Action:
        mine, theirs = _histories(self, opponent)
        action = self.policy.act(len(theirs), mine, theirs)
        return A_D if action == D else A_C

    def __repr__(self) -> str:
        return self.name


def build(seed: int = 20261024, mode: str = "sample",
          name: str | None = None, **_: object) -> BayesBase:
    return BayesBase(seed=seed, mode=mode, name=name)


__all__ = ["BayesBase", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((BayesBase(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{BayesBase.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")

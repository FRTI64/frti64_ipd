#!/usr/bin/env python3
"""Triple-v4 as a standalone Axelrod strategy.

This is a faithful single-file port of the pri_dile candidate
``ipd/policies/candidates/triple_v4.py`` and the minimum support code it
needs:

    * ``ipd/core/machine.py::Machine``
    * ``ipd/core/opponents/reciprocal.py::Responsive3``
    * ``ipd/core/bayes.py`` and ``ipd/core/bayes_plus.py``

The strategy fuses Final, Responsive3 and BayesPlus by majority vote.  It
assumes the same 100-round match horizon as the source candidate (defection on
day 100, late-harvester models relative to day 100).

Usage::

    from triple_v4 import TripleV4

    player = TripleV4()
    match = axelrod.Match((player, axelrod.TitForTat()), turns=100)
    match.play()

Only the ``axelrod`` package and the Python standard library are required.

Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
Use the module-level ``GAME`` with axelrod Match/Tournament.
"""
from __future__ import annotations

import math
import random

from axelrod import Action, Game, Player

# Internal action convention used by the original policy code.
C = 0
D = 1

# Axelrod action objects.
A_C = Action.C
A_D = Action.D

# Development/evaluation payoff matrix: T=10, R=6, P=1, S=0.
GAME = Game(r=6, s=0, t=10, p=1)

DISPLAY = "Triple-v4"
SEED = 20261027

FINAL_TUNED = {
    "P_UNPROVOKED": 0.2395699546614404,
    "P_MUTUAL": 0.6399155368157192,
    "P_RESPONSIVE": 0.9544467968114831,
    "MIN_C": 5,
    "MIN_D": 2,
    "RESP_GAP": 0.3871088255996472,
    "DEFECT_RATE": 0.1170465108152342,
    "PEACE_TESTS": 1,
    "PEACE_COOLDOWN": 9,
}

RESP_TUNED = {
    "g_unprov": 0.31711381663242244,
    "g_mutual": 0.26173130084075913,
    "g_resp": 0.9804446329081643,
    "min_c": 5,
    "min_d": 2,
    "resp": 0.46788262082576104,
    "rate": 0.197064176521839,
    "max_tests": 3,
    "cooldown": 14,
}

BAYES_TUNED = {
    "coop_scale": 1.8369823500713045,
    "punitive_scale": 2.4705958305576154,
    "late_scale": 0.5216017158776368,
    "noise_scale": 0.5817914534720606,
    "probe_scale": 3.315461421931959,
    "nonresp": 0.27974873408266854,
    "punitive_thr": 0.7531347429866336,
}

# Noise-tolerant branch thresholds.
NOISE_PD = 0.70
NOISE_PC_MIN = 0.20
NOISE_PC_MAX = 0.62
NOISE_GAP = 0.30
NOISE_RATE = 0.15
NOISE_FORGIVE = 0.75


def _payoff(me: int, them: int) -> int:
    """pri_dile payoff matrix: CC=6, CD=0, DC=10, DD=1."""
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


# ---------------------------------------------------------------------------
# Final state machine (ipd/core/machine.py)
# ---------------------------------------------------------------------------


class Machine:
    """Final adaptive strategy for a 100-round prisoner's dilemma."""

    P_UNPROVOKED = 0.25
    P_MUTUAL = 0.45
    P_RESPONSIVE = 0.95
    MIN_C = 5
    MIN_D = 3
    RESP_GAP = 0.38
    DEFECT_RATE = 0.18
    PEACE_TESTS = 2
    PEACE_COOLDOWN = 8

    def __init__(self, seed: int = 20261024):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.mode = "nice"
        self.tests = 0
        self.last_test = -100
        self.pattern = False

    @staticmethod
    def _stats(mine: list[int], theirs: list[int]):
        n_c = n_d = d_c = d_d = 0
        for i in range(1, len(theirs)):
            if mine[i - 1] == C:
                n_c += 1
                d_c += theirs[i] == D
            else:
                n_d += 1
                d_d += theirs[i] == D
        return n_c, n_d, d_c, d_d

    @staticmethod
    def _alternating(moves: list[int], run: int = 10) -> bool:
        if len(moves) < run:
            return False
        return all(moves[i] != moves[i - 1]
                   for i in range(len(moves) - run + 1, len(moves)))

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        if day >= 100:
            return D
        if day == 1:
            return C

        n_c, n_d, d_c, d_d = self._stats(mine, theirs)
        p_c = d_c / n_c if n_c else 0.0
        p_d = d_d / n_d if n_d else 0.0
        d_rate = sum(theirs) / len(theirs)

        if self.pattern:
            if self._alternating(theirs):
                return D
            self.pattern = False
        elif self._alternating(theirs):
            self.pattern = True
            return D

        if self.mode == "test1":
            self.mode = "test2"
            return C
        if self.mode == "test2":
            self.mode = "test3"
            return C
        if self.mode == "test3":
            if theirs[-1] == C:
                self.mode = "nice"
                return C
            self.mode = "punish"
            self.last_test = day
            return D

        enough = n_c >= self.MIN_C and n_d >= self.MIN_D
        responsive = (enough and p_c <= 0.35 and p_d >= 0.65
                      and p_d - p_c >= self.RESP_GAP)
        nonresponsive = (enough and p_d - p_c < 0.35
                         and d_rate >= self.DEFECT_RATE)

        if self.mode == "nice":
            if nonresponsive:
                self.mode = "punish"
                return D

            if theirs[-1] == C:
                return C

            if mine[-1] == C:
                p_forgive = (self.P_RESPONSIVE if responsive
                             else self.P_UNPROVOKED)
            else:
                p_forgive = (self.P_RESPONSIVE if responsive
                             else self.P_MUTUAL)
            return C if self.rng.random() < p_forgive else D

        if (self.tests < self.PEACE_TESTS
                and day - self.last_test >= self.PEACE_COOLDOWN
                and enough and p_c <= 0.35 and p_d >= 0.65
                and p_d - p_c >= 0.40):
            self.tests += 1
            self.last_test = day
            self.mode = "test1"
            return C
        return D


class TolerantFinal(Machine):
    """Final state machine with a narrow noisy-reciprocator branch."""

    name = "Triple-v4-Final"

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        if day >= 100:
            return D
        if day == 1:
            return C
        n_c, n_d, d_c, d_d = self._stats(mine, theirs)
        p_c = d_c / n_c if n_c else 0.0
        p_d = d_d / n_d if n_d else 0.0
        d_rate = sum(theirs) / len(theirs)
        if (n_c >= self.MIN_C and n_d >= self.MIN_D
                and p_d >= NOISE_PD
                and NOISE_PC_MIN <= p_c <= NOISE_PC_MAX
                and p_d - p_c >= NOISE_GAP
                and d_rate >= NOISE_RATE):
            if theirs[-1] == C:
                return C
            return C if self.rng.random() < NOISE_FORGIVE else D
        return super().move(day, mine, theirs)


# ---------------------------------------------------------------------------
# Responsive3 (ipd/core/opponents/reciprocal.py)
# ---------------------------------------------------------------------------


class Responsive3:
    """Responsive strategy with adaptive forgiveness."""

    def __init__(self, g_unprov=0.55, g_mutual=0.40, g_resp=0.95,
                 min_c=5, min_d=3, resp=0.38, rate=0.18,
                 last_d=True, max_tests=2, cooldown=8, rng=None):
        self.g_unprov = g_unprov
        self.g_mutual = g_mutual
        self.g_resp = g_resp
        self.min_c = min_c
        self.min_d = min_d
        self.resp = resp
        # Upstream quirk preserved for exact parity: Responsive3 accepts a
        # ``rate`` argument but its Responsive base constructor is called
        # without it, so the class always uses the base default 0.18.
        self.rate = 0.18
        self.last_d = last_d
        self.max_tests = max_tests
        self.cooldown = cooldown
        self.rng = rng if rng is not None else random.Random()
        self.reset()
        self.name = "Resp3"

    def reset(self) -> None:
        self.mode = "nice"
        self.tests = 0
        self.last_test = -100
        self.pattern_mode = False

    @staticmethod
    def stats(mine: list[int], theirs: list[int]):
        n_c = n_d = d_c = d_d = 0
        for i in range(1, len(theirs)):
            if mine[i - 1] == C:
                n_c += 1
                d_c += theirs[i] == D
            else:
                n_d += 1
                d_d += theirs[i] == D
        return n_c, n_d, d_c, d_d

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        if self.last_d and t == 99:
            return D
        if t == 0:
            return C

        n_c, n_d, d_c, d_d = self.stats(mine, theirs)
        p_c = d_c / n_c if n_c else 0.0
        p_d = d_d / n_d if n_d else 0.0
        rate = sum(theirs) / len(theirs)

        if self.pattern_mode:
            if (len(theirs) < 10 or not all(theirs[i] != theirs[i - 1]
                    for i in range(len(theirs) - 9, len(theirs)))):
                self.pattern_mode = False
            else:
                return D
        elif len(theirs) >= 10 and all(theirs[i] != theirs[i - 1]
                for i in range(len(theirs) - 9, len(theirs))):
            self.pattern_mode = True
            return D

        if self.mode == "test1":
            self.mode = "test2"
            return C
        if self.mode == "test2":
            self.mode = "test3"
            return C
        if self.mode == "test3":
            if theirs[-1] == C:
                self.mode = "nice"
                return C
            self.mode = "punish"
            self.last_test = t
            return D

        if self.mode == "nice":
            enough = n_c >= self.min_c and n_d >= self.min_d
            responsive = (enough and p_c <= 0.35 and p_d >= 0.65
                          and p_d - p_c >= self.resp)
            nonresponsive = (enough and (p_d - p_c) < 0.35
                             and rate >= self.rate)
            if nonresponsive:
                self.mode = "punish"
                return D
            if theirs[-1] == D:
                if mine[-1] == C:
                    g = self.g_resp if responsive else self.g_unprov
                else:
                    g = self.g_resp if responsive else self.g_mutual
                return C if self.rng.random() < g else D
            return C

        if (self.tests < self.max_tests and t - self.last_test >= self.cooldown
                and n_c >= self.min_c and n_d >= 2
                and p_c <= 0.35 and p_d >= 0.65 and p_d - p_c >= 0.40):
            self.tests += 1
            self.last_test = t
            self.mode = "test1"
            return C
        return D


# ---------------------------------------------------------------------------
# Bayes model bank (ipd/core/bayes.py, ipd/core/bayes_plus.py)
# ---------------------------------------------------------------------------


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


class _TunedBayes(BayesPlusMachine):
    name = "Triple-v4-Bayes"

    def reset(self) -> None:
        super().reset()
        for i, (name, prior, fn) in enumerate(self.models):
            if name.startswith("LateD"):
                scale = BAYES_TUNED["late_scale"]
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                scale = BAYES_TUNED["punitive_scale"]
            elif name.startswith(("Joss", "NoisyTFT", "Random")):
                scale = BAYES_TUNED["noise_scale"]
            elif name.startswith(("Tester", "Prober", "Remorseful")):
                scale = BAYES_TUNED["probe_scale"]
            else:
                scale = BAYES_TUNED["coop_scale"]
            self.models[i] = (name, prior * scale, fn)
        self._priors = [x[1] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0
        self.NONRESP_THRESHOLD = BAYES_TUNED["nonresp"]
        self.PUNITIVE_THRESHOLD = BAYES_TUNED["punitive_thr"]


# ---------------------------------------------------------------------------
# Fused policy and Axelrod player
# ---------------------------------------------------------------------------


class TripleV4Policy:
    """Final, Responsive3 and BayesPlus fused by majority voting."""

    name = DISPLAY

    def __init__(self, seed: int = SEED):
        self.f = TolerantFinal(seed=20261024)
        for key, value in FINAL_TUNED.items():
            setattr(self.f, key, value)
        self.r = Responsive3(rng=random.Random(20261024), **RESP_TUNED)
        self.b = _TunedBayes()
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.f.reset()
        self.r.reset()
        self.b.reset()

    def groups(self, mine: list[int], theirs: list[int]):
        post = self.b._posterior(mine, theirs)
        resp = nonresp = punitive = 0.0
        for name, p in post.items():
            if name.startswith(("TFT", "sTFT", "GTFT", "Joss", "NoisyTFT",
                                "Pavlov")):
                resp += p
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                punitive += p
            else:
                nonresp += p
        return resp, nonresp, punitive, max(post.values())

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        a = self.f.move(t + 1, mine, theirs)
        b = self.r.act(t, mine, theirs)
        c = self.b.act(t, mine, theirs)
        return C if (a + b + c) <= 1 else D


class TripleV4(Player):
    """Axelrod ``Player`` wrapper around :class:`TripleV4Policy`."""

    name = DISPLAY
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, seed: int = SEED, name: str | None = None):
        super().__init__()
        self.seed = int(seed)
        if name is not None:
            self.name = str(name)
        self.policy = TripleV4Policy(seed=self.seed)

    def strategy(self, opponent: Player) -> Action:
        mine = [D if move == A_D else C for move in self.history]
        theirs = [D if move == A_D else C for move in opponent.history]
        if len(mine) != len(theirs):
            raise RuntimeError("history lengths differ")
        action = self.policy.act(len(theirs), mine, theirs)
        return A_D if action == D else A_C

    def __repr__(self) -> str:
        return self.name


def build(seed: int = SEED, name: str | None = None,
          **_: object) -> TripleV4:
    """Compatibility helper mirroring the candidate ``build`` function."""
    return TripleV4(seed=seed, name=name)


__all__ = ["TripleV4", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((TripleV4(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{TripleV4.name:10} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")

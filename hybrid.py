#!/usr/bin/env python3
"""Final+Resp3 (hybrid) as a standalone Axelrod strategy.

Faithful single-file port of the pri_dile ``hybrid`` candidate.  Final uses
the tuned thresholds, Responsive3 uses its defaults, and disagreements are
resolved by a p=0.0 cooperative coin (always defect).  Defined for 100-round
matches.

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

    def act(self, t, mine, theirs):
        return self.move(t + 1, mine, theirs)
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


class _HybridPolicy:
    def __init__(self, seed: int = 20261025):
        self.f = Machine(seed=20261024)
        for key, value in FINAL_TUNED.items():
            setattr(self.f, key, value)
        self.r = Responsive3(rng=random.Random(20261024))
        self.p = 0.0
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.f.reset()
        self.r.reset()

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        a = self.f.move(t + 1, mine, theirs)
        b = self.r.act(t, mine, theirs)
        if a == b:
            return a
        return C if self.rng.random() < self.p else D


class Hybrid(Player):
    """Axelrod Player around the Final+Responsive3 hybrid."""

    name = "Final+Resp3"
    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, seed: int = 20261025, mode: str = "sample",
                 name: str | None = None):
        super().__init__()
        _require_sample(mode, "Final+Resp3")
        self.seed = int(seed)
        self.mode = mode
        if name is not None:
            self.name = str(name)
        self.policy = _HybridPolicy(seed=self.seed)

    def strategy(self, opponent: Player) -> Action:
        mine, theirs = _histories(self, opponent)
        action = self.policy.act(len(theirs), mine, theirs)
        return A_D if action == D else A_C

    def __repr__(self) -> str:
        return self.name


def build(seed: int = 20261025, mode: str = "sample",
          name: str | None = None, **_: object) -> Hybrid:
    return Hybrid(seed=seed, mode=mode, name=name)


__all__ = ["Hybrid", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((Hybrid(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{Hybrid.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")

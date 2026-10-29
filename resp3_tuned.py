#!/usr/bin/env python3
"""Resp3-tuned as a standalone Axelrod strategy.

Faithful single-file port of the pri_dile ``resp3-tuned`` candidate.  The
tuned Responsive3 parameters are embedded; the strategy is defined for
100-round matches.

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
G_UNPROV = 0.31711381663242244
G_MUTUAL = 0.26173130084075913
G_RESP = 0.9804446329081643
MIN_C = 5
MIN_D = 2
RESP = 0.46788262082576104
RATE = 0.197064176521839
MAX_TESTS = 3
COOLDOWN = 14

class Resp3Tuned(Player):
    """Axelrod Player around the tuned Responsive3 policy."""

    name = "Resp3-tuned"
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
        _require_sample(mode, "Resp3-tuned")
        self.seed = int(seed)
        self.mode = mode
        if name is not None:
            self.name = str(name)
        self.policy = Responsive3(g_unprov=G_UNPROV, g_mutual=G_MUTUAL, g_resp=G_RESP, min_c=MIN_C, min_d=MIN_D, resp=RESP, rate=RATE, max_tests=MAX_TESTS, cooldown=COOLDOWN, rng=random.Random(self.seed))

    def strategy(self, opponent: Player) -> Action:
        mine, theirs = _histories(self, opponent)
        action = self.policy.act(len(theirs), mine, theirs)
        return A_D if action == D else A_C

    def __repr__(self) -> str:
        return self.name


def build(seed: int = 20261024, mode: str = "sample",
          name: str | None = None, **_: object) -> Resp3Tuned:
    return Resp3Tuned(seed=seed, mode=mode, name=name)


__all__ = ["Resp3Tuned", "build", "GAME"]


if __name__ == "__main__":
    import axelrod as axl

    for opponent in (axl.Cooperator(), axl.Defector(), axl.TitForTat(),
                     axl.Grudger()):
        match = axl.Match((Resp3Tuned(), opponent), turns=100, game=GAME)
        match.play()
        scores = match.final_score()
        print(f"{Resp3Tuned.name:12} vs {opponent.name:15} "
              f"{scores[0]:4d} - {scores[1]:4d}")

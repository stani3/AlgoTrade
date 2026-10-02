"""What a frozen strategy wants after each closed bar, decided by our own strategy code.

No NautilusTrader here: the adapter feeds closed bars and fills in, and turns the decisions
into orders. The rules mirror the research engines so a live run can be checked against them:

* Position strategies (``target_position``): after each close, the wanted exposure is the
  strategy's last target times the stake; the adapter rebalances to it.
* Bracket strategies (``signals``): when flat, not killed and out of cooldown, a signal at the
  close asks for a market entry; stop and target are placed from the actual fill price with the
  signal bar's distances. Cooldowns follow the last trade's P&L after costs, a trade still open
  after ``max_bars`` bars is closed, and the kill switch stops trading for good once equity is
  ``kill_drawdown`` below its peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from algotrade.strategies import BracketStrategy, Strategy, from_spec

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "funding_rate"]


@dataclass(frozen=True)
class Entry:
    direction: int  # +1 long, -1 short
    stop_distance: float
    target_distance: float


@dataclass(frozen=True)
class Decision:
    """``target`` for position strategies; ``entry`` / ``close`` for bracket strategies."""

    target: float | None = None
    entry: Entry | None = None
    close: str | None = None  # why an open bracket trade should be closed (time / kill)


@dataclass
class BracketState:
    in_trade: bool = False
    entry_bar: int = -1
    last_exit_bar: int = -(10**9)
    cooldown: int = 0
    peak: float = 0.0
    killed: bool = False


@dataclass
class Decider:
    spec: dict
    stake: float = 1.0
    history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BAR_COLUMNS))
    max_history: int = 5000

    def __post_init__(self) -> None:
        self.strategy: Strategy = from_spec(self.spec)
        self.state = BracketState()
        self.history = self.history.reindex(columns=BAR_COLUMNS).astype("float64")
        self.history["funding_rate"] = self.history["funding_rate"].fillna(0.0)

    @property
    def bracket(self) -> bool:
        return isinstance(self.strategy, BracketStrategy)

    @property
    def bar_number(self) -> int:
        return len(self.history) - 1

    def add_bar(self, time: pd.Timestamp, open_: float, high: float, low: float, close: float,
                volume: float = 0.0, funding_rate: float = 0.0) -> None:  # fmt: skip
        """Append a closed bar (indexed by its open time, like the research data)."""

        if len(self.history) and time <= self.history.index[-1]:
            raise ValueError(f"bar {time} is not after the last bar {self.history.index[-1]}")
        row = pd.DataFrame([[open_, high, low, close, volume, funding_rate]], columns=BAR_COLUMNS,
                           index=pd.DatetimeIndex([time]))  # fmt: skip
        self.history = pd.concat([self.history, row]).iloc[-self.max_history :]

    def decide(self, equity: float | None = None) -> Decision:
        """The decision at the close of the last bar added."""

        if not self.bracket:
            target = float(self.strategy.target_position(self.history).iloc[-1])
            return Decision(target=0.0 if np.isnan(target) else target * self.stake)
        return self._bracket_decision(equity)

    def _bracket_decision(self, equity: float | None) -> Decision:
        strategy: BracketStrategy = self.strategy
        state = self.state
        if equity is not None:
            state.peak = max(state.peak, equity)
            if not state.killed and equity <= (1.0 - strategy.kill_drawdown) * state.peak:
                state.killed = True
                if state.in_trade:
                    return Decision(close="kill")
        if state.killed:
            return Decision()
        max_bars = getattr(strategy, "max_bars", 0)
        if state.in_trade:
            if max_bars and self.bar_number - state.entry_bar + 1 >= max_bars:
                return Decision(close="time")
            return Decision()
        if self.bar_number - state.last_exit_bar < state.cooldown:
            return Decision()
        signals = strategy.signals(self.history)
        stop = float(signals.stop_dist.iloc[-1])
        target = float(signals.target_dist.iloc[-1])
        if not (stop > 0 and target > 0):
            return Decision()
        if bool(signals.long.fillna(False).iloc[-1]):
            return Decision(entry=Entry(1, stop, target))
        if bool(signals.short.fillna(False).iloc[-1]):
            return Decision(entry=Entry(-1, stop, target))
        return Decision()

    # --- fills reported back by the adapter -----------------------------------------------

    def entered(self) -> None:
        """The entry filled at the open of the bar after the signal (the next bar added)."""

        self.state.in_trade = True
        self.state.entry_bar = self.bar_number + 1

    def exited(self, pnl_after_costs: float, at_close: bool = False) -> None:
        """The trade closed: a stop or target during the bar still forming (default), or a time
        exit decided at the last close (``at_close``). Starts the cooldown, like the simulator."""

        strategy: BracketStrategy = self.strategy
        self.state.in_trade = False
        self.state.last_exit_bar = self.bar_number if at_close else self.bar_number + 1
        self.state.cooldown = (
            strategy.cooldown_win if pnl_after_costs > 0 else strategy.cooldown_loss
        )

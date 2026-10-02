"""A NautilusTrader strategy that trades a frozen spec through :class:`~.decider.Decider`.

The trading logic is our own strategy code (via the decider); this class only turns its
decisions into orders:

* Position strategies: after each closed bar, rebalance to ``target x stake x capital share``
  of account equity with a market order (skipped below ``min_change`` of equity).
* Bracket strategies: a market entry; when it has filled, a reduce-only stop-market and a
  reduce-only limit target are placed from the actual fill price (as in our simulator). When
  the position closes, the other order is cancelled and the decider starts its cooldown. Time
  and kill-switch exits close with a reduce-only market order.

Every fill is appended to ``fills_path`` with the price the strategy expected (the decision
bar's close for market orders, the trigger or limit for stops and targets), which is what the
incubation report scores slippage on.

NautilusTrader's simulated venue does not enforce margin, so exposure is capped here; and its
bars carry no funding, so funding-based rules see zero funding when run through this adapter.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderFilled, PositionClosed
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from .decider import Decider

FILL_COLUMNS = ["ts", "symbol", "side", "quantity", "price", "expected_price"]


class SpecStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    spec: str  # the frozen spec as JSON
    stake: float = 1.0
    capital_share: float = 1.0  # share of the account's equity this strategy trades
    max_exposure: float = 2.0  # cap on |exposure| after the stake, as a fraction of its equity
    min_change: float = 0.0  # skip rebalances smaller than this fraction of its equity
    warmup_days: int = 0  # request this much history from the venue on start
    fills_path: str | None = None


class SpecStrategy(Strategy):
    def __init__(self, config: SpecStrategyConfig) -> None:
        super().__init__(config)
        self.decider = Decider(json.loads(config.spec), stake=config.stake)
        self.instrument = None
        self.bar_length = pd.Timedelta(config.bar_type.spec.timedelta)
        self.expected: dict[str, float] = {}  # client order id -> price the strategy expected
        self.pending_entry = None  # Entry waiting for its fill
        self.closing: str | None = None  # "time" / "kill" while a strategy exit is working
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    # --- lifecycle ---------------------------------------------------------------------------

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.config.warmup_days > 0:
            start = self.clock.utc_now() - pd.Timedelta(days=self.config.warmup_days)
            self.request_bars(self.config.bar_type, start=start)
        self.subscribe_bars(self.config.bar_type)

    def on_stop(self) -> None:
        """Positions are left as they are: stopping the node is not a trading decision."""

    def on_historical_data(self, data) -> None:
        if isinstance(data, Bar):
            self._add(data)

    # --- bars -----------------------------------------------------------------------------------

    def open_time(self, bar: Bar) -> pd.Timestamp:
        """Bars are stamped at (or just before) their close; the research data uses open times."""

        close = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC") - pd.Timedelta(1, "ns")
        return close.floor(self.bar_length)

    def _add(self, bar: Bar) -> bool:
        when = self.open_time(bar)
        history = self.decider.history
        if len(history) and when <= history.index[-1]:
            return False  # already seen (warm-up overlap)
        self.decider.add_bar(when, bar.open.as_double(), bar.high.as_double(),
                             bar.low.as_double(), bar.close.as_double(), bar.volume.as_double())  # fmt: skip
        return True

    def equity(self, bar: Bar) -> float:
        """This strategy's equity: its share of the account, marked to the bar's close."""

        account = self.portfolio.account(self.config.instrument_id.venue)
        currency = self.instrument.settlement_currency
        total = account.balance_total(currency).as_double()
        open_pnl = self.portfolio.unrealized_pnl(self.config.instrument_id, price=bar.close)
        total += open_pnl.as_double() if open_pnl is not None else 0.0
        return total * self.config.capital_share

    def net_quantity(self) -> float:
        return float(self.portfolio.net_position(self.config.instrument_id))

    def on_bar(self, bar: Bar) -> None:
        if not self._add(bar):
            return
        equity = self.equity(bar)
        self.equity_curve.append((pd.Timestamp(bar.ts_event, unit="ns", tz="UTC"), equity))
        if self.decider.bracket:
            self._bracket_bar(bar, equity)
        else:
            self._rebalance(bar, equity)

    # --- position strategies ------------------------------------------------------------------

    def _rebalance(self, bar: Bar, equity: float) -> None:
        cap = self.config.max_exposure
        target = max(-cap, min(cap, self.decider.decide().target))
        price = bar.close.as_double()
        delta = target * equity / price - self.net_quantity()
        if equity <= 0 or abs(delta) * price < self.config.min_change * equity:
            return
        self._market(delta, price, reduce_only=False, tag="rebalance")

    def _market(self, signed_qty: float, expected: float, reduce_only: bool, tag: str) -> None:
        if abs(signed_qty) < self.instrument.size_increment.as_double():
            return  # rounds to nothing
        quantity = self.instrument.make_qty(abs(signed_qty), round_down=True)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if signed_qty > 0 else OrderSide.SELL,
            quantity=quantity,
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only,
            tags=[tag],
        )
        self.expected[order.client_order_id.value] = expected
        self.submit_order(order)

    # --- bracket strategies --------------------------------------------------------------------

    def _bracket_bar(self, bar: Bar, equity: float) -> None:
        decision = self.decider.decide(equity=equity)
        price = bar.close.as_double()
        if decision.close and self.net_quantity():
            self.closing = decision.close
            self.cancel_all_orders(self.config.instrument_id)
            self._market(-self.net_quantity(), price, reduce_only=True, tag=decision.close)
        if decision.entry and not self.net_quantity():
            self.pending_entry = decision.entry
            size = self.config.stake * equity / price
            self._market(decision.entry.direction * size, price, reduce_only=False, tag="entry")

    def _place_bracket(self, direction: int, fill: float, quantity) -> None:
        entry, self.pending_entry = self.pending_entry, None
        exit_side = OrderSide.SELL if direction > 0 else OrderSide.BUY
        stop = self.instrument.make_price(fill - direction * entry.stop_distance)
        target = self.instrument.make_price(fill + direction * entry.target_distance)
        stop_order = self.order_factory.stop_market(
            instrument_id=self.config.instrument_id, order_side=exit_side, quantity=quantity,
            trigger_price=stop, time_in_force=TimeInForce.GTC, reduce_only=True, tags=["stop"],
        )  # fmt: skip
        target_order = self.order_factory.limit(
            instrument_id=self.config.instrument_id, order_side=exit_side, quantity=quantity,
            price=target, time_in_force=TimeInForce.GTC, post_only=False, reduce_only=True,
            tags=["target"],
        )  # fmt: skip
        self.expected[stop_order.client_order_id.value] = stop.as_double()
        self.expected[target_order.client_order_id.value] = target.as_double()
        self.submit_order(stop_order)
        self.submit_order(target_order)

    # --- events ---------------------------------------------------------------------------------

    def on_order_filled(self, event: OrderFilled) -> None:
        self._log_fill(event)
        order = self.cache.order(event.client_order_id)
        if "entry" in (order.tags or []) and order.is_closed and self.pending_entry is not None:
            direction = 1 if order.side == OrderSide.BUY else -1
            self.decider.entered()
            self._place_bracket(direction, float(order.avg_px), order.filled_qty)

    def on_position_closed(self, event: PositionClosed) -> None:
        if not self.decider.bracket:
            return
        self.cancel_all_orders(self.config.instrument_id)
        self.decider.exited(event.realized_pnl.as_double(), at_close=self.closing == "time")
        self.closing = None

    def _log_fill(self, event: OrderFilled) -> None:
        if not self.config.fills_path:
            return
        path = Path(self.config.fills_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(FILL_COLUMNS)
            writer.writerow(
                [
                    pd.Timestamp(event.ts_event, unit="ns", tz="UTC").isoformat(),
                    self.config.instrument_id.symbol.value,
                    1 if event.order_side == OrderSide.BUY else -1,
                    event.last_qty.as_double(),
                    event.last_px.as_double(),
                    self.expected.get(event.client_order_id.value, event.last_px.as_double()),
                ]
            )

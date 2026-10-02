"""Run a spec through NautilusTrader's backtest engine and our own, on the same bars.

Before a frozen strategy is paper traded, the adapter that will trade it must reproduce the
research engines: same positions, same trades, same equity. The comparison is made at zero
costs and funding (NautilusTrader models fees per fill and slippage in ticks, and does not
settle funding in backtests), on bars rounded to the instrument's precision.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

from algotrade.backtest.costs import ZERO_COSTS
from algotrade.backtest.engine import BacktestResult
from algotrade.backtest.runner import backtest
from algotrade.strategies import from_spec

from .adapter import SpecStrategy, SpecStrategyConfig

warnings.filterwarnings("ignore", message="Timestamp.utcnow is deprecated")

BAR_STEPS = {"1h": "1-HOUR", "4h": "4-HOUR", "1d": "1-DAY"}
PRECISION = 6


def instrument(
    symbol: str = "BTCUSDT", venue: str = "BINANCE", precision: int = PRECISION
) -> CryptoPerpetual:
    """A USDT-margined perpetual with fine increments and no fees (for parity runs)."""

    tick = Price(10.0**-precision, precision)
    step = Quantity(10.0**-precision, precision)
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(f"{symbol}-PERP"), Venue(venue)),
        raw_symbol=Symbol(symbol),
        base_currency=BTC,  # parity runs are synthetic: the base asset only labels the contract
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=precision,
        size_precision=precision,
        price_increment=tick,
        size_increment=step,
        ts_event=0,
        ts_init=0,
        max_quantity=Quantity(1e9, precision),
        min_quantity=step,
        max_price=Price(1e9, precision),
        min_price=tick,
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal(0),
        taker_fee=Decimal(0),
    )


def bar_type(instrument_id: InstrumentId, timeframe: str) -> BarType:
    return BarType.from_str(f"{instrument_id}-{BAR_STEPS[timeframe]}-LAST-EXTERNAL")


def to_bars(frame: pd.DataFrame, kind: BarType, perp: CryptoPerpetual) -> list[Bar]:
    """Our bars (indexed by open time) as Nautilus bars stamped at their close."""

    length = pd.Timedelta(kind.spec.timedelta)
    closes = (frame.index + length).as_unit("ns").asi8
    return [
        Bar(
            bar_type=kind,
            open=perp.make_price(o),
            high=perp.make_price(h),
            low=perp.make_price(lo),
            close=perp.make_price(c),
            volume=perp.make_qty(1e9),  # deep enough that no fill is capped by bar volume
            ts_event=int(t),
            ts_init=int(t),
        )
        for t, o, h, lo, c in zip(
            closes, frame["open"], frame["high"], frame["low"], frame["close"], strict=True
        )
    ]


def rounded(frame: pd.DataFrame, precision: int = PRECISION) -> pd.DataFrame:
    """Prices as the instrument will see them; zero funding (not settled by NautilusTrader)."""

    out = frame[["open", "high", "low", "close"]].round(precision).copy()
    out["volume"] = 0.0
    out["funding_rate"] = 0.0
    return out


@dataclass
class ParityRun:
    ours: BacktestResult
    equity: pd.Series  # NautilusTrader equity / starting capital, indexed by bar open time
    positions: pd.DataFrame  # closed (and open) positions from NautilusTrader
    fills: list = field(default_factory=list)

    def equity_gap(self) -> float:
        both = self.ours.equity.index.intersection(self.equity.index)
        return float(
            np.max(np.abs(self.ours.equity.loc[both].to_numpy() - self.equity.loc[both].to_numpy()))
        )

    def trades(self) -> pd.DataFrame:
        """NautilusTrader's round trips in research terms: entry and exit bar (open times)."""

        closed = (
            self.positions[self.positions["ts_closed"].notna()]
            if len(self.positions)
            else self.positions
        )
        if closed.empty:
            return pd.DataFrame(columns=["entry", "exit", "direction", "entry_price", "exit_price"])
        length = self.equity.index[1] - self.equity.index[0]
        unit = self.ours.ledger.index.unit  # same resolution as the research trades
        return pd.DataFrame(
            {
                "entry": pd.to_datetime(closed["ts_opened"], utc=True).dt.as_unit(unit),
                "exit": (pd.to_datetime(closed["ts_closed"], utc=True) - length).dt.as_unit(unit),
                "direction": np.where(closed["entry"] == "BUY", 1, -1),
                "entry_price": closed["avg_px_open"].astype(float).to_numpy(),
                "exit_price": closed["avg_px_close"].astype(float).to_numpy(),
            }
        ).sort_values("entry", ignore_index=True)


def run(
    spec: dict,
    frame: pd.DataFrame,
    timeframe: str,
    stake: float = 1.0,
    capital: float = 1_000_000.0,
    fills_path: str | None = None,
    min_change: float = 0.0,
) -> ParityRun:
    """Trade ``spec`` on ``frame`` with NautilusTrader (zero costs) and with our engine."""

    bars = rounded(frame)
    perp = instrument()
    kind = bar_type(perp.id, timeframe)
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("PARITY-001"),
            logging=LoggingConfig(log_level="ERROR", log_colors=False),
        )
    )
    engine.add_venue(
        venue=perp.id.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(capital, USDT)],
        base_currency=USDT,
        default_leverage=Decimal(10),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=0),
        bar_execution=True,
        bar_adaptive_high_low_ordering=False,
    )
    engine.add_instrument(perp)
    engine.add_data(to_bars(bars, kind, perp))
    strategy = SpecStrategy(
        SpecStrategyConfig(
            instrument_id=perp.id,
            bar_type=kind,
            spec=json.dumps(spec),
            stake=stake,
            max_exposure=max(1.0, stake),
            min_change=min_change,
            fills_path=fills_path,
        )
    )
    engine.add_strategy(strategy)
    engine.run()
    length = pd.Timedelta(kind.spec.timedelta)
    equity = (
        pd.Series(
            [value for _, value in strategy.equity_curve],
            index=pd.DatetimeIndex([when - length for when, _ in strategy.equity_curve]),
        )
        / capital
    )
    positions = engine.trader.generate_positions_report()
    engine.dispose()
    ours = backtest(from_spec(spec), bars, ZERO_COSTS, max_leverage=max(1.0, stake))
    return ParityRun(ours=ours, equity=equity, positions=positions)

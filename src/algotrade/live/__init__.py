"""Running frozen strategies outside the research backtests: NautilusTrader paper trading.

``decider`` holds the trading logic and never imports NautilusTrader, so it is tested like the
rest of the code; ``adapter`` turns its decisions into Nautilus orders. NautilusTrader is an
optional dependency (``pip install -e .[live]``).
"""

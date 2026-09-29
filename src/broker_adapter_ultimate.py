
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BrokerAdapter(ABC):
    """Broker-neutral interface.

    This project intentionally ships without a live order-submission implementation.
    """

    @abstractmethod
    def get_cash(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> pd.DataFrame:
        raise NotImplementedError

    def submit_orders(self, orders: pd.DataFrame):
        raise RuntimeError(
            "Live broker submission is disabled in this project. "
            "Use paper execution or export orders for manual broker review."
        )


class CSVReadOnlyBroker(BrokerAdapter):
    """
    Read-only broker snapshot adapter.

    Positions CSV accepted columns:
      ticker, qty, sellable_qty
    Optional:
      avg_cost, last_price, market_value
    """

    def __init__(self, positions_csv, cash):
        self.positions_csv = Path(positions_csv)
        self.cash = float(cash)

    def get_cash(self) -> float:
        return self.cash

    def get_positions(self) -> pd.DataFrame:
        df = pd.read_csv(self.positions_csv,dtype={"ticker":str})
        required = {"ticker", "qty", "sellable_qty"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Broker positions CSV missing columns: {sorted(missing)}")
        df["ticker"] = df["ticker"].astype(str)
        from .live_v8 import integer, require
        for col in ['qty','sellable_qty']:df[col]=df[col].map(lambda v:integer(v,col))
        require(not df['ticker'].duplicated().any(),'重复持仓')
        require((df['sellable_qty']<=df['qty']).all(),'可卖数量大于持仓')
        return df


class LiveBrokerDisabled(BrokerAdapter):
    """
    Explicit placeholder. Any attempt to submit real orders raises immediately.
    """

    def get_cash(self) -> float:
        raise RuntimeError("No live broker is configured.")

    def get_positions(self) -> pd.DataFrame:
        raise RuntimeError("No live broker is configured.")

    def submit_orders(self, orders: pd.DataFrame):
        raise RuntimeError(
            "LIVE ORDER SUBMISSION DISABLED. "
            "Implement a reviewed broker adapter separately and keep it disabled by default."
        )

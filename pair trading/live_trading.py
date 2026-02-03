"""
Live trading integration for pairs trading strategies.

Supports:
- Real-time price feeds
- Order placement
- Position management
- Risk monitoring
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PriceDataProvider(ABC):
    """Abstract base class for price data providers."""
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol."""
        pass
    
    @abstractmethod
    def get_bar_data(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Get latest bar data for symbol."""
        pass
    
    @abstractmethod
    def subscribe(self, symbols: List[str], callback) -> bool:
        """Subscribe to real-time price updates."""
        pass


class BrokerAPI(ABC):
    """Abstract broker API interface."""
    
    @abstractmethod
    def place_order(self, symbol: str, quantity: int, price: float,
                   order_type: OrderType) -> str:
        """Place an order. Returns order ID."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Dict:
        """Get current position for symbol."""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        pass


class MockPriceDataProvider(PriceDataProvider):
    """Mock price data provider for testing."""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.current_index = {}
        self.callbacks = {}
        
        for symbol in data:
            self.current_index[symbol] = 0
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price."""
        if symbol not in self.data:
            return None
        
        df = self.data[symbol]
        idx = min(self.current_index[symbol], len(df) - 1)
        return float(df.iloc[idx]['Close'])
    
    def get_bar_data(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Get latest bar."""
        if symbol not in self.data:
            return pd.DataFrame()
        
        df = self.data[symbol]
        idx = min(self.current_index[symbol], len(df) - 1)
        return df.iloc[:idx+1].copy()
    
    def subscribe(self, symbols: List[str], callback) -> bool:
        """Mock subscription."""
        for symbol in symbols:
            self.callbacks[symbol] = callback
        return True
    
    def advance_time(self, symbol: str):
        """Advance time for testing."""
        if symbol in self.current_index:
            self.current_index[symbol] += 1


class MockBrokerAPI(BrokerAPI):
    """Mock broker for paper trading."""
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        self.trade_history = []
        self.order_counter = 0
    
    def place_order(self, symbol: str, quantity: int, price: float,
                   order_type: OrderType = OrderType.MARKET) -> str:
        """Place an order."""
        order_id = f"ORD_{self.order_counter}"
        self.order_counter += 1
        
        cost = abs(quantity) * price * 1.001  # Add 0.1% commission
        
        if cost > self.balance and quantity > 0:
            logger.warning(f"Insufficient balance: {cost} > {self.balance}")
            self.orders[order_id] = {
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'status': OrderStatus.REJECTED,
                'timestamp': datetime.now()
            }
            return order_id
        
        # Execute order
        self.balance -= cost
        
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        
        pos = self.positions[symbol]
        if quantity > 0:
            pos['avg_price'] = (pos['avg_price'] * pos['quantity'] + price * quantity) / (pos['quantity'] + quantity)
        pos['quantity'] += quantity
        
        self.orders[order_id] = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'status': OrderStatus.FILLED,
            'timestamp': datetime.now()
        }
        
        self.trade_history.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Order {order_id}: {quantity} {symbol} @ {price}")
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order['status'] == OrderStatus.FILLED:
            return False  # Can't cancel filled order
        
        order['status'] = OrderStatus.CANCELLED
        return True
    
    def get_position(self, symbol: str) -> Dict:
        """Get position."""
        if symbol not in self.positions:
            return {'quantity': 0, 'avg_price': 0}
        return self.positions[symbol].copy()
    
    def get_account_balance(self) -> float:
        """Get balance."""
        return self.balance


class LiveTradingManager:
    """Manage live trading operations."""
    
    def __init__(self, price_provider: PriceDataProvider,
                broker_api: BrokerAPI,
                position_size: float = 0.1,  # Risk 10% per trade
                max_positions: int = 5,
                stop_loss_pct: float = 0.05):
        """
        Parameters:
        -----------
        price_provider : PriceDataProvider
            Source for price data
        broker_api : BrokerAPI
            Broker API
        position_size : float
            Position size as fraction of capital
        max_positions : int
            Maximum concurrent positions
        stop_loss_pct : float
            Stop loss as percentage
        """
        self.price_provider = price_provider
        self.broker_api = broker_api
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        
        self.active_positions: List[Dict] = []
        self.pending_signals: List[Dict] = []
        self.execution_log = []
    
    def calculate_position_size(self, account_balance: float,
                               entry_price: float) -> int:
        """Calculate position size based on account balance."""
        size = (account_balance * self.position_size) / entry_price
        return int(size)
    
    def validate_trade(self, signal: Dict, current_prices: Dict) -> bool:
        """Validate trade before execution."""
        # Check max positions
        if len(self.active_positions) >= self.max_positions:
            logger.warning("Max positions reached")
            return False
        
        # Check for conflicting positions
        symbol1, symbol2 = signal['symbol1'], signal['symbol2']
        for pos in self.active_positions:
            if pos['symbol1'] == symbol1 or pos['symbol1'] == symbol2 or \
               pos['symbol2'] == symbol1 or pos['symbol2'] == symbol2:
                logger.warning(f"Conflicting position with {symbol1}/{symbol2}")
                return False
        
        # Check prices are valid
        if symbol1 not in current_prices or symbol2 not in current_prices:
            logger.warning("Invalid prices")
            return False
        
        return True
    
    def execute_signal(self, signal: Dict) -> Tuple[bool, str]:
        """
        Execute a trading signal.
        
        Parameters:
        -----------
        signal : Dict
            Trading signal with keys: symbol1, symbol2, signal, hedge_ratio, strength
        
        Returns:
        --------
        Tuple of (success, message)
        """
        symbol1 = signal['symbol1']
        symbol2 = signal['symbol2']
        direction = signal.get('signal', 0)
        hedge_ratio = signal.get('hedge_ratio', 1.0)
        strength = signal.get('strength', 1)
        
        if direction == 0:
            # Close existing position
            return self._close_position(symbol1, symbol2)
        
        # Get current prices
        price1 = self.price_provider.get_latest_price(symbol1)
        price2 = self.price_provider.get_latest_price(symbol2)
        
        if price1 is None or price2 is None:
            return False, "Unable to get prices"
        
        # Validate
        if not self.validate_trade(signal, {symbol1: price1, symbol2: price2}):
            return False, "Trade validation failed"
        
        # Calculate sizes
        account_balance = self.broker_api.get_account_balance()
        qty1 = self.calculate_position_size(account_balance, price1)
        qty2 = int(qty1 * hedge_ratio)
        
        if qty1 == 0 or qty2 == 0:
            return False, "Invalid position size"
        
        # Place orders
        try:
            if direction == 1:  # Long pair1, short pair2
                order1_id = self.broker_api.place_order(symbol1, qty1, price1)
                order2_id = self.broker_api.place_order(symbol2, -qty2, price2)
            else:  # Short pair1, long pair2
                order1_id = self.broker_api.place_order(symbol1, -qty1, price1)
                order2_id = self.broker_api.place_order(symbol2, qty2, price2)
            
            # Record position
            position = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'direction': direction,
                'hedge_ratio': hedge_ratio,
                'quantity1': qty1,
                'quantity2': qty2,
                'entry_price1': price1,
                'entry_price2': price2,
                'entry_time': datetime.now(),
                'orders': [order1_id, order2_id],
                'strength': strength,
                'unrealized_pnl': 0
            }
            
            self.active_positions.append(position)
            
            msg = f"Opened {symbol1}/{symbol2} ({direction}) qty={qty1}/{qty2}"
            logger.info(msg)
            self.execution_log.append({
                'timestamp': datetime.now(),
                'action': 'OPEN',
                'message': msg
            })
            
            return True, msg
        
        except Exception as e:
            msg = f"Order execution failed: {str(e)}"
            logger.error(msg)
            return False, msg
    
    def _close_position(self, symbol1: str, symbol2: str) -> Tuple[bool, str]:
        """Close an existing position."""
        position = None
        pos_idx = None
        
        for i, pos in enumerate(self.active_positions):
            if pos['symbol1'] == symbol1 and pos['symbol2'] == symbol2:
                position = pos
                pos_idx = i
                break
        
        if position is None:
            return False, f"No position found for {symbol1}/{symbol2}"
        
        try:
            price1 = self.price_provider.get_latest_price(symbol1)
            price2 = self.price_provider.get_latest_price(symbol2)
            
            # Close positions
            self.broker_api.place_order(symbol1, -position['quantity1'], price1)
            self.broker_api.place_order(symbol2, position['quantity2'], price2)
            
            # Calculate P&L
            pnl1 = position['quantity1'] * (price1 - position['entry_price1'])
            pnl2 = position['quantity2'] * (position['entry_price2'] - price2)
            total_pnl = pnl1 + pnl2
            
            # Remove position
            self.active_positions.pop(pos_idx)
            
            msg = f"Closed {symbol1}/{symbol2} PnL: {total_pnl:.2f}"
            logger.info(msg)
            self.execution_log.append({
                'timestamp': datetime.now(),
                'action': 'CLOSE',
                'message': msg,
                'pnl': total_pnl
            })
            
            return True, msg
        
        except Exception as e:
            msg = f"Close failed: {str(e)}"
            logger.error(msg)
            return False, msg
    
    def monitor_positions(self) -> List[Dict]:
        """Monitor active positions and update unrealized P&L."""
        positions_summary = []
        
        for pos in self.active_positions:
            price1 = self.price_provider.get_latest_price(pos['symbol1'])
            price2 = self.price_provider.get_latest_price(pos['symbol2'])
            
            if price1 is None or price2 is None:
                continue
            
            pnl1 = pos['quantity1'] * (price1 - pos['entry_price1'])
            pnl2 = -pos['quantity2'] * (price2 - pos['entry_price2'])
            unrealized_pnl = pnl1 + pnl2
            
            pos['unrealized_pnl'] = unrealized_pnl
            
            positions_summary.append({
                'pair': f"{pos['symbol1']}/{pos['symbol2']}",
                'direction': pos['direction'],
                'unrealized_pnl': unrealized_pnl,
                'current_price1': price1,
                'current_price2': price2,
                'entry_time': pos['entry_time'],
                'holding_time': datetime.now() - pos['entry_time']
            })
        
        return positions_summary
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        account_balance = self.broker_api.get_account_balance()
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.active_positions)
        
        return {
            'account_balance': account_balance,
            'active_positions': len(self.active_positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions': self.monitor_positions()
        }

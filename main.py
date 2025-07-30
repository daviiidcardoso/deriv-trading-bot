import asyncio
import websockets
import json
import logging
import time
import os
import math
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import statistics

# Carregar variáveis do arquivo .env
load_dotenv()

# Configuração de logging otimizada para velocidade
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AssetState:
    """Estado de um ativo individual - DUAL ENTRY OTIMIZADO"""
    symbol: str
    current_sequence: int = 1
    in_cooldown: bool = False
    cooldown_end_time: float = 0
    active_contracts: List = field(default_factory=list)
    balance_before_operation: float = 0
    last_entry_direction: str = ""
    last_signal_time: float = 0
    last_signal_reason: str = ""
    martingale_loss_accumulator: float = 0.0
    
    # Para dual entry - direções atuais da operação
    current_call_direction: str = "CALL"
    current_put_direction: str = "PUT"

@dataclass
class TickData:
    """Dados de tick para análise técnica"""
    timestamp: float
    price: float

@dataclass
class CandleData:
    """Dados de candle para análise"""
    timestamp: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float

    @property
    def is_green(self) -> bool:
        return self.close_price > self.open_price

    @property
    def is_red(self) -> bool:
        return self.close_price < self.open_price

    @property
    def color_str(self) -> str:
        if self.is_green: return "GREEN"
        if self.is_red: return "RED"
        return "DOJI"


class TechnicalAnalysis:
    """Classe para cálculos de análise técnica"""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calcula RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        if len(gains) < period:
            return 50.0
            
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calcula EMA"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

class Strategy:
    """Classe base para estratégias"""
    def __init__(self, config: dict):
        self.config = config

    def analyze_signal(self, symbol: str, candles: List[CandleData]) -> Optional[Tuple[str, List[str]]]:
        raise NotImplementedError

class RSIStrategy(Strategy):
    """Estratégia baseada em RSI com 3 tipos"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.rsi_type = config.get('RSI_STRATEGY_TYPE', 'REVERSAL_CONFIRMATION')
        self.period = int(config.get('RSI_PERIOD', 14))
        self.confirmation_candles = int(config.get('RSI_CONFIRMATION_CANDLES', 2))
        self.oversold_pure = float(config.get('RSI_OVERSOLD_PURE', 25))
        self.overbought_pure = float(config.get('RSI_OVERBOUGHT_PURE', 75))
        self.oversold_conf = float(config.get('RSI_OVERSOLD_CONF', 30))
        self.overbought_conf = float(config.get('RSI_OVERBOUGHT_CONF', 70))
        self.exit_oversold = float(config.get('RSI_EXIT_OVERSOLD', 35))
        self.exit_overbought = float(config.get('RSI_EXIT_OVERBOUGHT', 65))
        self.rsi_history = {}

    def analyze_signal(self, symbol: str, candles: List[CandleData]) -> Optional[Tuple[str, List[str]]]:
        min_required = max(8, self.period // 2)
        if len(candles) < min_required: return None
        prices = [candle.close_price for candle in candles]
        current_rsi = TechnicalAnalysis.calculate_rsi(prices, min(self.period, len(prices) - 1))
        if symbol not in self.rsi_history: self.rsi_history[symbol] = deque(maxlen=10)
        self.rsi_history[symbol].append(current_rsi)
        signal = None
        if self.rsi_type == 'REVERSAL_PURE': signal = self._reversal_pure(current_rsi)
        elif self.rsi_type == 'REVERSAL_CONFIRMATION': signal = self._reversal_confirmation(symbol, current_rsi)
        elif self.rsi_type == 'TREND_FOLLOWING': signal = self._trend_following(symbol, current_rsi)
        if signal: return (signal, [])
        return None

    def _reversal_pure(self, rsi: float) -> Optional[str]:
        if rsi <= self.oversold_pure: return 'CALL'
        if rsi >= self.overbought_pure: return 'PUT'
        return None

    def _reversal_confirmation(self, symbol: str, rsi: float) -> Optional[str]:
        history = self.rsi_history[symbol]
        if len(history) < self.confirmation_candles + 1: return None
        past_rsi = list(history)[-self.confirmation_candles-1:-1]
        if all(r <= self.oversold_conf for r in past_rsi) and rsi <= self.oversold_conf: return 'CALL'
        if all(r >= self.overbought_conf for r in past_rsi) and rsi >= self.overbought_conf: return 'PUT'
        return None

    def _trend_following(self, symbol: str, rsi: float) -> Optional[str]:
        history = self.rsi_history[symbol]
        if len(history) < 2: return None
        prev_rsi = history[-2]
        if prev_rsi <= self.oversold_conf and rsi > self.exit_oversold: return 'CALL'
        if prev_rsi >= self.overbought_conf and rsi < self.exit_overbought: return 'PUT'
        return None

class CandlePatternStrategy(Strategy):
    """Estratégia baseada em padrões de candles"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.call_patterns = []
        for i in range(1, 11):
            color = config.get(f'CALL_CANDLE_{i}', 'ANY').upper()
            if color in ['RED', 'GREEN', 'ANY']:
                self.call_patterns.append(color)
            else:
                break
        self.put_patterns = []
        for i in range(1, 11):
            color = config.get(f'PUT_CANDLE_{i}', 'ANY').upper()
            if color in ['RED', 'GREEN', 'ANY']:
                self.put_patterns.append(color)
            else:
                break

    def analyze_signal(self, symbol: str, candles: List[CandleData]) -> Optional[Tuple[str, List[str]]]:
        """Analisa padrões de candles e retorna o sinal com as cores das velas"""
        if len(candles) < max(len(self.call_patterns), len(self.put_patterns)) + 1:
            return None
        
        closed_candles = candles[:-1]
        
        # Verificar padrão CALL
        is_match, matched_candles = self._check_pattern(closed_candles, self.call_patterns)
        if is_match:
            return ('CALL', matched_candles)
            
        # Verificar padrão PUT
        is_match, matched_candles = self._check_pattern(closed_candles, self.put_patterns)
        if is_match:
            return ('PUT', matched_candles)
            
        return None

    def _check_pattern(self, closed_candles: List[CandleData], pattern: List[str]) -> Tuple[bool, List[str]]:
        """Verifica se o padrão corresponde e retorna as cores das velas reais"""
        if not pattern: return False, []
        
        specific_pattern = [p for p in pattern if p != 'ANY']
        if not specific_pattern: return False, []
        
        recent_candles = closed_candles[-len(specific_pattern):]
        if len(recent_candles) < len(specific_pattern): return False, []
        
        actual_candle_colors = []
        for i, expected_color in enumerate(specific_pattern):
            candle = recent_candles[i]
            actual_candle_colors.append(candle.color_str)
            if (expected_color == 'GREEN' and not candle.is_green) or \
               (expected_color == 'RED' and not candle.is_red):
                return False, []
                
        return True, actual_candle_colors

class MovingAverageStrategy(Strategy):
    """Estratégia baseada em cruzamento de médias móveis"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.ema_fast = int(config.get('EMA_FAST', 8))
        self.ema_slow = int(config.get('EMA_SLOW', 21))
        self.confirmation_candles = int(config.get('CONFIRMATION_CANDLES', 1))

    def analyze_signal(self, symbol: str, candles: List[CandleData]) -> Optional[Tuple[str, List[str]]]:
        min_required = max(8, self.ema_slow // 2)
        if len(candles) < min_required: return None
        prices = [c.close_price for c in candles]
        fast_period = min(self.ema_fast, len(prices) // 2)
        slow_period = min(self.ema_slow, len(prices) - 1)
        if fast_period >= slow_period: return None
        current_fast = TechnicalAnalysis.calculate_ema(prices[-fast_period:], fast_period)
        current_slow = TechnicalAnalysis.calculate_ema(prices[-slow_period:], slow_period)
        if len(prices) > slow_period:
            prev_fast = TechnicalAnalysis.calculate_ema(prices[-fast_period-1:-1], fast_period)
            prev_slow = TechnicalAnalysis.calculate_ema(prices[-slow_period-1:-1], slow_period)
            if prev_fast <= prev_slow and current_fast > current_slow: return ('CALL', [])
            if prev_fast >= prev_slow and current_fast < current_slow: return ('PUT', [])
        return None

class DerivMultiAssetBot:
    """Robô Deriv Multi-Ativo - OTIMIZADO PARA TICKS RÁPIDOS"""
    def __init__(self):
        """Inicializa o robô com todas as configurações"""
        self.api_token = os.getenv("DERIV_API_TOKEN")
        self.app_id = int(os.getenv("DERIV_APP_ID", 1089))
        symbols_str = os.getenv("SYMBOLS", "")
        self.symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_OPERATIONS", 3))
        self.dual_entry = os.getenv("DUAL_ENTRY", "false").lower() == "true"
        self.operation_type = os.getenv("OPERATION_TYPE", "MARTINGALE_NEXT_SIGNAL").upper()
        self.analysis_timeframe_minutes = int(os.getenv("ANALYSIS_TIMEFRAME", 1))
        self.analysis_timeframe_seconds = self.analysis_timeframe_minutes * 60
        self.duration = int(os.getenv("DURATION", 4))
        self.duration_unit = os.getenv("DURATION_UNIT", "t")
        self.amount_type = os.getenv("AMOUNT_TYPE", "PERCENTAGE").upper()
        self.initial_amount = float(os.getenv("INITIAL_AMOUNT", 0.35))
        self.initial_percentage = float(os.getenv("INITIAL_PERCENTAGE", 0.4))
        self.min_amount = float(os.getenv("MIN_AMOUNT", 0.35))
        self.max_amount = float(os.getenv("MAX_AMOUNT", 2000))
        self.multiplier = float(os.getenv("MULTIPLIER", 2.0))
        self.max_martingale_sequence = int(os.getenv("MAX_MARTINGALE_SEQUENCE", 2))
        self.cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", 1))
        self.signal_debounce = float(os.getenv("SIGNAL_DEBOUNCE", 5.0))
        call_dirs_str = os.getenv("MARTINGALE_DIRECTIONS_CALL", "CALL,PUT,CALL,PUT")
        put_dirs_str = os.getenv("MARTINGALE_DIRECTIONS_PUT", "PUT,CALL,PUT,CALL")
        self.martingale_directions_call = [d.strip().upper() for d in call_dirs_str.split(",")]
        self.martingale_directions_put = [d.strip().upper() for d in put_dirs_str.split(",")]
        self.stop_loss_value = float(os.getenv("STOP_LOSS_VALUE", 100))
        self.stop_win_value = float(os.getenv("STOP_WIN_VALUE", 100))
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.max_verification_attempts = int(os.getenv("MAX_VERIFICATION_ATTEMPTS", 10))
        self.verification_timeout = int(os.getenv("VERIFICATION_TIMEOUT", 5))
        self.max_signal_age = float(os.getenv("MAX_SIGNAL_AGE", 5.0))
        self.delay_between_ops = float(os.getenv("DELAY_BETWEEN_OPS", 0.05))
        
        # 🔧 OTIMIZAÇÕES PARA TICKS
        self.is_tick_mode = (self.duration_unit == "t")
        if self.is_tick_mode:
            # Ajustar timeouts para velocidade em modo tick
            self.verification_timeout = min(self.verification_timeout, 3)
            self.max_verification_attempts = min(self.max_verification_attempts, 3)
            self.delay_between_ops = min(self.delay_between_ops, 0.02)
            logger.info(f"⚡ MODO TICK ATIVADO: Timeouts otimizados para {self.duration} ticks")
        
        self.time_offset = 0.0
        
        self.global_sequence = 1
        self.global_operation_active = False
        self.websocket = None
        self.is_connected = False
        self.balance = 0.0
        self.initial_balance = 0.0
        self.reconnection_attempts = 0
        self.max_reconnection_attempts = 5
        self.reconnection_delay = 3
        self.connection_stable = False
        self.is_reconnecting = False
        self.reconnection_lock = asyncio.Lock()
        self.last_successful_message = time.time()
        
        self.asset_states = {symbol: AssetState(symbol) for symbol in self.symbols}
        
        self.tick_cache = {symbol: deque(maxlen=1000) for symbol in self.symbols}
        self.candle_cache = {symbol: deque(maxlen=200) for symbol in self.symbols}
        
        self.processed_contracts = {}
        
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        
        self.pending_requests = {}
        
        self.asset_locks = {symbol: asyncio.Lock() for symbol in self.symbols}
        
        self.session_stats = {
            'operations_total': 0, 'operations_won': 0, 'operations_lost': 0,
            'contracts_total': 0, 'contracts_won': 0, 'contracts_lost': 0,
            'asset_stats': {symbol: {'operations': 0, 'wins': 0, 'losses': 0, 'profit': 0.0, 'best_sequence': 1} for symbol in self.symbols}
        }
        
        self._init_strategies()
        self._validate_config()

    def get_current_time(self) -> float:
        """Retorna o tempo atual da máquina ajustado pelo offset do servidor."""
        return time.time() + self.time_offset
    
    async def _synchronize_clock(self):
        """Calcula e armazena a diferença entre o relógio local e o do servidor."""
        if not self.is_tick_mode:
            logger.info("🕒 Sincronizando relógio com o servidor Deriv...")
        try:
            t0 = time.time()
            response = await self.send_request({"time": 1}, timeout=5)
            t1 = time.time()
            if response and "time" in response:
                server_time = response["time"]
                local_time_estimate = t0 + (t1 - t0) / 2
                self.time_offset = server_time - local_time_estimate
                self.last_successful_message = self.get_current_time()
                if not self.is_tick_mode:
                    logger.info(f"✅ Relógio sincronizado! Offset: {self.time_offset:+.3f} segundos.")
            else:
                self.time_offset = 0.0
        except Exception as e:
            if self.debug_mode:
                logger.error(f"❌ Erro ao sincronizar relógio: {e}. Usando o tempo local.")
            self.time_offset = 0.0

    def _init_strategies(self):
        """Inicializa estratégias ativas"""
        config = dict(os.environ)
        self.strategies = []
        
        if os.getenv("STRATEGY_1_ACTIVE", "false").lower() == "true": 
            rsi_strategy = RSIStrategy(config)
            self.strategies.append(rsi_strategy)
            logger.info(f"✅ RSI Strategy ativada: Tipo={rsi_strategy.rsi_type}, Período={rsi_strategy.period}")
            
        if os.getenv("STRATEGY_2_ACTIVE", "true").lower() == "true": 
            candle_strategy = CandlePatternStrategy(config)
            self.strategies.append(candle_strategy)
            logger.info(f"✅ Candle Pattern Strategy ativada:")
            logger.info(f"   📈 CALL: {' → '.join(candle_strategy.call_patterns[:5])}{'...' if len(candle_strategy.call_patterns) > 5 else ''}")
            logger.info(f"   📉 PUT: {' → '.join(candle_strategy.put_patterns[:5])}{'...' if len(candle_strategy.put_patterns) > 5 else ''}")
            
        if os.getenv("STRATEGY_3_ACTIVE", "false").lower() == "true": 
            ma_strategy = MovingAverageStrategy(config)
            self.strategies.append(ma_strategy)
            logger.info(f"✅ Moving Average Strategy ativada: EMA({ma_strategy.ema_fast}, {ma_strategy.ema_slow})")
        
        logger.info(f"🎯 Total de estratégias ativas: {len(self.strategies)}")

    def _validate_config(self):
        """Valida configurações"""
        if not self.api_token: raise ValueError("❌ Configure DERIV_API_TOKEN no arquivo .env!")
        if not self.symbols: raise ValueError("❌ Configure pelo menos um símbolo em SYMBOLS!")
        valid_types = ["FIXED", "MARTINGALE_IMMEDIATE", "MARTINGALE_NEXT_SIGNAL", "MARTINGALE_GLOBAL"]
        if self.operation_type not in valid_types: raise ValueError(f"❌ OPERATION_TYPE '{self.operation_type}' é inválido. Use um dos: {', '.join(valid_types)}")
        if not self.strategies: raise ValueError("❌ Ative pelo menos uma estratégia!")

    def _log_config(self):
        """Exibe configuração carregada"""
        mode_info = "⚡ MODO TICKS RÁPIDOS" if self.is_tick_mode else "DUAL ENTRY CORRETO"
        logger.info(f"🚀 ROBÔ DERIV MULTI-ATIVO - {mode_info}")
        logger.info("=" * 80)
        logger.info(f"📊 Análise: M{self.analysis_timeframe_minutes}")
        logger.info(f"⏰ Expiração: {self.duration}{self.duration_unit}")
        logger.info(f"⚙️ Tipo de Operação: {self.operation_type}")
        
        # 🎯 INFORMAÇÕES DETALHADAS DE ESTRATÉGIAS
        logger.info(f"🧠 Estratégias Ativas ({len(self.strategies)}):")
        strategy_names = [s.__class__.__name__.replace('Strategy', '') for s in self.strategies]
        if strategy_names:
            logger.info(f"   📊 {', '.join(strategy_names)}")
        else:
            logger.info("   ❌ NENHUMA ESTRATÉGIA ATIVA!")
        
        if self.dual_entry:
            logger.info("🔄 Dual Entry: ✅ ATIVO (CALL+PUT como operação ÚNICA)")
            if self.is_tick_mode:
                logger.info("   ⚡ OTIMIZADO para operações rápidas de ticks")
                expected_duration = self.duration * 2.5
                logger.info(f"   ⚡ Duração esperada: ~{expected_duration:.1f}s por operação")
            logger.info("   💡 Vitória: Se CALL OU PUT ganhar")
            logger.info("   💡 Martingale: Apenas se AMBOS perderem")
            logger.info(f"🎯 Martingale CALL: {' → '.join(self.martingale_directions_call[:3])}...")
            logger.info(f"🎯 Martingale PUT: {' → '.join(self.martingale_directions_put[:3])}...")
        else:
            logger.info("🔄 Dual Entry: ❌ INATIVO (operação single)")
        logger.info(f"📈 Ativos: {', '.join(self.symbols)}")
        if self.is_tick_mode:
            logger.info(f"⚡ Timeouts otimizados: Verificação={self.verification_timeout}s, Delay={self.delay_between_ops}s")
        logger.info("=" * 80)

    # =================================
    # 🔧 MÉTODOS OTIMIZADOS PARA TICKS
    # =================================

    def determine_dual_directions(self, symbol: str, signal_direction: str):
        """Determina as direções CALL e PUT para a operação dual"""
        asset_state = self.asset_states[symbol]
        
        if asset_state.current_sequence == 1:
            # Primeira operação: sempre CALL + PUT
            call_direction = "CALL"
            put_direction = "PUT"
            asset_state.last_entry_direction = signal_direction
        else:
            # Martingale: aplicar inversões baseado no sinal original
            if asset_state.last_entry_direction == "CALL":
                call_dirs = self.martingale_directions_call
                put_dirs = self.martingale_directions_put
            else:
                call_dirs = self.martingale_directions_put
                put_dirs = self.martingale_directions_call
            
            # Aplicar direções do martingale
            direction_index = min(asset_state.current_sequence - 2, len(call_dirs) - 1)
            call_direction = call_dirs[direction_index] if call_dirs else "CALL"
            put_direction = put_dirs[direction_index] if put_dirs else "PUT"
        
        # Armazenar direções atuais
        asset_state.current_call_direction = call_direction
        asset_state.current_put_direction = put_direction
        
        return call_direction, put_direction

    async def send_request(self, request: dict, timeout: float = 10.0) -> Optional[dict]:
        """Envia uma requisição e aguarda a resposta - OTIMIZADO PARA TICKS"""
        if not self.websocket:
            return None

        # 🔧 TIMEOUT REDUZIDO PARA OPERAÇÕES DE TICKS
        if self.is_tick_mode and timeout > 5.0:
            timeout = 5.0
            
        wait_start_time = time.time()
        while not self.websocket.open and (time.time() - wait_start_time < timeout):
            await asyncio.sleep(0.01)  # ← Reduzido de 0.05 para 0.01
        
        if not self.websocket.open or self.websocket.closed:
            return None
        
        req_id = int(self.get_current_time() * 1000000)
        request['req_id'] = req_id
        future = asyncio.Future()
        self.pending_requests[req_id] = future
        
        try:
            await self.websocket.send(json.dumps(request))
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except websockets.exceptions.ConnectionClosed:
            await self._trigger_reconnection()
            return None
        except Exception as e:
            return None
        finally:
            self.pending_requests.pop(req_id, None)

    async def get_proposal_and_buy(self, symbol: str, contract_type: str, amount: float):
        """Obtém uma proposta e executa a compra - OTIMIZADO PARA VELOCIDADE"""
        try:
            if not self.connection_stable:
                return None

            # 🔧 TIMEOUT REDUZIDO PARA TICKS
            timeout = 3.0 if self.is_tick_mode else 10.0
            
            proposal_response = await self.send_request({
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type.upper(),
                "currency": "USD",
                "duration": self.duration,
                "duration_unit": self.duration_unit,
                "symbol": symbol
            }, timeout=timeout)
            
            if not proposal_response or "proposal" not in proposal_response or "error" in proposal_response:
                return None
            
            buy_response = await self.send_request({
                "buy": proposal_response["proposal"]["id"], 
                "price": amount
            }, timeout=timeout)
            
            if buy_response and "buy" in buy_response and "error" not in buy_response:
                contract_info = {
                    "id": buy_response["buy"]["contract_id"],
                    "symbol": symbol,
                    "type": contract_type,
                    "amount": amount,
                    "status": "open",
                    "start_time": self.get_current_time()
                }
                
                # 🔧 INSCREVER IMEDIATAMENTE NO CONTRATO PARA RECEBER UPDATES VIA WEBSOCKET
                await self.subscribe_to_contract(contract_info["id"])
                
                logger.info(f"✅ {symbol} {contract_type}: ${amount:.2f} | ID: {contract_info['id']}")
                return contract_info
            
            return None
            
        except Exception as e:
            return None

    async def wait_for_results(self, symbol: str):
        """Aguarda a finalização dos contratos abertos - OTIMIZADO PARA TICKS"""
        asset_state = self.asset_states[symbol]
        if not asset_state.active_contracts: 
            return
        
        # 🔧 CÁLCULO CORRETO PARA TICKS
        if self.duration_unit == "t":
            # Para ticks: cada tick = ~2.5 segundos + margem mínima
            expected_duration_seconds = self.duration * 2.5
            total_wait_time = expected_duration_seconds + 3.0  # ← Margem reduzida de 10s para 3s
        elif self.duration_unit == "s":
            expected_duration_seconds = self.duration
            total_wait_time = expected_duration_seconds + 3.0
        elif self.duration_unit == "m":
            expected_duration_seconds = self.duration * 60
            total_wait_time = expected_duration_seconds + 10.0
        else:
            expected_duration_seconds = self.duration
            total_wait_time = expected_duration_seconds + 3.0
        
        logger.info(f"⏳ {symbol}: Aguardando resultado por até {total_wait_time:.0f}s")
        start_time = self.get_current_time()
        
        # 🔧 VERIFICAÇÃO MAIS FREQUENTE PARA TICKS
        check_interval = 0.3 if self.is_tick_mode else 1.0
        
        while self.get_current_time() - start_time < total_wait_time:
            if all(c.get("status") == "finished" for c in asset_state.active_contracts):
                elapsed = self.get_current_time() - start_time
                logger.info(f"✅ {symbol}: Resultado recebido em {elapsed:.1f}s")
                return
            await asyncio.sleep(check_interval)
        
        # Timeout - verificação final mais rápida
        logger.warning(f"⚠️ {symbol}: Timeout após {total_wait_time:.0f}s. Verificação rápida...")
        
        for contract in asset_state.active_contracts:
            if contract.get("status") != "finished":
                logger.info(f"🔎 Verificando contrato {contract['id']}...")
                
                # 🔧 TIMEOUT REDUZIDO PARA VERIFICAÇÃO
                verification_timeout = self.verification_timeout
                contract_data = await self.send_request(
                    {"proposal_open_contract": 1, "contract_id": contract["id"]}, 
                    timeout=verification_timeout
                )
                
                if contract_data and "proposal_open_contract" in contract_data and contract_data["proposal_open_contract"].get("status") in ["sold", "won", "lost"]:
                    await self._process_contract_result(contract, contract_data["proposal_open_contract"], symbol)
                else:
                    contract["status"] = "finished"
                    contract["profit"] = -contract.get("amount", 0)
                    logger.error(f"❌ Contrato {contract['id']} forçado como perda após verificação rápida")

    async def execute_dual_operation(self, symbol: str, signal_direction: str):
        """Executa operação dual entry - OTIMIZADO PARA VELOCIDADE"""
        if not self.connection_stable:
            logger.warning(f"⚠️ Não é possível operar em {symbol}: conexão não estável.")
            return False

        asset_state = self.asset_states[symbol]
        
        # Determinar direções
        call_direction, put_direction = self.determine_dual_directions(symbol, signal_direction)
        
        # Calcular valor por operação
        base_amount = self.calculate_amount(symbol)
        
        # Verificar saldo suficiente para operação dual
        total_needed = base_amount * 2
        if total_needed > self.balance:
            logger.error(f"❌ {symbol}: Saldo insuficiente! Necessário: ${total_needed:.2f}, Disponível: ${self.balance:.2f}")
            return False
        
        # Log da operação
        duration_info = f"{self.duration}{self.duration_unit}"
        logger.info(f"🎯 {symbol} DUAL S{asset_state.current_sequence}: {call_direction}(${base_amount:.2f}) + {put_direction}(${base_amount:.2f}) | Total: ${total_needed:.2f}")
        
        # Armazenar saldo antes da operação
        asset_state.balance_before_operation = self.balance
        
        # Limpar contratos ativos
        asset_state.active_contracts = []
        
        # 🔧 EXECUÇÃO SEQUENCIAL MAIS RÁPIDA
        execution_start = self.get_current_time()
        
        # Executar primeira operação
        call_contract = await self.get_proposal_and_buy(symbol, call_direction, base_amount)
        
        # 🔧 DELAY MÍNIMO ENTRE OPERAÇÕES PARA TICKS
        await asyncio.sleep(self.delay_between_ops)
        
        # Executar segunda operação
        put_contract = await self.get_proposal_and_buy(symbol, put_direction, base_amount)
        
        execution_time = self.get_current_time() - execution_start
        
        success_count = 0
        if call_contract:
            asset_state.active_contracts.append(call_contract)
            success_count += 1
        if put_contract:
            asset_state.active_contracts.append(put_contract)
            success_count += 1
        
        if success_count >= 1:
            logger.info(f"🚀 {symbol}: {success_count}/2 operações dual executadas em {execution_time:.2f}s")
            
            # Aguardar resultados com tempo otimizado
            await self.wait_for_results(symbol)
            
            # Atualizar saldo
            await self.get_balance()
            
            # Analisar resultado
            result = self.analyze_dual_operation_result(symbol)
            
            # 🔧 MARTINGALE IMEDIATO COM DELAY MÍNIMO PARA TICKS
            if result == "continue_immediate":
                delay_before_martingale = 0.2 if self.is_tick_mode else 1.0
                logger.info(f"⚡ {symbol}: Continuando martingale dual imediato em {delay_before_martingale}s...")
                await asyncio.sleep(delay_before_martingale)
                await self.execute_dual_operation(symbol, signal_direction)
            elif result in ["victory", "max_sequence"]:
                self.show_session_summary()
            
            return True
        else:
            logger.error(f"❌ {symbol}: Falha total na operação dual")
            return False

    def analyze_dual_operation_result(self, symbol: str) -> str:
        """Analisa resultado da operação dual como operação única"""
        asset_state = self.asset_states[symbol]
        
        if not asset_state.active_contracts or not all(c.get("status") == "finished" for c in asset_state.active_contracts):
            logger.error(f"Estado inconsistente para {symbol}: Nem todos os contratos foram finalizados.")
            return "error"
        
        # 🎯 CORREÇÃO CRÍTICA: Calcular lucro total diretamente dos contratos
        # Para evitar duplicação quando direções são iguais
        total_profit = sum(c.get("profit", 0) for c in asset_state.active_contracts)
        
        # Separar contratos por tipo para logging (mas não para cálculo)
        call_contracts = [c for c in asset_state.active_contracts if c.get("type") == "CALL"]
        put_contracts = [c for c in asset_state.active_contracts if c.get("type") == "PUT"]
        
        # Calcular lucros para exibição
        call_profit = sum(c.get("profit", 0) for c in call_contracts)
        put_profit = sum(c.get("profit", 0) for c in put_contracts)
        
        call_won = call_profit > 0
        put_won = put_profit > 0
        
        # 🎯 CORREÇÃO: Para contabilização por ativo, usar o total real
        operation_net_profit = total_profit
        
        # Log detalhado - mostrar apenas os tipos que realmente existem
        call_status = "WIN" if call_won else "LOSS"
        put_status = "WIN" if put_won else "LOSS"
        
        # Montar display dos resultados
        call_display = ""
        put_display = ""
        
        if call_contracts:
            if len(call_contracts) > 1:
                call_display = f"CALL×{len(call_contracts)}={call_status}(${call_profit:+.2f})"
            else:
                call_display = f"CALL={call_status}(${call_profit:+.2f})"
                
        if put_contracts:
            if len(put_contracts) > 1:
                put_display = f"PUT×{len(put_contracts)}={put_status}(${put_profit:+.2f})"
            else:
                put_display = f"PUT={put_status}(${put_profit:+.2f})"
        
        # Montar log final
        if call_display and put_display:
            result_display = f"{call_display} | {put_display}"
        elif call_display:
            result_display = f"{call_display} | PUT=NONE($0.00)"
        elif put_display:
            result_display = f"CALL=NONE($0.00) | {put_display}"
        else:
            result_display = "CALL=NONE($0.00) | PUT=NONE($0.00)"
        
        logger.info(f"📊 {symbol} DUAL RESULT: {result_display} | Total: ${total_profit:+.2f}")
        
        # LÓGICA DUAL ENTRY CORRETA:
        # Vitória = Se CALL OU PUT ganhar
        # Derrota = Apenas se AMBOS perderem
        
        if call_won or put_won:
            # VITÓRIA: Pelo menos uma operação ganhou
            
            # 🎯 CORREÇÃO: Calcular resultado final da sequência completa
            full_sequence_result = operation_net_profit + asset_state.martingale_loss_accumulator
            
            if asset_state.martingale_loss_accumulator < 0:
                # Houve martingale - usar resultado líquido de toda a sequência
                logger.info(f"🎉 {symbol}: VITÓRIA DUAL! Resultado final da sequência: ${full_sequence_result:+.2f}")
                logger.info(f"   💰 (Lucro desta operação: ${operation_net_profit:+.2f} | Perdas anteriores: ${asset_state.martingale_loss_accumulator:+.2f})")
                self.update_session_stats(symbol, "victory", full_sequence_result)
            else:
                # Operação única sem martingale - usar apenas lucro desta operação
                logger.info(f"🎉 {symbol}: VITÓRIA DUAL! Lucro desta operação: ${operation_net_profit:+.2f}")
                self.update_session_stats(symbol, "victory", operation_net_profit)
            
            # Reset da sequência
            asset_state.current_sequence = 1
            asset_state.martingale_loss_accumulator = 0.0
            
            # Colocar em cooldown
            self.put_asset_in_cooldown(symbol)
            
            return "victory"
        else:
            # DERROTA: Ambas operações perderam
            if asset_state.current_sequence >= self.max_martingale_sequence:
                # 🎯 CORREÇÃO: Para estatísticas, usar o resultado de toda a sequência
                full_sequence_result = operation_net_profit + asset_state.martingale_loss_accumulator
                logger.warning(f"🛑 {symbol}: Limite de martingale atingido! Perda total da sequência: ${full_sequence_result:+.2f}")
                self.update_session_stats(symbol, "max_sequence", full_sequence_result)
                
                # Reset da sequência
                asset_state.current_sequence = 1
                asset_state.martingale_loss_accumulator = 0.0
                
                # Colocar em cooldown
                self.put_asset_in_cooldown(symbol)
                
                return "max_sequence"
            else:
                # Continuar martingale
                asset_state.martingale_loss_accumulator += operation_net_profit
                asset_state.current_sequence += 1
                
                logger.info(f"📈 {symbol}: Martingale DUAL → S{asset_state.current_sequence}. Perda acumulada: ${asset_state.martingale_loss_accumulator:+.2f}")
                
                # Mostrar próximas direções
                next_call, next_put = self.determine_dual_directions(symbol, asset_state.last_entry_direction)
                logger.info(f"🎯 Próxima operação dual: {next_call} + {next_put}")
                
                return "continue_immediate" if self.operation_type == "MARTINGALE_IMMEDIATE" else "continue"

    # =================================
    # 🔧 CÁLCULO CORRIGIDO DO MARTINGALE
    # =================================

    def calculate_amount(self, symbol: str) -> float:
        """
        🎯 CORREÇÃO: Calcula o valor da entrada seguindo EXATAMENTE a lógica da calculadora.
        
        - Para AMOUNT_TYPE='PERCENTAGE': Usa BANCA INICIAL × percentual × multiplicador acumulado
        - Para AMOUNT_TYPE='FIXED': Usa valor_fixo × multiplicador acumulado
        """
        # Se o tipo de operação é FIXED, sempre retorna o valor inicial fixo
        if self.operation_type == "FIXED":
            return self.initial_amount
        
        # Determinar a sequência atual (global ou por ativo)
        if self.operation_type == "MARTINGALE_GLOBAL":
            sequence = self.global_sequence
        else:
            sequence = self.asset_states[symbol].current_sequence
        
        # Calcular o multiplicador acumulado (igual à calculadora)
        current_multiplier_value = self.multiplier ** (sequence - 1)
        
        # 🎯 CORREÇÃO PRINCIPAL: Seguir exatamente a lógica da calculadora
        if self.amount_type == "FIXED":
            # VALOR FIXO: valor_fixo_base × multiplicador_acumulado
            # Equivale a: entryValue.value * currentMultiplierValue
            amount = self.initial_amount * current_multiplier_value
            
            if self.debug_mode:
                logger.info(f"💡 {symbol} S{sequence} (FIXO): ${self.initial_amount:.2f} × {current_multiplier_value:.3f} = ${amount:.2f}")
                
        elif self.amount_type == "PERCENTAGE":
            # PERCENTUAL: (banca_inicial × percentual_base × multiplicador_acumulado) / 100
            # Equivale a: (bankroll * entryValue.value * currentMultiplierValue) / 100
            amount = (self.initial_balance * self.initial_percentage * current_multiplier_value) / 100
            
            if self.debug_mode:
                logger.info(f"💡 {symbol} S{sequence} (PERCENTUAL): (${self.initial_balance:.2f} × {self.initial_percentage:.2f}% × {current_multiplier_value:.3f}) / 100 = ${amount:.2f}")
        else:
            # Fallback para valor fixo
            amount = self.initial_amount
            logger.warning(f"⚠️ AMOUNT_TYPE '{self.amount_type}' não reconhecido. Usando valor fixo.")
        
        # Aplicar limite mínimo (como na calculadora: Math.max(roundEntry, minAmount))
        final_amount = max(self.min_amount, round(amount, 2))
        
        # Aplicar limite máximo para segurança
        final_amount = min(self.max_amount, final_amount)
        
        # Log se houve ajuste pelos limites
        if final_amount != round(amount, 2):
            logger.info(f"⚖️ {symbol}: Valor ajustado de ${amount:.2f} para ${final_amount:.2f} (limites: ${self.min_amount:.2f} - ${self.max_amount:.2f})")
        
        return final_amount

    # =================================
    # 🔧 MÉTODOS PRINCIPAIS ADAPTADOS
    # =================================

    async def main_loop(self):
        """Loop principal adaptado para dual entry otimizado"""
        logger.info("🚀 Iniciando loop principal...")
        active_tasks = {}
        
        try:
            while self.is_connected:
                stop_reason = self.check_stop_conditions()
                if stop_reason:
                    logger.info(f"🏁 {stop_reason.upper()} atingido!")
                    break
                
                # Limpar tarefas finalizadas
                finished_symbols = [s for s, t in active_tasks.items() if t.done()]
                for s in finished_symbols:
                    try:
                        await active_tasks.pop(s)
                    except Exception as e:
                        logger.error(f"❌ Erro na tarefa de {s} ao finalizar: {e}")

                if self.connection_stable and len(active_tasks) < self.max_concurrent:
                    signals = await self.analyze_signals()
                    
                    for symbol, direction in signals.items():
                        if symbol not in active_tasks:
                            if self.dual_entry:
                                # DUAL ENTRY: Executar operação dual
                                if not self.is_asset_busy(symbol):
                                    logger.info(f"🚀 {symbol}: Disparando OPERAÇÃO DUAL para sinal {direction}")
                                    active_tasks[symbol] = asyncio.create_task(
                                        self.execute_dual_operation(symbol, direction)
                                    )
                            else:
                                # SINGLE ENTRY: Comportamento original
                                if not self.is_asset_busy(symbol):
                                    logger.info(f"🚀 {symbol}: Disparando tarefa para sinal {direction}")
                                    active_tasks[symbol] = asyncio.create_task(
                                        self.run_asset_operation(symbol, direction)
                                    )
                            
                            if self.operation_type == "MARTINGALE_GLOBAL":
                                break
                
                await asyncio.sleep(0.5 if self.is_tick_mode else 1.0)

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("⚠️ Robô interrompido.")
        finally:
            if active_tasks:
                logger.info(f"⏳ Aguardando {len(active_tasks)} operações ativas finalizarem...")
                await asyncio.gather(*active_tasks.values(), return_exceptions=True)

    async def analyze_signals(self):
        """Analisa sinais para todos os ativos"""
        signals = {}
        current_time = self.get_current_time()
        
        # 🔧 LOG DE STATUS MENOS FREQUENTE PARA MODO TICK
        log_interval = 60 if self.is_tick_mode else 30
        if not hasattr(self, '_last_status_log') or current_time - self._last_status_log > log_interval:
            self._log_data_status()
            self._last_status_log = current_time
        
        if not self.connection_stable:
            return signals

        for symbol in self.symbols:
            if self.is_asset_busy(symbol):
                continue
            
            candles = list(self.candle_cache.get(symbol, []))
            if not candles:
                continue

            # Verificar debounce normal
            if not self.check_signal_debounce(symbol):
                continue

            # 🎯 ANÁLISE DETALHADA DAS ESTRATÉGIAS
            strategy_results = []
            
            # Analisar estratégias
            for i, strategy in enumerate(self.strategies, 1):
                strategy_name = strategy.__class__.__name__.replace('Strategy', '')
                
                try:
                    analysis_result = strategy.analyze_signal(symbol, candles)
                    
                    if analysis_result:
                        signal, matched_colors = analysis_result
                        signal_age = self.get_current_time() - candles[-1].timestamp
                        
                        if signal_age > self.max_signal_age:
                            strategy_results.append(f"❌ {strategy_name}: {signal} (muito antigo: {signal_age:.1f}s)")
                            continue

                        # 🎯 LOG DETALHADO DO SINAL ENCONTRADO
                        strategy_detail = f"✅ {strategy_name}: {signal}"
                        
                        # Adicionar detalhes específicos da estratégia
                        if isinstance(strategy, CandlePatternStrategy) and matched_colors:
                            strategy_detail += f" [Padrão: {' → '.join(matched_colors)}]"
                        elif isinstance(strategy, RSIStrategy):
                            # Calcular RSI atual para mostrar
                            prices = [candle.close_price for candle in candles]
                            current_rsi = TechnicalAnalysis.calculate_rsi(prices, min(strategy.period, len(prices) - 1))
                            strategy_detail += f" [RSI: {current_rsi:.1f}]"
                        elif isinstance(strategy, MovingAverageStrategy):
                            strategy_detail += f" [EMA: {strategy.ema_fast}/{strategy.ema_slow}]"
                        
                        strategy_results.append(strategy_detail)

                        # Log do sinal encontrado
                        if self.dual_entry:
                            if self.is_tick_mode:
                                logger.info(f"🎯 {symbol}: DUAL {signal} ({self.duration}t) via {strategy_name}")
                            else:
                                logger.info(f"🎯 {symbol}: SINAL DUAL {signal} detectado via {strategy_name}!")
                        else:
                            logger.info(f"🎯 {symbol}: SINAL {signal} detectado via {strategy_name}!")
                        
                        # Adicionar detalhes da estratégia se não for modo tick
                        if not self.is_tick_mode and matched_colors:
                            logger.info(f"   📊 Padrão detectado: {' → '.join(matched_colors)}")
                        
                        signals[symbol] = signal
                        break  # Para na primeira estratégia que encontrar sinal
                    else:
                        # Estratégia não encontrou sinal
                        reason = "sem padrão"
                        if isinstance(strategy, CandlePatternStrategy):
                            reason = "padrão não encontrado"
                        elif isinstance(strategy, RSIStrategy):
                            prices = [candle.close_price for candle in candles]
                            if len(prices) >= strategy.period:
                                current_rsi = TechnicalAnalysis.calculate_rsi(prices, min(strategy.period, len(prices) - 1))
                                reason = f"RSI {current_rsi:.1f} fora dos níveis"
                            else:
                                reason = "dados insuficientes"
                        elif isinstance(strategy, MovingAverageStrategy):
                            reason = "sem cruzamento de EMAs"
                        
                        strategy_results.append(f"⚪ {strategy_name}: {reason}")
                        
                except Exception as e:
                    strategy_results.append(f"❌ {strategy_name}: erro ({str(e)[:30]})")
                    if self.debug_mode:
                        logger.error(f"❌ Erro na estratégia {strategy_name} para {symbol}: {e}")
            
            # 🎯 LOG RESUMO DAS ESTRATÉGIAS (apenas se debug ativado)
            if self.debug_mode and strategy_results:
                logger.info(f"📊 {symbol} - Análise de estratégias:")
                for result in strategy_results:
                    logger.info(f"   {result}")
        
        return signals

    # =================================
    # 🔧 MÉTODOS ORIGINAIS PRESERVADOS (com otimizações menores)
    # =================================

    async def connect(self):
        """Conecta ao WebSocket da Deriv"""
        async with self.reconnection_lock:
            while self.reconnection_attempts < self.max_reconnection_attempts:
                try:
                    if self.websocket and not self.websocket.closed: await self.websocket.close()
                    logger.info(f"🔌 Conectando... (tentativa {self.reconnection_attempts + 1}/{self.max_reconnection_attempts})")
                    self.websocket = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20)
                    self.connection_stable = False
                    self.reconnection_attempts = 0
                    self.is_reconnecting = False
                    self.last_successful_message = self.get_current_time()
                    logger.info("✅ Conectado ao WebSocket da Deriv")
                    return True
                except Exception as e:
                    self.is_connected = False
                    self.reconnection_attempts += 1
                    delay = min(self.reconnection_delay * (2 ** self.reconnection_attempts), 30)
                    logger.error(f"❌ Tentativa {self.reconnection_attempts} falhou: {e}")
                    if self.reconnection_attempts < self.max_reconnection_attempts:
                        logger.info(f"⏳ Aguardando {delay}s..."); await asyncio.sleep(delay)
                    else:
                        logger.error("🚫 Máximo de tentativas de reconexão atingido!"); return False
            return False

    async def _connection_health_monitor(self):
        """Monitora a saúde da conexão WebSocket."""
        while True:
            try:
                if self.websocket:
                    if not self.websocket.open:
                        logger.warning("💔 Conexão WebSocket não está aberta. Tentando reconectar...")
                        await self._trigger_reconnection()
                        if not self.is_connected:
                            break
                        continue
                    
                    if self.get_current_time() - self.last_successful_message > 90:  # Aumentado para modo tick
                        logger.warning("📡 Sem resposta do servidor por 90s. Reconectando...")
                        await self._trigger_reconnection()
                        if not self.is_connected:
                            break
                        continue
                
                await asyncio.sleep(45 if self.is_tick_mode else 30)
            except asyncio.CancelledError:
                logger.info("Monitor de saúde cancelado.")
                break
            except Exception as e:
                logger.error(f"❌ Erro no monitor de saúde: {e}"); await asyncio.sleep(10)

    async def _trigger_reconnection(self):
        """Dispara o processo de reconexão."""
        async with self.reconnection_lock:
            if self.is_reconnecting:
                return
            
            logger.warning("🔄 Iniciando reconexão automática...")
            self.is_reconnecting = True
            self.is_connected = False
            self.connection_stable = False

            if hasattr(self, 'processor_task') and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.sleep(0.1)
            if hasattr(self, 'health_monitor_task') and not self.health_monitor_task.done():
                self.health_monitor_task.cancel()
                await asyncio.sleep(0.1)

            if await self.connect():
                self.processor_task = asyncio.create_task(self._message_processor())
                self.health_monitor_task = asyncio.create_task(self._connection_health_monitor())
                
                await self._post_connection_setup()
                
                if self.connection_stable:
                    logger.info("✅ Reconexão bem-sucedida!")
                    self.is_connected = True
                else:
                    logger.error("❌ Falha no setup pós-reconexão. O robô será encerrado.")
                    self.is_connected = False
            else:
                logger.error("❌ Falha na reconexão automática. O robô será encerrado.")
                self.is_connected = False
            
            self.is_reconnecting = False

    async def _message_processor(self):
        """Processa mensagens recebidas do WebSocket."""
        while True:
            try:
                if not self.websocket or not self.websocket.open:
                    await asyncio.sleep(1)
                    continue
                
                timeout = 90 if self.is_tick_mode else 60
                message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
                self.last_successful_message = self.get_current_time()
                await self._process_message_data(message)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("💔 Conexão WebSocket perdida."); await self._trigger_reconnection(); break
            except asyncio.CancelledError:
                logger.info("Processador de mensagens cancelado.")
                break
            except Exception as e:
                logger.error(f"❌ Erro ao processar mensagem: {e}"); await asyncio.sleep(1)

    async def _process_message_data(self, message: str):
        """Decodifica e direciona a mensagem JSON."""
        try:
            data = json.loads(message)
            if 'req_id' in data and data['req_id'] in self.pending_requests:
                future = self.pending_requests.pop(data['req_id'], None)
                if future and not future.done(): future.set_result(data)
            elif "tick" in data:
                await self._process_tick(data["tick"])
            elif "proposal_open_contract" in data:
                await self._process_contract_update(data["proposal_open_contract"])
        except json.JSONDecodeError:
            pass
        except Exception as e:
            if self.debug_mode: logger.error(f"❌ Erro ao processar dados da mensagem: {e}")

    async def _process_tick(self, tick_info):
        """Processa dados de tick e atualiza o cache e as velas."""
        symbol = tick_info["symbol"]
        price = float(tick_info["quote"])
        timestamp = float(tick_info["epoch"])
        
        tick_data = TickData(timestamp, price)
        if symbol in self.tick_cache:
            self.tick_cache[symbol].append(tick_data)
            self.update_candle_data(symbol, tick_data)

    async def _process_contract_update(self, contract_data: dict):
        """Processa atualizações de contratos abertos."""
        contract_id = contract_data.get("contract_id"); status = contract_data.get("status")
        if not (contract_id and status in ["sold", "won", "lost"]): return
        for symbol, asset_state in self.asset_states.items():
            for contract in asset_state.active_contracts:
                if contract.get("id") == contract_id and contract.get("status") != "finished":
                    await self._process_contract_result(contract, contract_data, symbol); return

    async def subscribe_to_contract(self, contract_id: str):
        """Inscreve-se para receber atualizações de um contrato."""
        try:
            if self.websocket and self.websocket.open:
                await self.websocket.send(json.dumps({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1}))
        except Exception as e:
            if self.debug_mode:
                logger.error(f"❌ Erro inscrevendo no contrato {contract_id}: {e}")

    async def authorize(self):
        """Autoriza a sessão."""
        response = await self.send_request({"authorize": self.api_token})
        if not response or ("error" in response and response.get('error', {}).get('message') != None):
            error_message = response.get('error', {}).get('message', 'Desconhecido') if response else 'Response is None'
            raise Exception(f"Erro na autorização: {error_message}")
        if "authorize" in response:
            logger.info("✅ Autorização bem-sucedida")
        else:
            raise Exception(f"Erro na autorização: Resposta inesperada - {response}")

    async def get_balance(self):
        """Obtém o saldo da conta."""
        response = await self.send_request({"balance": 1})
        if response and "balance" in response:
            self.balance = float(response["balance"]["balance"])
            if self.initial_balance == 0:
                self.initial_balance = self.balance; logger.info(f"💰 Saldo inicial: ${self.balance:.2f} {response['balance']['currency']}")
            elif not self.is_tick_mode:  # Log menos frequente em modo tick
                logger.info(f"💰 Saldo atual: ${self.balance:.2f} {response['balance']['currency']}")
        else:
            logger.error("❌ Erro ao obter saldo.")

    async def load_historical_data(self):
        """Carrega o histórico de velas para os ativos."""
        if not self.is_tick_mode:
            logger.info("📊 Carregando dados históricos...")
        await asyncio.gather(*(self._fetch_history_for_symbol(s) for s in self.symbols))
        ready_symbols = [s for s in self.symbols if len(self.candle_cache.get(s, [])) > 10]
        if ready_symbols: 
            logger.info(f"🚀 {len(ready_symbols)}/{len(self.symbols)} ativos prontos para operar!")
        else: 
            logger.warning("⚠️ Nenhum ativo com dados históricos suficientes.")

    async def _fetch_history_for_symbol(self, symbol: str):
        """Busca o histórico para um único símbolo."""
        try:
            response = await self.send_request({"ticks_history": symbol, "adjust_start_time": 1, "count": 50, "end": "latest", "granularity": self.analysis_timeframe_seconds, "style": "candles"})
            if response and "candles" in response and response["candles"]:
                for c in response["candles"]:
                    self.candle_cache[symbol].append(CandleData(float(c["epoch"]), float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])))
                if not self.is_tick_mode:
                    logger.info(f"✅ {symbol}: {len(response['candles'])} velas carregadas.")
            else: 
                if not self.is_tick_mode:
                    logger.warning(f"⚠️ {symbol}: Não foi possível carregar histórico")
        except Exception as e: 
            if self.debug_mode:
                logger.error(f"❌ Erro carregando histórico de {symbol}: {e}")

    async def subscribe_to_ticks(self):
        """Inscreve-se no stream de ticks para todos os ativos."""
        for s in self.symbols:
            if self.websocket and self.websocket.open:
                await self.websocket.send(json.dumps({"ticks": s, "subscribe": 1}))
        logger.info(f"📈 Inscrito nos ticks de {len(self.symbols)} ativos.")
    
    def update_candle_data(self, symbol: str, tick_data: TickData):
        """Atualiza os dados de candle com base nos ticks recebidos."""
        if symbol not in self.candle_cache: self.candle_cache[symbol] = deque(maxlen=200)
        candles = self.candle_cache[symbol]
        candle_start_time = int(tick_data.timestamp // self.analysis_timeframe_seconds) * self.analysis_timeframe_seconds
        if not candles or candles[-1].timestamp != candle_start_time:
            if candles: candles[-1].close_price = tick_data.price
            candles.append(CandleData(timestamp=candle_start_time, open_price=tick_data.price, high_price=tick_data.price, low_price=tick_data.price, close_price=tick_data.price))
        else:
            candles[-1].high_price = max(candles[-1].high_price, tick_data.price); candles[-1].low_price = min(candles[-1].low_price, tick_data.price); candles[-1].close_price = tick_data.price

    def get_martingale_direction(self, entry_direction: str, sequence: int) -> str:
        """Determina a direção para a próxima entrada de martingale."""
        if sequence == 1: return entry_direction
        directions = self.martingale_directions_call if entry_direction == "CALL" else self.martingale_directions_put
        return directions[min(sequence - 2, len(directions) - 1)] if directions else entry_direction

    def is_asset_in_cooldown(self, symbol: str) -> bool:
        """Verifica se o ativo está em cooldown."""
        asset_state = self.asset_states[symbol]
        if asset_state.in_cooldown and self.get_current_time() >= asset_state.cooldown_end_time:
            asset_state.in_cooldown = False
            if not self.is_tick_mode:
                logger.info(f"❄️ {symbol}: Cooldown finalizado")
        return asset_state.in_cooldown

    def put_asset_in_cooldown(self, symbol: str):
        """Coloca um ativo em cooldown."""
        asset_state = self.asset_states[symbol]
        asset_state.in_cooldown = True
        asset_state.cooldown_end_time = self.get_current_time() + (self.cooldown_minutes * 60)
        if self.cooldown_minutes > 0:
            logger.info(f"🧊 {symbol}: Entrando em cooldown por {self.cooldown_minutes}min")

    def is_asset_busy(self, symbol: str) -> bool:
        """Verifica se um ativo está em cooldown ou com uma operação ativa."""
        return self.is_asset_in_cooldown(symbol) or any(c.get("status") != "finished" for c in self.asset_states[symbol].active_contracts)

    def check_stop_conditions(self) -> Optional[str]:
        """Verifica as condições de parada (stop loss/win)."""
        if self.initial_balance > 0 and self.balance <= (self.initial_balance - self.stop_loss_value): return "stop_loss"
        if self.initial_balance > 0 and self.stop_win_value > 0 and self.balance >= (self.initial_balance + self.stop_win_value): return "stop_win"
        return None
    
    def check_signal_debounce(self, symbol: str) -> bool:
        """Evita que o mesmo sinal seja processado múltiplas vezes."""
        asset_state = self.asset_states[symbol]
        if self.get_current_time() - asset_state.last_signal_time < self.signal_debounce:
            return False
        asset_state.last_signal_time = self.get_current_time()
        return True

    def update_session_stats(self, symbol: str, operation_result: str, net_profit_for_sequence: float):
        """Atualiza as estatísticas da sessão."""
        asset_stats = self.session_stats['asset_stats'][symbol]
        
        # 🎯 LOG: Sempre mostrar o que está sendo adicionado às estatísticas
        logger.info(f"📊 Contabilizando: {symbol} → ${net_profit_for_sequence:+.2f}")
        
        # 🎯 DEBUG: Log detalhado se debug ativado
        if self.debug_mode:
            logger.info(f"📊 DEBUG: Adicionando ${net_profit_for_sequence:+.2f} às estatísticas de {symbol}")
            logger.info(f"   Lucro anterior: ${asset_stats['profit']:+.2f}")
            logger.info(f"   Lucro após soma: ${asset_stats['profit'] + net_profit_for_sequence:+.2f}")
        
        self.session_stats['operations_total'] += 1
        asset_stats['operations'] += 1
        asset_stats['profit'] += net_profit_for_sequence
        
        asset_state = self.asset_states[symbol]
        if asset_state.current_sequence > asset_stats['best_sequence']:
            asset_stats['best_sequence'] = asset_state.current_sequence
        
        if operation_result == "victory":
            self.session_stats['operations_won'] += 1
            asset_stats['wins'] += 1
        else:
            self.session_stats['operations_lost'] += 1
            asset_stats['losses'] += 1

    def calculate_session_metrics(self):
        """Calcula as métricas financeiras e de assertividade da sessão."""
        balance_change = self.balance - self.initial_balance
        total_ops = self.session_stats['operations_total']
        win_rate = (self.session_stats['operations_won'] / total_ops * 100) if total_ops > 0 else 0
        return {'balance_change': balance_change, 'balance_change_percent': (balance_change / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
                'win_rate': win_rate, 'total_operations': total_ops, 'total_wins': self.session_stats['operations_won'],
                'total_losses': self.session_stats['operations_lost']}

    async def _process_contract_result(self, contract: dict, contract_data: dict, symbol: str):
        """Processa os dados de um contrato finalizado."""
        if contract.get("status") == "finished": return
        
        profit = 0.0
        if contract_data.get("status") == "won":
            profit = float(contract_data.get("payout", 0)) - float(contract_data.get("buy_price", 0))
        else:
            profit = -float(contract_data.get("buy_price", 0))
        
        contract["status"] = "finished"
        contract["profit"] = profit
        logger.info(f"{'✅' if profit > 0 else '❌'} {symbol} {contract['type']} {'GANHOU' if profit > 0 else 'PERDEU'}: ${profit:+.2f}")

    def get_asset_status_summary(self, symbol: str) -> str:
        """Retorna uma string com o resumo do status do ativo"""
        asset_state = self.asset_states[symbol]
        status_parts = []
        
        if self.is_asset_in_cooldown(symbol):
            remaining = asset_state.cooldown_end_time - self.get_current_time()
            status_parts.append(f"cooldown({remaining:.0f}s)")
        
        if any(c.get("status") != "finished" for c in asset_state.active_contracts):
            active_count = len([c for c in asset_state.active_contracts if c.get("status") != "finished"])
            if self.dual_entry:
                status_parts.append(f"dual_active({active_count})")
            else:
                status_parts.append(f"active({active_count})")
        
        current_seq = asset_state.current_sequence
        if self.operation_type != "FIXED" and current_seq > 1:
            if self.dual_entry:
                status_parts.append(f"dual_S{current_seq}")
            else:
                status_parts.append(f"S{current_seq}")
        
        return " | ".join(status_parts) if status_parts else "ready"

    def _log_data_status(self):
        """Log do status de coleta de dados"""
        connection_status = "🟢 CONECTADO" if self.is_connected and self.connection_stable else "🔴 DESCONECTADO"
        entry_mode = "DUAL ENTRY (Ticks Rápidos)" if self.dual_entry and self.is_tick_mode else ("DUAL ENTRY (Operação Única)" if self.dual_entry else "SINGLE ENTRY")
        
        logger.info(f"📊 STATUS ({connection_status} | {entry_mode}):")
        
        min_needed = 10
        for symbol in self.symbols:
            candle_count = len(self.candle_cache.get(symbol, []))
            tick_count = len(self.tick_cache.get(symbol, []))
            data_status = "✅ PRONTO" if candle_count >= min_needed else f"⏳ {candle_count}/{min_needed}"
            operation_status = self.get_asset_status_summary(symbol)
            logger.info(f"    {symbol}: {data_status} | {tick_count} ticks | {operation_status}")
            
        ready_count = sum(1 for s in self.symbols if not self.is_asset_busy(s))
        logger.info(f"🎯 Monitorando sinais | {ready_count}/{len(self.symbols)} ativos disponíveis")

    def show_session_summary(self):
        """Exibe o resumo completo da sessão"""
        metrics = self.calculate_session_metrics()
        logger.info("=" * 80)
        logger.info("📊 RESUMO DA SESSÃO ATUAL")
        logger.info("=" * 80)
        logger.info(f"💰 PERFORMANCE FINANCEIRA:\n    💵 Saldo inicial: ${self.initial_balance:.2f}\n    💵 Saldo atual: ${self.balance:.2f}")
        logger.info(f"    {'📈' if metrics['balance_change'] >= 0 else '📉'} Variação: ${metrics['balance_change']:+.2f} ({metrics['balance_change_percent']:+.2f}%)")
        
        if self.dual_entry:
            mode_detail = " (Ticks Rápidos)" if self.is_tick_mode else ""
            logger.info(f"🔄 MODO: DUAL ENTRY{mode_detail}")
            logger.info("   💡 Vitória quando CALL OU PUT ganha")
            logger.info("   💡 Martingale apenas se AMBOS perdem")
        else:
            logger.info("⚡ MODO: SINGLE ENTRY")
        
        logger.info("🎯 ASSERTIVIDADE DAS OPERAÇÕES:\n    📊 Total: {total_operations} | ✅ Vitórias: {total_wins} | ❌ Derrotas: {total_losses}".format(**metrics))
        
        if metrics['total_operations'] > 0:
            logger.info(f"    {'🔥' if metrics['win_rate'] >= 60 else '⚡'} Taxa de acerto: {metrics['win_rate']:.1f}%")
        
        logger.info("🏆 PERFORMANCE POR ATIVO:")
        total_profit_by_assets = 0.0
        
        for asset_symbol, stats in self.session_stats['asset_stats'].items():
            if stats['operations'] > 0:
                win_rate = (stats['wins'] / stats['operations'] * 100)
                logger.info(f"    📊 {asset_symbol}: {stats['wins']}/{stats['operations']} ({win_rate:.1f}%) | {'💚' if stats['profit'] >= 0 else '❤️'} ${stats['profit']:+.2f} | (max: S{stats['best_sequence']})")
                total_profit_by_assets += stats['profit']
        
        # 🎯 VERIFICAÇÃO DE CONSISTÊNCIA (tolerância para arredondamentos)
        difference = abs(total_profit_by_assets - metrics['balance_change'])
        if difference > 0.10:  # Tolerância de 10 centavos para arredondamentos
            logger.warning("=" * 80)
            logger.warning("⚠️ INCONSISTÊNCIA DETECTADA:")
            logger.warning(f"   📊 Soma dos lucros por ativo: ${total_profit_by_assets:+.2f}")
            logger.warning(f"   💰 Variação real do saldo: ${metrics['balance_change']:+.2f}")
            logger.warning(f"   🔍 Diferença: ${difference:.2f}")
            if difference > 5.00:
                logger.warning("   🚨 DIFERENÇA CRÍTICA - Verificar cálculos!")
            logger.warning("=" * 80)
        else:
            logger.info(f"✅ Verificação: Soma dos lucros por ativo (${total_profit_by_assets:+.2f}) confere com variação do saldo!")
            if difference > 0.01:
                logger.info(f"   ℹ️ Pequena diferença de ${difference:.2f} (tolerável - arredondamentos)")
        
        if hasattr(self, 'session_start_time'):
            minutes, seconds = divmod(int(self.get_current_time() - self.session_start_time), 60)
            hours, minutes = divmod(minutes, 60)
            logger.info(f"⏰ Tempo de sessão: {f'{hours}h ' if hours > 0 else ''}{minutes}m {seconds}s")
        
        logger.info("=" * 80)

    async def run_asset_operation(self, symbol: str, direction: str):
        """Gerencia o ciclo de vida completo de uma operação para um ativo - SINGLE ENTRY."""
        asset_state = self.asset_states[symbol]
        
        if not self.connection_stable:
            logger.warning(f"⚠️ Não é possível operar em {symbol}: conexão não estável. Aguardando estabilidade.")
            return

        if asset_state.current_sequence == 1:
            asset_state.martingale_loss_accumulator = 0.0; asset_state.last_entry_direction = direction
        
        if await self.execute_entry(symbol, asset_state.last_entry_direction):
            await self.wait_for_results(symbol)
            await self.get_balance()
            result = self.analyze_operation_result(symbol)
            
            if result == "continue_immediate":
                delay = 0.2 if self.is_tick_mode else 1.0
                logger.info(f"⚡ {symbol}: Continuando martingale imediato...")
                await asyncio.sleep(delay)
                await self.run_asset_operation(symbol, asset_state.last_entry_direction)
            elif result in ["victory", "max_sequence"]:
                self.show_session_summary()
        else:
            if self.operation_type == "MARTINGALE_IMMEDIATE":
                logger.warning(f"⚠️ Falha na entrada de {symbol} na sequência S{asset_state.current_sequence}. Resetando sequência ou aplicando cooldown.")
                asset_state.martingale_loss_accumulator += -self.calculate_amount(symbol)
                self.update_session_stats(symbol, "failed_entry", asset_state.martingale_loss_accumulator)
                asset_state.current_sequence = 1
                asset_state.martingale_loss_accumulator = 0.0
                self.put_asset_in_cooldown(symbol)
                self.show_session_summary()

    async def execute_entry(self, symbol: str, direction: str):
        """Prepara e executa a entrada da operação."""
        asset_state = self.asset_states[symbol]; current_sequence = asset_state.current_sequence
        amount = self.calculate_amount(symbol)
        if amount > self.balance: logger.error(f"❌ Saldo insuficiente!"); return False
        asset_state.active_contracts = []
        actual_direction = self.get_martingale_direction(direction, current_sequence)
        logger.info(f"🎯 {symbol} S{current_sequence}: Entrada {actual_direction} | ${amount:.2f}")
        contract_info = await self.get_proposal_and_buy(symbol, actual_direction, amount)
        if contract_info:
            asset_state.active_contracts = [contract_info]; return True
        return False

    def analyze_operation_result(self, symbol: str) -> str:
        """Analisa o resultado de uma operação e decide o próximo passo (martingale, cooldown, etc)."""
        asset_state = self.asset_states[symbol]
        if not asset_state.active_contracts or not all(c.get("status") == "finished" for c in asset_state.active_contracts):
            logger.error(f"Estado inconsistente para {symbol}: Nem todos os contratos foram finalizados.")
            return "error" 
            
        # 🎯 CORREÇÃO: Calcular lucro líquido desta operação
        net_profit = sum(c.get("profit", 0) for c in asset_state.active_contracts)
        operation_won = net_profit > 0
        
        # Para decisão de continuar martingale, considerar perdas acumuladas
        full_sequence_result = net_profit + asset_state.martingale_loss_accumulator
        
        if operation_won:
            # 🎯 CORREÇÃO: Calcular resultado final da sequência completa
            full_sequence_result = net_profit + asset_state.martingale_loss_accumulator
            
            if asset_state.martingale_loss_accumulator < 0:
                # Houve martingale - usar resultado líquido de toda a sequência
                logger.info(f"🎉 {symbol}: VITÓRIA! Resultado final da sequência: ${full_sequence_result:+.2f}")
                logger.info(f"   💰 (Lucro desta operação: ${net_profit:+.2f} | Perdas anteriores: ${asset_state.martingale_loss_accumulator:+.2f})")
                self.update_session_stats(symbol, "victory", full_sequence_result)
            else:
                # Operação única sem martingale - usar apenas lucro desta operação
                logger.info(f"🎉 {symbol}: VITÓRIA! Lucro desta operação: ${net_profit:+.2f}")
                self.update_session_stats(symbol, "victory", net_profit)
            
            asset_state.current_sequence = 1
            asset_state.martingale_loss_accumulator = 0.0
            self.put_asset_in_cooldown(symbol)
            return "victory"
        else:
            if asset_state.current_sequence >= self.max_martingale_sequence:
                logger.warning(f"🛑 {symbol}: Limite de martingale atingido! Perda total da sequência: ${full_sequence_result:+.2f}")
                
                # 🎯 CORREÇÃO: Para estatísticas, usar o resultado de toda a sequência
                self.update_session_stats(symbol, "max_sequence", full_sequence_result)
                
                asset_state.current_sequence = 1
                asset_state.martingale_loss_accumulator = 0.0
                self.put_asset_in_cooldown(symbol)
                return "max_sequence"
            else:
                asset_state.martingale_loss_accumulator += net_profit
                asset_state.current_sequence += 1
                logger.info(f"📈 {symbol}: Martingale → S{asset_state.current_sequence}. Perda acumulada: ${asset_state.martingale_loss_accumulator:+.2f}")
                return "continue_immediate" if self.operation_type == "MARTINGALE_IMMEDIATE" else "continue"

    async def _post_connection_setup(self):
        """Tarefas a serem executadas após uma conexão ou reconexão bem-sucedida."""
        try:
            await self.authorize()
            await self._synchronize_clock()
            await self.get_balance()
            if self.initial_balance == 0: self.initial_balance = self.balance
            
            await self.load_historical_data()
            await self.subscribe_to_ticks()
            
            self.is_connected = True
            self.connection_stable = True
            logger.info("🟢 Robô pronto e operacional.")
            return True
        except Exception as e:
            logger.error(f"❌ Falha no setup pós-conexão: {e}")
            self.is_connected = False
            self.connection_stable = False
            return False

    async def run(self):
        """Método principal para iniciar e gerenciar o robô."""
        try:
            self._log_config()
            if not await self.connect():
                logger.error("❌ Falha na conexão inicial. Encerrando."); return
            
            self.processor_task = asyncio.create_task(self._message_processor())
            self.health_monitor_task = asyncio.create_task(self._connection_health_monitor())
            
            if not await self._post_connection_setup():
                logger.error("❌ Setup inicial falhou, encerrando robô."); return
            
            self.session_start_time = self.get_current_time()
            
            await self.main_loop()

        except Exception as e:
            logger.critical(f"❌ Erro crítico no run: {e}", exc_info=True)
        finally:
            logger.info("🏁 SESSÃO FINALIZADA")
            if hasattr(self, 'initial_balance') and self.initial_balance > 0:
                self.show_session_summary()
            if self.websocket and not self.websocket.closed:
                await self.websocket.close(); logger.info("🔌 Conexão fechada")

def main():
    """Função principal para executar o robô."""
    try:
        # 🎯 LOG DAS VARIÁVEIS DE ESTRATÉGIA PARA DEBUG
        logger.info("🔍 Verificando configuração de estratégias:")
        logger.info(f"   STRATEGY_1_ACTIVE (RSI): {os.getenv('STRATEGY_1_ACTIVE', 'false')}")
        logger.info(f"   STRATEGY_2_ACTIVE (Candle): {os.getenv('STRATEGY_2_ACTIVE', 'true')}")
        logger.info(f"   STRATEGY_3_ACTIVE (MA): {os.getenv('STRATEGY_3_ACTIVE', 'false')}")
        
        if os.getenv('STRATEGY_2_ACTIVE', 'true').lower() == 'true':
            logger.info("📊 Configuração de padrões de candle:")
            call_pattern = []
            put_pattern = []
            for i in range(1, 6):  # Mostrar apenas os primeiros 5
                call_color = os.getenv(f'CALL_CANDLE_{i}', 'ANY')
                put_color = os.getenv(f'PUT_CANDLE_{i}', 'ANY')
                if call_color in ['RED', 'GREEN', 'ANY']:
                    call_pattern.append(call_color)
                if put_color in ['RED', 'GREEN', 'ANY']:
                    put_pattern.append(put_color)
            logger.info(f"   📈 CALL: {' → '.join(call_pattern) if call_pattern else 'Não configurado'}")
            logger.info(f"   📉 PUT: {' → '.join(put_pattern) if put_pattern else 'Não configurado'}")
        
        bot = DerivMultiAssetBot()
        asyncio.run(bot.run())
    except (ValueError, FileNotFoundError) as e:
        logger.critical(f"❌ Erro de configuração: {e}")
    except KeyboardInterrupt:
        logger.info("\n👋 Encerrando robô...")
    except Exception as e:
        logger.critical(f"❌ Erro inesperado na inicialização: {e}", exc_info=True)

if __name__ == "__main__":
    main()

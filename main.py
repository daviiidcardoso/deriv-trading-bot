import asyncio
import websockets
import json
import logging
import time
import os
import math
from dotenv import load_dotenv

# Carregar variáveis do arquivo .env
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DerivMartingaleBotDirectional:
    def __init__(self):
        """
        Robô Deriv Martingale - VERSÃO DIRECIONAL AVANÇADA
        
        NOVAS FUNCIONALIDADES:
        • Estratégia HEDGE (atual) ou DIRECIONAL (nova)
        • Martingale diferenciado (uma mantém direção, outra inverte)
        • Martingale IMEDIATO após resultado ou no próximo sinal
        • Configuração flexível de direções por operação
        • Todas as proteções e funcionalidades existentes mantidas
        """
        # Carregar configurações do .env
        self.api_token = os.getenv("DERIV_API_TOKEN")
        self.app_id = int(os.getenv("DERIV_APP_ID", 1089))
        
        # Configurações do ativo
        self.symbol = os.getenv("SYMBOL", "STPRNG5")
        self.duration = int(os.getenv("DURATION", 15))
        self.duration_unit = os.getenv("DURATION_UNIT", "s")
        
        # Configurações do martingale
        self.amount_type = os.getenv("AMOUNT_TYPE", "FIXED").upper()
        self.initial_amount = float(os.getenv("INITIAL_AMOUNT", 0.35))
        
        env_percentage = os.getenv("INITIAL_PERCENTAGE")
        if env_percentage is not None:
            self.initial_percentage = float(env_percentage)
        else:
            self.initial_percentage = 2.0
            
        self.min_amount = float(os.getenv("MIN_AMOUNT", 0.35))
        self.max_amount = float(os.getenv("MAX_AMOUNT", 50.0))
        self.multiplier = float(os.getenv("MULTIPLIER", 2.4))
        
        self.max_martingale_rounds = int(os.getenv("MAX_MARTINGALE_ROUNDS", 4))
        self.max_rounds = int(os.getenv("MAX_ROUNDS", 999))
        
        self.stop_loss = float(os.getenv("STOP_LOSS", 20.0))
        self.stop_win = float(os.getenv("STOP_WIN", 0.0))
        
        # ========== NOVAS CONFIGURAÇÕES DIRECIONAIS ==========
        
        # Modo de estratégia
        self.strategy_mode = os.getenv("STRATEGY_MODE", "HEDGE").upper()  # HEDGE ou DIRECTIONAL
        
        # Direção inicial (para modo direcional)
        self.initial_direction = os.getenv("INITIAL_DIRECTION", "CALL").upper()  # CALL ou PUT
        
        # Modo de martingale
        self.martingale_mode = os.getenv("MARTINGALE_MODE", "NEXT_SIGNAL").upper()  # IMMEDIATE ou NEXT_SIGNAL
        
        # Direções no martingale para cada operação
        self.op1_martingale_direction = os.getenv("OP1_MARTINGALE_DIRECTION", "SAME").upper()  # SAME ou INVERT
        self.op2_martingale_direction = os.getenv("OP2_MARTINGALE_DIRECTION", "INVERT").upper()  # SAME ou INVERT
        
        # =====================================================
        
        # Configurações de ciclo
        self.rounds_interval = int(os.getenv("ROUNDS_INTERVAL", 30))
        self.cycles_interval = int(os.getenv("CYCLES_INTERVAL", 60))
        self.auto_restart = os.getenv("AUTO_RESTART", "true").lower() == "true"
        self.max_cycles = int(os.getenv("MAX_CYCLES", 999))
        
        # Timing sincronizado
        self.timing_mode = os.getenv("TIMING_MODE", "IMMEDIATE").upper()
        execution_seconds_str = os.getenv("EXECUTION_SECONDS", "")
        self.execution_seconds = []
        if execution_seconds_str:
            try:
                self.execution_seconds = [int(s.strip()) for s in execution_seconds_str.split(",")]
                self.execution_seconds = [s for s in self.execution_seconds if 0 <= s <= 59]
                self.execution_seconds.sort()
            except ValueError:
                logger.warning("⚠️ EXECUTION_SECONDS inválido, usando modo IMMEDIATE")
                self.timing_mode = "IMMEDIATE"
        
        if self.timing_mode == "SCHEDULED" and not self.execution_seconds:
            logger.warning("⚠️ TIMING_MODE=SCHEDULED mas EXECUTION_SECONDS vazio, usando modo IMMEDIATE")
            self.timing_mode = "IMMEDIATE"
            
        self.max_wait_time = int(os.getenv("MAX_WAIT_TIME", 60))
        self.timing_tolerance = float(os.getenv("TIMING_TOLERANCE", 0.5))
        
        # Intervalos efetivos
        if self.timing_mode == "SCHEDULED":
            self.effective_rounds_interval = 0
            self.effective_cycles_interval = 0
            logger.info("🎯 TIMING SINCRONIZADO ATIVO: Intervalos gerenciados pelo timer")
        else:
            self.effective_rounds_interval = self.rounds_interval
            self.effective_cycles_interval = self.cycles_interval
            
        # Configurações de timing e segurança
        self.delay_between_ops = float(os.getenv("DELAY_BETWEEN_OPS", 0.05))
        self.use_fast_mode = os.getenv("FAST_MODE", "true").lower() == "true"
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.max_verification_attempts = int(os.getenv("MAX_VERIFICATION_ATTEMPTS", 10))
        self.verification_timeout = int(os.getenv("VERIFICATION_TIMEOUT", 5))
        
        # Estado do robô
        self.websocket = None
        self.is_connected = False
        self.balance = 0
        self.balance_before_round = 0
        self.initial_balance = 0
        self.current_amount = 0
        self.current_percentage = self.initial_percentage
        self.current_round = 1
        self.current_cycle = 1
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.active_contracts = []
        self.subscribed_contracts = set()
        self.processed_contracts = {}
        
        # ========== NOVOS ESTADOS DIRECIONAIS ==========
        
        # Direções atuais de cada operação
        self.current_op1_direction = None  # Será definida na primeira execução
        self.current_op2_direction = None  # Será definida na primeira execução
        
        # Flags de controle
        self.pending_martingale = False  # Se há martingale pendente para execução imediata
        self.last_result = None  # Resultado da última rodada para decisão de martingale
        
        # ==============================================
        
        # Contadores para estatísticas
        self.martingale_resets = 0
        self.total_sequences = 0
        self.total_strategies = 0
        self.successful_strategies = 0
        self.failed_strategies = 0
        
        # WebSocket URL
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        
        # Validar configurações
        self._validate_config()
        self._log_config()
        
    def _validate_config(self):
        """Valida configurações do .env incluindo novas configurações direcionais"""
        if not self.api_token or self.api_token == "SEU_TOKEN_AQUI":
            raise ValueError("❌ Configure DERIV_API_TOKEN no arquivo .env!")
            
        if self.duration_unit not in ["t", "s"]:
            raise ValueError("❌ DURATION_UNIT deve ser 't' (ticks) ou 's' (segundos)!")
            
        if self.amount_type not in ["FIXED", "PERCENTAGE"]:
            raise ValueError("❌ AMOUNT_TYPE deve ser 'FIXED' ou 'PERCENTAGE'!")
            
        if self.amount_type == "FIXED":
            if self.initial_amount <= 0:
                raise ValueError("❌ INITIAL_AMOUNT deve ser maior que 0!")
        else:
            if self.initial_percentage <= 0 or self.initial_percentage > 50:
                raise ValueError("❌ INITIAL_PERCENTAGE deve estar entre 0.1 e 50!")
                
        if self.min_amount <= 0:
            raise ValueError("❌ MIN_AMOUNT deve ser maior que 0!")
            
        if self.max_amount <= self.min_amount:
            raise ValueError("❌ MAX_AMOUNT deve ser maior que MIN_AMOUNT!")
            
        if self.multiplier < 1.1:
            raise ValueError("❌ MULTIPLIER deve ser pelo menos 1.1!")
            
        if self.max_martingale_rounds < 1:
            raise ValueError("❌ MAX_MARTINGALE_ROUNDS deve ser pelo menos 1!")
            
        if self.timing_mode not in ["IMMEDIATE", "SCHEDULED"]:
            raise ValueError("❌ TIMING_MODE deve ser 'IMMEDIATE' ou 'SCHEDULED'!")
            
        if self.timing_mode == "SCHEDULED":
            if not self.execution_seconds:
                raise ValueError("❌ TIMING_MODE=SCHEDULED requer EXECUTION_SECONDS válidos!")
            if self.max_wait_time < 10:
                raise ValueError("❌ MAX_WAIT_TIME deve ser pelo menos 10 segundos!")
            if self.timing_tolerance < 0.1 or self.timing_tolerance > 5.0:
                raise ValueError("❌ TIMING_TOLERANCE deve estar entre 0.1 e 5.0 segundos!")
        
        # ========== NOVAS VALIDAÇÕES DIRECIONAIS ==========
        
        if self.strategy_mode not in ["HEDGE", "DIRECTIONAL"]:
            raise ValueError("❌ STRATEGY_MODE deve ser 'HEDGE' ou 'DIRECTIONAL'!")
            
        if self.initial_direction not in ["CALL", "PUT"]:
            raise ValueError("❌ INITIAL_DIRECTION deve ser 'CALL' ou 'PUT'!")
            
        if self.martingale_mode not in ["IMMEDIATE", "NEXT_SIGNAL"]:
            raise ValueError("❌ MARTINGALE_MODE deve ser 'IMMEDIATE' ou 'NEXT_SIGNAL'!")
            
        if self.op1_martingale_direction not in ["SAME", "INVERT"]:
            raise ValueError("❌ OP1_MARTINGALE_DIRECTION deve ser 'SAME' ou 'INVERT'!")
            
        if self.op2_martingale_direction not in ["SAME", "INVERT"]:
            raise ValueError("❌ OP2_MARTINGALE_DIRECTION deve ser 'SAME' ou 'INVERT'!")
            
        # =================================================
            
    def _log_config(self):
        """Exibe configuração carregada com novas configurações direcionais"""
        logger.info("🛡️ ROBÔ MARTINGALE DIRECIONAL - VERSÃO SUPER-ROBUSTA:")
        logger.info(f"   💎 Símbolo: {self.symbol}")
        logger.info(f"   ⏱️  Expiração: {self.duration} {self.duration_unit}")
        
        # Log do sistema de valores
        if self.amount_type == "FIXED":
            logger.info(f"   💰 Modo: VALOR FIXO (${self.initial_amount})")
        else:
            logger.info(f"   📊 Modo: PERCENTUAL ({self.initial_percentage}% da banca)")
            logger.info(f"   💰 Limites: ${self.min_amount} - ${self.max_amount}")
            
        logger.info(f"   📈 Multiplicador: {self.multiplier}x")
        logger.info(f"   🎯 Limite martingale: {self.max_martingale_rounds} rounds")
        
        if self.max_rounds < 999:
            logger.info(f"   🎲 Máx rounds/ciclo: {self.max_rounds}")
        else:
            logger.info(f"   🎲 Máx rounds/ciclo: Ilimitado")
            
        logger.info(f"   🛑 Stop loss: ${self.stop_loss}")
        
        if self.stop_win > 0:
            logger.info(f"   🎉 Stop win: ${self.stop_win}")
        else:
            logger.info(f"   🎉 Stop win: Desabilitado")
            
        logger.info(f"   🔄 Ciclos máx: {self.max_cycles}")
        logger.info(f"   ⚡ Modo rápido: {'✅' if self.use_fast_mode else '❌'}")
        logger.info(f"   🔍 Debug: {'✅' if self.debug_mode else '❌'}")
        logger.info(f"   🛡️ Tentativas verificação: {self.max_verification_attempts}")
        logger.info(f"   🔄 Auto restart: {'✅' if self.auto_restart else '❌'}")
        
        # ========== NOVOS LOGS DIRECIONAIS ==========
        
        logger.info("")
        logger.info("🎯 CONFIGURAÇÕES ESTRATÉGICAS:")
        logger.info(f"   • Modo de estratégia: {self.strategy_mode}")
        
        if self.strategy_mode == "DIRECTIONAL":
            logger.info(f"   • Direção inicial: {self.initial_direction}")
        else:
            logger.info(f"   • Operações: CALL + PUT (hedge tradicional)")
            
        logger.info(f"   • Martingale: {self.martingale_mode}")
        logger.info(f"   • OP1 no martingale: {self.op1_martingale_direction}")
        logger.info(f"   • OP2 no martingale: {self.op2_martingale_direction}")
        
        # ===========================================
        
        logger.info("")
        logger.info("🎯 FUNCIONALIDADES ESPECIAIS:")
        logger.info(f"   • Limite de {self.max_martingale_rounds} martingales consecutivos")
        logger.info(f"   • Ao atingir limite + perder: RESETA para valor inicial")
        logger.info(f"   • ASSERTIVIDADE: Rastreia eficácia da estratégia")
        
        if self.strategy_mode == "DIRECTIONAL":
            logger.info(f"   • MODO DIRECIONAL: 2x {self.initial_direction} ou 2x perda")
            logger.info(f"   • MARTINGALE DIFERENCIADO: OP1={self.op1_martingale_direction}, OP2={self.op2_martingale_direction}")
        
        if self.martingale_mode == "IMMEDIATE":
            logger.info(f"   • MARTINGALE IMEDIATO: Executa após resultado da operação")
        else:
            logger.info(f"   • MARTINGALE NO SINAL: Executa no próximo timing configurado")
        
        # Timing
        logger.info("")
        logger.info("⏰ SISTEMA DE TIMING:")
        if self.timing_mode == "IMMEDIATE":
            logger.info("   • Modo: IMEDIATO (máxima velocidade)")
            logger.info(f"   • Intervalo entre rodadas: {self.effective_rounds_interval}s")
            logger.info(f"   • Intervalo entre ciclos: {self.effective_cycles_interval}s")
        else:
            logger.info("   • Modo: SINCRONIZADO (segundos específicos)")
            logger.info(f"   • Segundos de execução: {self.execution_seconds}")
            logger.info(f"   • Frequência: {len(self.execution_seconds)} oportunidades/minuto")
            logger.info(f"   • Tempo máximo de espera: {self.max_wait_time}s")
            logger.info(f"   • Tolerância: ±{self.timing_tolerance}s")
            logger.info("   • Intervalos: IGNORADOS (timer controla tudo)")
        
        if self.amount_type == "PERCENTAGE":
            logger.info("📊 SISTEMA DE JUROS COMPOSTOS + INFERÊNCIA INTELIGENTE ATIVADO!")

    def determine_operation_directions(self):
        """Determina as direções das operações baseado no modo e round atual"""
        if self.strategy_mode == "HEDGE":
            # Modo hedge tradicional: sempre CALL + PUT
            return "CALL", "PUT"
        else:
            # Modo direcional
            if self.current_round == 1:
                # Primeira rodada: ambas na direção inicial
                self.current_op1_direction = self.initial_direction
                if self.initial_direction == "CALL":
                    self.current_op2_direction = "CALL"
                else:
                    self.current_op2_direction = "PUT"
            else:
                # Martingale: aplicar regras de direção
                previous_op1 = self.current_op1_direction
                previous_op2 = self.current_op2_direction
                
                # OP1
                if self.op1_martingale_direction == "SAME":
                    self.current_op1_direction = previous_op1
                else:  # INVERT
                    self.current_op1_direction = "PUT" if previous_op1 == "CALL" else "CALL"
                
                # OP2
                if self.op2_martingale_direction == "SAME":
                    self.current_op2_direction = previous_op2
                else:  # INVERT
                    self.current_op2_direction = "PUT" if previous_op2 == "CALL" else "CALL"
            
            return self.current_op1_direction, self.current_op2_direction

    async def connect(self):
        """Conecta ao WebSocket da Deriv com retry automático ultra-robusto"""
        max_connection_attempts = 5
        for attempt in range(max_connection_attempts):
            try:
                if self.websocket and not self.websocket.closed:
                    await self.websocket.close()
                    
                logger.info(f"🔌 Tentativa de conexão {attempt + 1}/{max_connection_attempts}")
                self.websocket = await websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                self.is_connected = True
                logger.info("✅ Conectado ao WebSocket da Deriv")
                
                await self.authorize()
                await self.get_balance()
                return
                
            except Exception as e:
                logger.error(f"❌ Tentativa {attempt + 1}/{max_connection_attempts} falhou: {e}")
                self.is_connected = False
                if attempt < max_connection_attempts - 1:
                    wait_time = min(5 + (attempt * 2), 15)
                    logger.info(f"⏳ Aguardando {wait_time}s antes da próxima tentativa...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception("Falha ao conectar após múltiplas tentativas")

    async def ensure_connection(self):
        """Verifica e reconecta se necessário"""
        try:
            if not self.websocket or self.websocket.closed:
                logger.warning("🔄 Conexão perdida, reconectando...")
                await self.connect()
                return True
                
            pong = await self.websocket.ping()
            await asyncio.wait_for(pong, timeout=5)
            return True
            
        except Exception as e:
            logger.error(f"❌ Problema de conexão: {e}")
            try:
                await self.connect()
                return True
            except Exception as reconnect_error:
                logger.error(f"❌ Falha na reconexão: {reconnect_error}")
                return False
            
    async def authorize(self):
        """Autoriza a conexão com retry"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                auth_request = {
                    "authorize": self.api_token,
                    "req_id": int(time.time() * 1000)
                }
                
                await self.websocket.send(json.dumps(auth_request))
                response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
                auth_response = json.loads(response)
                
                if "error" in auth_response:
                    logger.error(f"❌ Erro na autorização (tentativa {attempt + 1}): {auth_response['error']}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise Exception("Falha na autorização após múltiplas tentativas")
                else:
                    logger.info("✅ Autorização bem-sucedida")
                    return True
                    
            except Exception as e:
                logger.error(f"❌ Erro na autorização (tentativa {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)
                else:
                    raise Exception("Falha na autorização após múltiplas tentativas")
            
    async def get_balance_with_retry(self, max_attempts=5):
        """Obtém saldo com retry robusto e reconexão automática"""
        for attempt in range(max_attempts):
            try:
                if not await self.ensure_connection():
                    continue
                    
                balance_request = {
                    "balance": 1,
                    "req_id": int(time.time() * 1000)
                }
                
                await self.websocket.send(json.dumps(balance_request))
                response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
                balance_response = json.loads(response)
                
                if "balance" in balance_response:
                    old_balance = self.balance
                    self.balance = float(balance_response["balance"]["balance"])
                    currency = balance_response['balance']['currency']
                    
                    if self.initial_balance == 0:
                        self.initial_balance = self.balance
                        logger.info(f"💰 Saldo inicial: {self.balance} {currency}")
                    else:
                        logger.info(f"💰 Saldo atual: {self.balance} {currency}")
                    return True
                else:
                    logger.error(f"❌ Erro ao obter saldo (tentativa {attempt + 1}): {balance_response}")
                    
            except Exception as e:
                logger.error(f"❌ Tentativa {attempt + 1}/{max_attempts} de obter saldo falhou: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 + attempt)
                    
        logger.error("❌ Falha crítica: Não foi possível obter saldo!")
        return False

    async def get_balance(self):
        """Wrapper para compatibilidade"""
        return await self.get_balance_with_retry()

    def calculate_current_amount(self):
        """Calcula o valor atual da entrada baseado no modo configurado"""
        if self.amount_type == "FIXED":
            if self.current_round == 1:
                calculated_amount = self.initial_amount
            else:
                calculated_amount = self.initial_amount * (self.multiplier ** (self.current_round - 1))
        else:
            if self.current_round == 1:
                calculated_amount = (self.balance * self.current_percentage) / 100
            else:
                martingale_percentage = self.current_percentage * (self.multiplier ** (self.current_round - 1))
                calculated_amount = (self.balance * martingale_percentage) / 100
        
        calculated_amount = max(self.min_amount, min(self.max_amount, calculated_amount))
        calculated_amount = round(calculated_amount, 2)
        
        return calculated_amount

    def log_amount_calculation(self):
        """Exibe log detalhado do cálculo do valor"""
        if self.amount_type == "FIXED":
            if self.current_round == 1:
                logger.info(f"💰 Cálculo FIXO: ${self.initial_amount} (valor inicial)")
            else:
                base_calc = self.initial_amount * (self.multiplier ** (self.current_round - 1))
                logger.info(f"💰 Cálculo FIXO: ${self.initial_amount} × {self.multiplier}^{self.current_round-1} = ${base_calc:.2f}")
        else:
            if self.current_round == 1:
                percentage_used = self.current_percentage
                logger.info(f"📊 Cálculo PERCENTUAL: ${self.balance} × {percentage_used}% = ${self.current_amount}")
            else:
                percentage_used = self.current_percentage * (self.multiplier ** (self.current_round - 1))
                logger.info(f"📊 Cálculo PERCENTUAL: ${self.balance} × {percentage_used:.2f}% = ${self.current_amount}")
                
        original_calc = self.current_amount
        if self.amount_type == "PERCENTAGE":
            if self.current_round == 1:
                original_calc = (self.balance * self.current_percentage) / 100
            else:
                martingale_percentage = self.current_percentage * (self.multiplier ** (self.current_round - 1))
                original_calc = (self.balance * martingale_percentage) / 100
                
        if original_calc < self.min_amount:
            logger.info(f"⬆️ Valor ajustado para mínimo: ${self.min_amount}")
        elif original_calc > self.max_amount:
            logger.info(f"⬇️ Valor ajustado para máximo: ${self.max_amount}")

    async def wait_for_execution_time(self):
        """Sistema de timing sincronizado"""
        if self.timing_mode == "IMMEDIATE":
            logger.info("⚡ Modo IMEDIATO: Executando agora")
            return True
            
        logger.info("⏰ Modo AGENDADO: Sincronizando timing...")
        
        start_wait = time.time()
        
        while True:
            now = time.time()
            current_second = int(now) % 60
            microsecond_fraction = now - int(now)
            
            next_execution_second = None
            for target_second in self.execution_seconds:
                if abs(current_second - target_second) <= self.timing_tolerance:
                    logger.info(f"🎯 EXECUTANDO no segundo {current_second} (alvo: {target_second})")
                    return True
                    
                if target_second > current_second:
                    next_execution_second = target_second
                    break
            
            if next_execution_second is None:
                next_execution_second = self.execution_seconds[0] + 60
                
            seconds_to_wait = next_execution_second - current_second - microsecond_fraction
            
            if next_execution_second > 59:
                seconds_to_wait = (60 - current_second) + (next_execution_second - 60) - microsecond_fraction
                next_execution_second = next_execution_second - 60
                
            wait_elapsed = time.time() - start_wait
            if wait_elapsed > self.max_wait_time:
                logger.warning(f"⚠️ Timeout de espera ({self.max_wait_time}s) - Executando agora")
                return True
                
            if seconds_to_wait > 1:
                logger.info(f"⏳ Aguardando {seconds_to_wait:.1f}s para executar no segundo {next_execution_second}")
                
            if seconds_to_wait > 2:
                await asyncio.sleep(1)
            elif seconds_to_wait > 0.5:
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.01)
                
        return True

    async def get_proposal_and_buy(self, contract_type, amount):
        """Obtém proposta e compra contrato com validação extra e reconexão"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if not await self.ensure_connection():
                    logger.error(f"❌ Falha de conexão na tentativa {attempt + 1}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(2)
                        continue
                    else:
                        return None
                
                timestamp = int(time.time() * 1000)
                
                proposal_request = {
                    "proposal": 1,
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": "USD",
                    "duration": self.duration,
                    "duration_unit": self.duration_unit,
                    "symbol": self.symbol,
                    "req_id": timestamp
                }
                
                if self.debug_mode:
                    logger.info(f"🔍 Enviando proposta {contract_type}: {proposal_request}")
                
                await self.websocket.send(json.dumps(proposal_request))
                proposal_response = await asyncio.wait_for(self.websocket.recv(), timeout=15)
                proposal_data = json.loads(proposal_response)
                
                if self.debug_mode:
                    logger.info(f"🔍 Resposta proposta {contract_type}: {proposal_data}")
                
                if "proposal" not in proposal_data or "id" not in proposal_data["proposal"]:
                    logger.error(f"❌ Erro proposta {contract_type} (tentativa {attempt + 1}): {proposal_data}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        return None
                    
                proposal_id = proposal_data["proposal"]["id"]
                payout_info = proposal_data["proposal"].get("payout", 0)
                
                buy_request = {
                    "buy": proposal_id,
                    "price": amount,
                    "req_id": timestamp + 1
                }
                
                if self.debug_mode:
                    logger.info(f"🔍 Enviando compra {contract_type}: {buy_request}")
                
                await self.websocket.send(json.dumps(buy_request))
                buy_response = await asyncio.wait_for(self.websocket.recv(), timeout=15)
                buy_data = json.loads(buy_response)
                
                if self.debug_mode:
                    logger.info(f"🔍 Resposta compra {contract_type}: {buy_data}")
                
                if "buy" in buy_data and buy_data["buy"]:
                    contract_id = buy_data["buy"]["contract_id"]
                    buy_price = buy_data["buy"]["buy_price"]
                    transaction_id = buy_data["buy"].get("transaction_id")
                    
                    contract_info = {
                        "id": contract_id,
                        "type": contract_type,
                        "amount": amount,
                        "buy_price": buy_price,
                        "payout": payout_info,
                        "status": "open",
                        "start_time": time.time(),
                        "transaction_id": transaction_id,
                        "verification_attempts": 0
                    }
                    
                    self.active_contracts.append(contract_info)
                    
                    logger.info(f"✅ {contract_type} comprado: ${buy_price} | ID: {contract_id} | TX: {transaction_id}")
                    return contract_info
                else:
                    logger.error(f"❌ Erro compra {contract_type} (tentativa {attempt + 1}): {buy_data}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        return None
                    
            except Exception as e:
                logger.error(f"❌ Exceção {contract_type} (tentativa {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)
                else:
                    return None
        
        return None

    async def execute_dual_entry(self):
        """Executa entrada dupla com suporte a modo direcional e hedge"""
        await self.get_balance_with_retry()
        self.current_amount = self.calculate_current_amount()
        
        # Determinar direções das operações
        op1_direction, op2_direction = self.determine_operation_directions()
        
        # Log da rodada com informações direcionais
        if self.strategy_mode == "HEDGE":
            logger.info(f"🎲 RODADA {self.current_round} | Modo: HEDGE | Valor: ${self.current_amount} cada")
            logger.info(f"🎯 Operações: {op1_direction} + {op2_direction}")
        else:
            logger.info(f"🎲 RODADA {self.current_round} | Modo: DIRECIONAL | Valor: ${self.current_amount} cada")
            logger.info(f"🎯 Operações: {op1_direction} + {op2_direction}")
            if self.current_round > 1:
                logger.info(f"📈 Martingale: OP1={self.op1_martingale_direction}, OP2={self.op2_martingale_direction}")
        
        self.log_amount_calculation()
        self.balance_before_round = self.balance
        
        logger.info(f"💰 Saldo: ${self.balance_before_round} | Loss: ${self.total_loss} | Ciclo: {self.current_cycle}")
        
        total_needed = self.current_amount * 2
        if total_needed > self.balance:
            logger.error(f"❌ Saldo insuficiente! Necessário: ${total_needed}, Disponível: ${self.balance}")
            return False
            
        if self.current_amount < self.min_amount:
            logger.error(f"❌ Valor muito baixo! Mínimo: ${self.min_amount}, Calculado: ${self.current_amount}")
            return False
            
        if self.current_amount > self.max_amount:
            logger.error(f"❌ Valor muito alto! Máximo: ${self.max_amount}, Calculado: ${self.current_amount}")
            return False
            
        self.active_contracts = []
        self.subscribed_contracts = set()
        
        # Aguardar timing se necessário (apenas para primeiro timing ou modo imediato)
        if not self.pending_martingale or self.martingale_mode == "NEXT_SIGNAL":
            timing_success = await self.wait_for_execution_time()
            if not timing_success:
                logger.error("❌ Falha no sistema de timing!")
                return False
        else:
            logger.info("⚡ MARTINGALE IMEDIATO: Executando sem aguardar timing")
        
        try:
            logger.info("🔄 MODO SEQUENCIAL BLINDADO")
            
            execution_start = time.time()
            op1_result = await self.get_proposal_and_buy(op1_direction, self.current_amount)
            await asyncio.sleep(self.delay_between_ops)
            op2_result = await self.get_proposal_and_buy(op2_direction, self.current_amount)
            execution_time = time.time() - execution_start
            
            if self.timing_mode == "SCHEDULED" and not self.pending_martingale:
                current_second = int(time.time()) % 60
                logger.info(f"⏰ Execução completada em {execution_time:.2f}s no segundo {current_second}")
            
            success_count = 0
            if op1_result and not isinstance(op1_result, Exception):
                success_count += 1
            if op2_result and not isinstance(op2_result, Exception):
                success_count += 1
                
            if success_count >= 1:
                logger.info(f"🚀 Entrada dupla: {success_count}/2 operações executadas")
                
                await asyncio.sleep(2)
                await self.get_balance_with_retry()
                expected_balance = self.balance_before_round - (success_count * self.current_amount)
                balance_diff = abs(self.balance - expected_balance)
                
                if balance_diff <= 0.01:
                    logger.info(f"✅ Verificação de saldo OK: ${self.balance} (esperado: ~${expected_balance})")
                else:
                    logger.warning(f"⚠️ Divergência no saldo: ${self.balance} vs esperado ${expected_balance}")
                
                # Reset flag de martingale pendente
                self.pending_martingale = False
                
                return True
            else:
                logger.error("❌ Falha total na entrada dupla!")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro na entrada dupla: {e}")
            return False

    async def robust_contract_verification(self, contract_id, max_attempts=None):
        """Verificação robusta de contrato com múltiplas tentativas e reconexão"""
        if max_attempts is None:
            max_attempts = self.max_verification_attempts
            
        if contract_id in self.processed_contracts:
            if self.debug_mode:
                logger.info(f"🔍 Contrato {contract_id} já processado: {self.processed_contracts[contract_id]}")
            return self.processed_contracts[contract_id]
        
        for attempt in range(max_attempts):
            try:
                if not await self.ensure_connection():
                    logger.warning(f"⚠️ Falha de conexão na verificação (tentativa {attempt + 1})")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(2)
                        continue
                    else:
                        break
                
                contract_request = {
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                    "req_id": int(time.time() * 1000) + attempt
                }
                
                await self.websocket.send(json.dumps(contract_request))
                
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=self.verification_timeout
                    )
                    data = json.loads(response)
                    
                    if "proposal_open_contract" in data:
                        poc_data = data["proposal_open_contract"]
                        if poc_data.get("contract_id") == contract_id:
                            status = poc_data.get("status")
                            if status in ["sold", "won", "lost"]:
                                self.processed_contracts[contract_id] = poc_data
                                
                            if self.debug_mode:
                                logger.info(f"🔍 Tentativa {attempt + 1}: Contrato {contract_id} status: {status}")
                            
                            return poc_data
                    
                    if "error" in data and data.get("req_id") == contract_request["req_id"]:
                        logger.warning(f"⚠️ Erro na verificação (tentativa {attempt + 1}): {data['error']}")
                        
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout na verificação (tentativa {attempt + 1}/{max_attempts})")
                    
            except Exception as e:
                logger.error(f"❌ Erro na tentativa {attempt + 1}: {e}")
                
            if attempt < max_attempts - 1:
                await asyncio.sleep(1 + attempt * 0.5)
                
        logger.error(f"❌ Falha após {max_attempts} tentativas para contrato {contract_id}")
        return None

    async def wait_for_results_bulletproof(self):
        """Sistema de verificação ultra-robusto com fallback inteligente"""
        if not self.active_contracts:
            return False
            
        logger.info("⏳ Aguardando resultados com sistema blindado...")
        
        if self.duration_unit == "t":
            wait_time = max(25, self.duration * 5)
            logger.info(f"⚡ Aguardando {wait_time}s (modo ticks blindado)")
        else:
            wait_time = self.duration + 20
            logger.info(f"⏰ Aguardando {wait_time}s (modo segundos blindado)")
            
        start_time = time.time()
        check_interval = 2
        last_check = start_time
        
        phase = 1
        max_phases = 3
        connection_failures = 0
        max_connection_failures = 3
        
        while time.time() - start_time < wait_time and phase <= max_phases:
            current_time = time.time()
            
            pending_contracts = [c for c in self.active_contracts if c.get("status") != "finished"]
            
            if not pending_contracts:
                logger.info("✅ Todos os contratos finalizados!")
                break
                
            if current_time - last_check >= check_interval:
                logger.info(f"🔍 FASE {phase}: Verificando {len(pending_contracts)} contratos...")
                
                if not await self.ensure_connection():
                    connection_failures += 1
                    logger.error(f"❌ Falha de conexão {connection_failures}/{max_connection_failures}")
                    
                    if connection_failures >= max_connection_failures:
                        logger.warning("🧠 Muitas falhas de conexão, ativando sistema de inferência...")
                        inference_result = await self.smart_balance_inference()
                        if inference_result in ["inferred"]:
                            return True
                        else:
                            logger.error("❌ Falha crítica: Sistema de inferência também falhou!")
                            break
                    
                    await asyncio.sleep(5)
                    continue
                
                connection_failures = 0
                
                for contract in pending_contracts:
                    contract_id = contract["id"]
                    contract["verification_attempts"] += 1
                    
                    attempts_for_phase = max(2, self.max_verification_attempts // max_phases)
                    contract_data = await self.robust_contract_verification(
                        contract_id, 
                        attempts_for_phase
                    )
                    
                    if contract_data:
                        status = contract_data.get("status")
                        
                        if status in ["sold", "won", "lost"]:
                            if status == "sold":
                                profit = float(contract_data.get("profit", 0))
                            elif status == "won":
                                payout = float(contract_data.get("payout", 0))
                                buy_price = float(contract_data.get("buy_price", 0))
                                profit = payout - buy_price
                            else:
                                buy_price = float(contract_data.get("buy_price", 0))
                                profit = -buy_price
                            
                            contract["status"] = "finished"
                            contract["profit"] = profit
                            contract["final_status"] = status
                            contract["exit_tick"] = contract_data.get("exit_tick")
                            contract["sell_time"] = contract_data.get("sell_time")
                            
                            result_emoji = "✅" if profit > 0 else "❌"
                            result_text = "GANHOU" if profit > 0 else "PERDEU"
                            
                            logger.info(f"{result_emoji} {contract['type']} {result_text}: ${profit:+.2f} (Status: {status})")
                            
                            if self.debug_mode:
                                logger.info(f"🔍 Detalhes: Exit={contract_data.get('exit_tick')}, Sell_time={contract_data.get('sell_time')}")
                        
                        elif status == "open":
                            elapsed = time.time() - contract["start_time"]
                            if self.debug_mode:
                                logger.info(f"🔍 {contract['type']} ainda aberto após {elapsed:.1f}s (tentativa {contract['verification_attempts']})")
                
                last_check = current_time
                
            if current_time - start_time > (wait_time * phase / max_phases):
                phase += 1
                if phase <= max_phases:
                    logger.info(f"🔄 Avançando para FASE {phase}")
                
            await asyncio.sleep(0.5)
        
        unfinished_contracts = [c for c in self.active_contracts if c.get("status") != "finished"]
        
        if unfinished_contracts:
            logger.warning(f"⚠️ {len(unfinished_contracts)} contratos ainda pendentes após {wait_time}s")
            logger.info("🧠 Ativando sistema de inferência por timeout...")
            
            inference_result = await self.smart_balance_inference()
            if inference_result == "inferred":
                logger.info("✅ Resultados inferidos com sucesso!")
            else:
                logger.error("❌ Falha na inferência, assumindo perdas")
                for contract in unfinished_contracts:
                    contract["status"] = "finished"
                    contract["profit"] = -contract["amount"]
                    contract["final_status"] = "lost (timeout)"
                    logger.warning(f"❌ {contract['type']} assumido como perda por timeout")
        
        return True

    async def smart_balance_inference(self):
        """Sistema inteligente de inferência baseado em saldo quando WebSocket falha"""
        logger.info("🧠 Ativando sistema de inferência inteligente...")
        
        balance_obtained = False
        for attempt in range(5):
            try:
                if await self.get_balance_with_retry(3):
                    balance_obtained = True
                    break
            except:
                logger.warning(f"⚠️ Tentativa {attempt + 1}/5 de obter saldo para inferência falhou")
                await asyncio.sleep(2)
        
        if not balance_obtained:
            logger.error("❌ Não foi possível obter saldo para inferência!")
            return "failed"
        
        balance_after = self.balance
        balance_change = balance_after - self.balance_before_round
        
        total_investment = sum(c["amount"] for c in self.active_contracts)
        
        if self.duration_unit == "t" and self.duration <= 5:
            estimated_payout_multiplier = 2.4
        elif self.duration_unit == "t":
            estimated_payout_multiplier = 2.0
        else:
            estimated_payout_multiplier = 1.9
        
        expected_profit_per_win = (total_investment / 2) * (estimated_payout_multiplier - 1)
        
        logger.info(f"🧠 Análise de inferência:")
        logger.info(f"   Saldo antes: ${self.balance_before_round}")
        logger.info(f"   Saldo depois: ${balance_after}")
        logger.info(f"   Mudança real: ${balance_change:+.2f}")
        logger.info(f"   Investimento total: ${total_investment}")
        logger.info(f"   Lucro esperado por vitória: ~${expected_profit_per_win:.2f}")
        
        tolerance = 0.1
        
        if balance_change >= (expected_profit_per_win * 2) - tolerance:
            logger.info("🧠 INFERÊNCIA: Ambas as operações GANHARAM!")
            return self._apply_inference_results(2, expected_profit_per_win)
            
        elif balance_change >= expected_profit_per_win - tolerance:
            logger.info("🧠 INFERÊNCIA: Uma operação GANHOU!")
            return self._apply_inference_results(1, expected_profit_per_win)
            
        elif abs(balance_change + total_investment) <= tolerance:
            logger.info("🧠 INFERÊNCIA: Ambas as operações PERDERAM!")
            return self._apply_inference_results(0, 0)
            
        else:
            logger.warning(f"🧠 INFERÊNCIA INCERTA: Mudança de ${balance_change:+.2f} não corresponde a padrões conhecidos")
            
            if balance_change > -(total_investment * 0.5):
                logger.info("🧠 Assumindo vitória parcial baseado em saldo positivo")
                return self._apply_inference_results(1, abs(balance_change + total_investment/2))
            else:
                logger.info("🧠 Assumindo derrota baseado em saldo negativo")
                return self._apply_inference_results(0, 0)

    def _apply_inference_results(self, wins, profit_per_win):
        """Aplica resultados inferidos aos contratos"""
        logger.info(f"🧠 Aplicando inferência: {wins} vitórias com ${profit_per_win:.2f} cada")
        
        for i, contract in enumerate(self.active_contracts):
            if i < wins:
                contract["status"] = "finished"
                contract["profit"] = profit_per_win
                contract["final_status"] = "won (inferido)"
                logger.info(f"✅ {contract['type']} GANHOU (inferido): ${profit_per_win:+.2f}")
            else:
                contract["status"] = "finished"
                contract["profit"] = -contract["amount"]
                contract["final_status"] = "lost (inferido)"
                logger.info(f"❌ {contract['type']} PERDEU (inferido): ${-contract['amount']:.2f}")
        
        return "inferred"

    async def final_reconciliation(self):
        """Reconciliação final com verificação de saldo"""
        logger.info("🔍 Iniciando reconciliação final...")
        
        await self.get_balance()
        balance_after = self.balance
        
        balance_change = balance_after - self.balance_before_round
        
        unfinished_contracts = [c for c in self.active_contracts if c.get("status") != "finished"]
        
        if unfinished_contracts:
            logger.warning(f"⚠️ {len(unfinished_contracts)} contratos não finalizados. Fazendo verificação final...")
            
            for contract in unfinished_contracts:
                contract_data = await self.robust_contract_verification(
                    contract["id"], 
                    max_attempts=5
                )
                
                if contract_data and contract_data.get("status") in ["sold", "won", "lost"]:
                    status = contract_data.get("status")
                    if status == "sold":
                        profit = float(contract_data.get("profit", 0))
                    elif status == "won":
                        payout = float(contract_data.get("payout", 0))
                        buy_price = float(contract_data.get("buy_price", 0))
                        profit = payout - buy_price
                    else:
                        buy_price = float(contract_data.get("buy_price", 0))
                        profit = -buy_price
                    
                    contract["status"] = "finished"
                    contract["profit"] = profit
                    
                    result_emoji = "✅" if profit > 0 else "❌"
                    result_text = "GANHOU" if profit > 0 else "PERDEU"
                    logger.info(f"{result_emoji} {contract['type']} {result_text} (reconciliação): ${profit:+.2f}")
                
                else:
                    logger.warning(f"⚠️ Não foi possível verificar {contract['type']} (ID: {contract['id']})")
                    
                    if balance_change > 0:
                        potential_profit = balance_change
                        contract["status"] = "finished"
                        contract["profit"] = potential_profit
                        logger.warning(f"🔄 {contract['type']} inferido como GANHO baseado no saldo: ${potential_profit:+.2f}")
                    else:
                        contract["status"] = "finished"
                        contract["profit"] = -contract["amount"]
                        logger.warning(f"❌ {contract['type']} assumido como perda: ${-contract['amount']:.2f}")
        
        total_expected_change = sum(c.get("profit", -c["amount"]) for c in self.active_contracts)
        
        logger.info(f"💰 Reconciliação:")
        logger.info(f"   Saldo antes: ${self.balance_before_round}")
        logger.info(f"   Saldo depois: ${balance_after}")
        logger.info(f"   Mudança real: ${balance_change:+.2f}")
        logger.info(f"   Mudança esperada: ${total_expected_change:+.2f}")
        logger.info(f"   Diferença: ${abs(balance_change - total_expected_change):.2f}")
        
        if abs(balance_change - total_expected_change) > 0.05:
            logger.warning("⚠️ Divergência detectada na reconciliação!")
        else:
            logger.info("✅ Reconciliação bem-sucedida!")

    def analyze_round_result(self):
        """
        Analisa resultado da rodada com lógica direcional e martingale diferenciado
        """
        wins = 0
        total_profit_round = 0
        
        unprocessed = [c for c in self.active_contracts if c.get("status") != "finished"]
        if unprocessed:
            logger.warning(f"⚠️ {len(unprocessed)} contratos não processados na análise!")
        
        for contract in self.active_contracts:
            if contract.get("status") == "finished":
                profit = contract.get("profit", 0)
                total_profit_round += profit
                if profit > 0:
                    wins += 1
        
        # Contabilizar assertividade da estratégia
        self.total_strategies += 1
        
        if wins >= 1:
            self.successful_strategies += 1
            strategy_result = "SUCESSO"
            strategy_emoji = "🎉"
        else:
            self.failed_strategies += 1
            strategy_result = "FRACASSO"
            strategy_emoji = "💥"
        
        # Calcular assertividade atual
        if self.total_strategies > 0:
            assertivity = (self.successful_strategies / self.total_strategies) * 100
        else:
            assertivity = 0
        
        # Atualizar totais
        if total_profit_round > 0:
            self.total_profit += total_profit_round
        else:
            self.total_loss += abs(total_profit_round)
            
        # Calcular resultado real da operação baseado no saldo
        real_round_result = self.balance - self.balance_before_round
        
        # Log detalhado do resultado
        logger.info(f"📊 RESULTADO R{self.current_round}: {wins}/2 vitórias | Contratos: ${total_profit_round:+.2f} | Saldo: ${real_round_result:+.2f}")
        logger.info(f"{strategy_emoji} ESTRATÉGIA: {strategy_result} | Assertividade: {assertivity:.1f}% ({self.successful_strategies}/{self.total_strategies})")
        logger.info(f"💰 Total acumulado: Lucro ${self.total_profit:.2f} | Perda ${self.total_loss:.2f}")
        
        # Log individual dos contratos
        for i, contract in enumerate(self.active_contracts, 1):
            profit = contract.get("profit", 0)
            status = contract.get("final_status", "unknown")
            logger.info(f"   {contract['type']}: ${profit:+.2f} (Status: {status})")
        
        # NOVA LÓGICA DE DECISÃO COM MARTINGALE DIFERENCIADO
        if wins >= 1:
            logger.info("🎉 VITÓRIA! Uma ou ambas ganharam - RESETANDO")
            self.total_sequences += 1
            
            # Reset baseado no modo
            if self.amount_type == "FIXED":
                self.current_amount = self.initial_amount
                logger.info(f"💰 Resetando para valor fixo: ${self.initial_amount}")
            else:
                self.current_percentage = self.initial_percentage
                logger.info(f"📊 Resetando para percentual inicial: {self.initial_percentage}%")
                
            self.current_round = 1
            
            # Reset direções para modo direcional
            if self.strategy_mode == "DIRECTIONAL":
                self.current_op1_direction = None
                self.current_op2_direction = None
                
            return "victory"
        else:
            logger.info("❌ DERROTA - Aplicando MARTINGALE")
            
            # Verificar se deve executar martingale imediatamente
            if self.martingale_mode == "IMMEDIATE":
                logger.info("⚡ MARTINGALE IMEDIATO ativado!")
                self.pending_martingale = True
            else:
                logger.info("⏰ MARTINGALE NO PRÓXIMO SINAL")
                self.pending_martingale = False
            
            self.current_round += 1
            
            # Verificar limite de martingale
            if self.current_round > self.max_martingale_rounds:
                logger.warning(f"🎯 LIMITE DE MARTINGALE ATINGIDO ({self.max_martingale_rounds} rounds)")
                logger.info("🔄 RESETANDO para valor inicial e continuando...")
                
                self.martingale_resets += 1
                self.total_sequences += 1
                
                # Reset baseado no modo
                if self.amount_type == "FIXED":
                    self.current_amount = self.initial_amount
                    logger.info(f"💰 Resetando para valor fixo: ${self.initial_amount}")
                else:
                    self.current_percentage = self.initial_percentage
                    logger.info(f"📊 Resetando para percentual inicial: {self.initial_percentage}%")
                
                self.current_round = 1
                
                # Reset direções para modo direcional
                if self.strategy_mode == "DIRECTIONAL":
                    self.current_op1_direction = None
                    self.current_op2_direction = None
                
                # Verificar stop loss
                if self.total_loss >= self.stop_loss:
                    logger.error(f"🛑 STOP LOSS atingido após reset: ${self.stop_loss}")
                    return "stop_loss"
                    
                # Verificar stop win
                if self.stop_win > 0 and self.total_profit >= self.stop_win:
                    logger.info(f"🎉 STOP WIN atingido: ${self.stop_win}")
                    return "stop_win"
                
                return "martingale_reset"
            else:
                # Continuar martingale normalmente
                
                # Log das próximas direções (para modo direcional)
                if self.strategy_mode == "DIRECTIONAL":
                    next_op1, next_op2 = self.determine_operation_directions()
                    logger.info(f"🎯 Próximas direções R{self.current_round}: OP1={next_op1}, OP2={next_op2}")
                
                # Log do próximo valor
                if self.amount_type == "FIXED":
                    next_amount = self.initial_amount * (self.multiplier ** (self.current_round - 1))
                    next_amount = max(self.min_amount, min(self.max_amount, next_amount))
                    logger.info(f"📈 Próxima R{self.current_round}: ~${next_amount:.2f} cada")
                else:
                    next_percentage = self.current_percentage * (self.multiplier ** (self.current_round - 1))
                    logger.info(f"📈 Próxima R{self.current_round}: ~{next_percentage:.2f}% da banca")
                
                # Verificar limites
                if self.max_rounds < 999 and self.current_round > self.max_rounds:
                    logger.error(f"🛑 LIMITE TOTAL de rounds por ciclo ({self.max_rounds})")
                    return "max_rounds"
                    
                if self.total_loss >= self.stop_loss:
                    logger.error(f"🛑 STOP LOSS atingido: ${self.stop_loss}")
                    return "stop_loss"
                    
                if self.stop_win > 0 and self.total_profit >= self.stop_win:
                    logger.info(f"🎉 STOP WIN atingido: ${self.stop_win}")
                    return "stop_win"
                    
                return "continue"

    async def run_cycle(self):
        """Executa um ciclo completo com suporte a martingale imediato e direcional"""
        logger.info(f"🔄 INICIANDO CICLO {self.current_cycle}")
        logger.info("=" * 60)
        
        cycle_start = time.time()
        decision = "unknown"
        
        try:
            while True:
                if not self.is_connected:
                    logger.error("❌ Conexão perdida!")
                    await self.connect()
                
                # Executar rodada
                success = await self.execute_dual_entry()
                if not success:
                    decision = "execution_failed"
                    break
                    
                # Aguardar resultados
                await self.wait_for_results_bulletproof()
                
                # Atualizar saldo
                await self.get_balance_with_retry()
                
                # Analisar resultado
                decision = self.analyze_round_result()
                
                # Tratar diferentes tipos de decisão
                if decision == "victory":
                    break
                elif decision == "martingale_reset":
                    logger.info("🔄 Continuando após reset de martingale...")
                    
                    # Se martingale imediato está ativo e não foi resetado, continuar imediatamente
                    if self.martingale_mode == "IMMEDIATE" and not self.pending_martingale:
                        # Foi resetado, então aguardar próximo timing normal
                        if self.effective_rounds_interval > 0:
                            logger.info(f"⏸️ Intervalo: {self.effective_rounds_interval}s")
                            await asyncio.sleep(self.effective_rounds_interval)
                    continue
                elif decision in ["stop_loss", "stop_win"]:
                    break
                elif decision == "max_rounds":
                    break
                elif decision == "continue":
                    # Verificar se deve aguardar ou executar imediatamente
                    if self.pending_martingale and self.martingale_mode == "IMMEDIATE":
                        logger.info("⚡ Executando MARTINGALE IMEDIATO...")
                        # Pequena pausa para evitar sobrecarga
                        await asyncio.sleep(1)
                    else:
                        # Aguardar intervalo normal ou próximo timing
                        if self.effective_rounds_interval > 0:
                            logger.info(f"⏸️ Intervalo: {self.effective_rounds_interval}s")
                            await asyncio.sleep(self.effective_rounds_interval)
                    continue
                else:
                    break
                    
        except Exception as e:
            logger.error(f"❌ Erro no ciclo: {e}")
            decision = "error"
            
        cycle_time = (time.time() - cycle_start) / 60
        logger.info(f"⏱️ Ciclo {self.current_cycle} durou: {cycle_time:.1f} min")
        
        return decision

    async def run(self):
        """Executa o robô completo com todas as funcionalidades direcionais"""
        try:
            logger.info("🚀 INICIANDO ROBÔ MARTINGALE DIRECIONAL - VERSÃO SUPER BLINDADA")
            logger.info("   🎯 Estratégias HEDGE/DIRECIONAL + ⚡ Martingale Imediato + 📊 Assertividade")
            logger.info("=" * 60)
            
            await self.connect()
            
            if not self.is_connected:
                return
                
            # Verificar saldo mínimo
            if self.amount_type == "FIXED":
                min_balance = self.initial_amount * 2 * 3
                logger.info(f"💰 Verificação saldo (FIXO): Mínimo recomendado ${min_balance}")
            else:
                min_balance = self.min_amount * 2
                logger.info(f"💰 Verificação saldo (PERCENTUAL): Mínimo absoluto ${min_balance}")
                
            if self.balance < min_balance:
                logger.error(f"❌ Saldo insuficiente! Atual: ${self.balance}, Mínimo: ${min_balance}")
                return
                
            # Loop principal
            while self.current_cycle <= self.max_cycles:
                result = await self.run_cycle()
                
                # Calcular performance
                await self.get_balance_with_retry()
                current_balance = self.balance
                balance_change = current_balance - self.initial_balance
                balance_growth = (balance_change / self.initial_balance) * 100
                
                # Assertividade
                if self.total_strategies > 0:
                    current_assertivity = (self.successful_strategies / self.total_strategies) * 100
                else:
                    current_assertivity = 0
                
                logger.info("=" * 60)
                logger.info(f"📊 RESUMO CICLO {self.current_cycle}:")
                logger.info(f"   Resultado: {result}")
                logger.info(f"   Modo: {self.strategy_mode}")
                if self.strategy_mode == "DIRECTIONAL":
                    logger.info(f"   Martingale: {self.martingale_mode}")
                logger.info(f"   Lucro: ${self.total_profit:.2f}")
                logger.info(f"   Perda: ${self.total_loss:.2f}")
                logger.info(f"   Líquido: ${self.total_profit - self.total_loss:.2f}")
                logger.info(f"   Saldo inicial: ${self.initial_balance:.2f}")
                logger.info(f"   Saldo atual: ${current_balance:.2f}")
                logger.info(f"   Mudança real: ${balance_change:+.2f}")
                logger.info(f"   Crescimento da banca: {balance_growth:+.2f}%")
                logger.info(f"🎯 ASSERTIVIDADE DA ESTRATÉGIA: {current_assertivity:.1f}% ({self.successful_strategies}/{self.total_strategies})")
                logger.info(f"🎯 Resets de martingale: {self.martingale_resets}")
                logger.info(f"📈 Sequências totais: {self.total_sequences}")
                logger.info(f"   Contratos processados: {len(self.processed_contracts)}")
                
                if self.amount_type == "PERCENTAGE":
                    logger.info(f"📊 JUROS COMPOSTOS: A banca cresceu {balance_growth:+.2f}%!")
                
                # Verificar condições de parada
                if result in ["stop_loss", "stop_win"]:
                    logger.info(f"🛑 Parando por: {result}")
                    break
                    
                if not self.auto_restart:
                    logger.info("🛑 Auto-restart desabilitado")
                    break
                    
                if result in ["max_rounds", "execution_failed", "error"]:
                    logger.info(f"🛑 Parando por: {result}")
                    break
                    
                # Preparar próximo ciclo
                self.current_cycle += 1
                
                # Reset
                if self.amount_type == "FIXED":
                    self.current_amount = self.initial_amount
                else:
                    self.current_percentage = self.initial_percentage
                    
                self.current_round = 1
                
                # Reset direções para modo direcional
                if self.strategy_mode == "DIRECTIONAL":
                    self.current_op1_direction = None
                    self.current_op2_direction = None
                
                # Reset flags
                self.pending_martingale = False
                
                if self.current_cycle <= self.max_cycles and self.effective_cycles_interval > 0:
                    logger.info(f"🔄 Próximo ciclo em {self.effective_cycles_interval}s")
                    await asyncio.sleep(self.effective_cycles_interval)
                elif self.timing_mode == "SCHEDULED" and self.current_cycle <= self.max_cycles:
                    logger.info("🎯 Próximo ciclo controlado pelo timing sincronizado")
                    await asyncio.sleep(1)
                    
            logger.info("🏁 ROBÔ FINALIZADO")
            
            # Resultado final
            final_balance_change = self.balance - self.initial_balance
            logger.info(f"📊 RESULTADO FINAL: ${final_balance_change:+.2f}")
            
            # Estatísticas finais
            logger.info(f"🎯 ESTATÍSTICAS FINAIS:")
            if self.total_strategies > 0:
                final_assertivity = (self.successful_strategies / self.total_strategies) * 100
                logger.info(f"   📊 ASSERTIVIDADE FINAL: {final_assertivity:.1f}%")
                logger.info(f"   ✅ Estratégias bem-sucedidas: {self.successful_strategies}")
                logger.info(f"   ❌ Estratégias fracassadas: {self.failed_strategies}")
                logger.info(f"   📈 Total de estratégias: {self.total_strategies}")
            else:
                logger.info(f"   📊 Nenhuma estratégia foi executada")
                
            logger.info(f"   🎯 Total de resets por limite de martingale: {self.martingale_resets}")
            logger.info(f"   📈 Total de sequências completas: {self.total_sequences}")
            
            if self.total_sequences > 0:
                success_rate = ((self.total_sequences - self.martingale_resets) / self.total_sequences) * 100
                logger.info(f"   🏆 Taxa de sucesso das sequências: {success_rate:.1f}%")
            
            # Resumo executivo
            logger.info("=" * 60)
            logger.info("🏆 RESUMO EXECUTIVO:")
            logger.info(f"   💰 Resultado financeiro: ${final_balance_change:+.2f}")
            if self.total_strategies > 0:
                logger.info(f"   🎯 Assertividade da estratégia: {final_assertivity:.1f}%")
            logger.info(f"   📊 Performance: {((final_balance_change/self.initial_balance)*100):+.2f}%")
            logger.info(f"   🎯 Modo utilizado: {self.strategy_mode}")
            if self.strategy_mode == "DIRECTIONAL":
                logger.info(f"   ⚡ Martingale: {self.martingale_mode}")
            
        except KeyboardInterrupt:
            logger.info("⚠️ Robô interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro crítico: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
                logger.info("🔌 Conexão fechada")

def main():
    """Função principal"""
    try:
        robot = DerivMartingaleBotDirectional()
        asyncio.run(robot.run())
    except ValueError as e:
        print(f"❌ Erro de configuração: {e}")
        print("💡 Verifique o arquivo .env!")
    except KeyboardInterrupt:
        print("\n👋 Robô interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro fatal: {e}")

if __name__ == "__main__":
    main()
# Global state
ex: Optional[AlpacaExchange] = None
strategy: Optional[TradingStrategy] = None
risk_manager: Optional[RiskManager] = None
latest_scan_results: Dict[str, dict] = {}
scan_cycle_count: int = 0
# Per-symbol cooldown: timestamp (time.time()) before which a new entry on
# that symbol is blocked, set right after closing a position. See
# settings.COOLDOWN_SECONDS_BUY for why this exists.
cooldowns: Dict[str, float] = {}

# ... rest of the file content remains unchanged ...

# Get current positions
positions = await ex.get_positions()
position_dict = {p["symbol"].replace("/", ""): p for p in positions}
current_position = position_dict.get(symbol.replace("/", ""))

# Cooldown check: block a fresh entry right after this symbol closed a
# position, but never block evaluation of an EXISTING position's exit
# logic (a position we're already holding must always be free to close).
if not current_position:
    now_ts = time.time()
    cooldown_until = cooldowns.get(symbol, 0)
    if now_ts < cooldown_until:
        remaining = int(cooldown_until - now_ts)
        logger.debug(f"[{symbol}] On entry cooldown, {remaining}s remaining -- skipping")
        return

# Trailing Stop Check

# ... rest of bot.py content remains unchanged ...

# Trailing Stop Check
        # Get current positions (already done above)
        # Cooldown check (already done above)
        # Trailing Stop Check
        if not current_position:
            # This path already handled cooldown
            pass
        else:
            # Existing position logic (no cooldown check needed here)
            pass

        # Trailing Stop Execution
        # ... existing trailing stop logic remains ...

# Position closed path
            logger.info(f"[TRADE] Position closed: {symbol} (was {qty}) - reason: {signal.get('reason', 'unknown')}")
            logger.debug(f"[TRADE] Close order result: {order_result}")
            await send_telegram_alert(f"📉 <b>Position Closed</b>\nSymbol: {symbol}\nReason: {signal.get('reason', 'unknown')}")
            # Update cooldown for this symbol
            cooldowns[symbol] = time.time() + settings.COOLDOWN_SECONDS_BUY
            await _record_committee_outcome(symbol, current_price, exit_reason=signal.get('reason', 'unknown'))
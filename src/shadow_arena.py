import logging
import uuid
import datetime
import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from sqlalchemy import select
from src.db import get_engine, get_db_session, ShadowTrade, Base
from src.config import settings

logger = logging.getLogger("shadow_arena")

class GQA_TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_q_heads=8, num_kv_heads=2, dropout=0.1):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        import torch.nn.functional as F
        self.F = F

    def forward(self, x):
        residual = x
        norm_x = self.norm1(x)
        batch, seq, _ = norm_x.shape
        q = self.q_proj(norm_x).view(batch, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        attn = self.F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.num_q_heads * self.head_dim)
        x = residual + self.dropout(self.out_proj(attn))
        residual = x
        norm_x = self.norm2(x)
        x = residual + self.ffn(norm_x)
        return x

class GrokGQA_Transformer(nn.Module):
    def __init__(self, input_dim=11, seq_len=32, embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GQA_TransformerBlock(embed_dim, num_q_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoder 
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_head(x[:, -1, :])
        return x

_candidates_cache = {}

def get_candidates():
    if _candidates_cache:
        return _candidates_cache
    
    candidates_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'candidates')
    if not os.path.exists(candidates_dir):
        return {}
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import joblib
    import numpy as np
    scaler_path = os.path.join(candidates_dir, "feature_scaler.pkl")
    if not os.path.exists(scaler_path):
        return {}
    scaler = joblib.load(scaler_path)
    
    if hasattr(scaler, "feature_names_in_"):
        cols = list(scaler.feature_names_in_)
    else:
        from src.feature_engineering import get_active_features
        cols = get_active_features()
    
    for f in os.listdir(candidates_dir):
        if f.endswith("_config.json"):
            name = f.replace("_config.json", "")
            pth_file = os.path.join(candidates_dir, f"{name}.pth")
            if os.path.exists(pth_file):
                with open(os.path.join(candidates_dir, f), "r") as json_f:
                    config = json.load(json_f)
                
                model = GrokGQA_Transformer(
                    input_dim=len(cols),
                    seq_len=32,
                    embed_dim=config.get("embed", 128),
                    num_layers=config.get("layers", 4),
                    dropout=config.get("dropout", 0.1)
                ).to(device)
                
                model.load_state_dict(torch.load(pth_file, map_location=device, weights_only=True))
                model.eval()
                
                _candidates_cache[name] = {
                    "model": model,
                    "config": config,
                    "device": device,
                    "scaler": scaler,
                    "cols": cols
                }
    
    return _candidates_cache

def evaluate_candidates(symbol: str, current_price: float, df_feat: Any) -> None:
    """Evaluates all candidate models using the raw input DataFrame."""
    import numpy as np
    candidates = get_candidates()
    if not candidates:
        return
        
    for name, c_info in candidates.items():
        try:
            model = c_info["model"]
            device = c_info["device"]
            config = c_info["config"]
            scaler = c_info["scaler"]
            cols = c_info["cols"]
            
            data = df_feat[cols].tail(32).values.astype(np.float32)
            if len(data) < 32:
                continue
                
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = np.clip(data, -1e6, 1e6)
            data_scaled = scaler.transform(data).astype(np.float32)
            data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            x = torch.tensor(data_scaled).unsqueeze(0).to(device)
            with torch.no_grad():
                raw_logit = model(x).squeeze(1).item()
                prob = float(torch.sigmoid(torch.tensor(raw_logit)).item())
                
            threshold = config.get("threshold", 0.58)
            stop_loss = config.get("stop_loss", 0.02)
            profit_target = config.get("profit_target", 0.04)
            
            # Check stops first
            check_shadow_stops(name, symbol, current_price, stop_loss, profit_target)
            
            # Generate new signal
            action = "hold"
            if prob > threshold:
                action = "buy"
            elif prob < (1.0 - threshold):
                action = "sell" # Not doing shorting yet, but could be "close"
                
            if action == "buy":
                process_shadow_signal(name, symbol, current_price, "buy")
            elif action == "sell":
                process_shadow_signal(name, symbol, current_price, "close")
                
        except Exception as e:
            logger.error(f"Error evaluating candidate {name}: {e}")

def process_shadow_signal(candidate_name: str, symbol: str, current_price: float, action: str, qty: float = 1.0) -> None:
    """Processes a signal for a candidate model in the Shadow Arena."""
    try:
        Base.metadata.create_all(get_engine())
        with get_db_session() as session:
            # Get open trade for this candidate and symbol
            stmt = select(ShadowTrade).where(
                ShadowTrade.candidate_name == candidate_name,
                ShadowTrade.symbol == symbol,
                ShadowTrade.status == "open"
            ).order_by(ShadowTrade.created_at.desc())
            
            open_trade = session.execute(stmt).scalars().first()

            if action in ["buy", "sell"] and not open_trade:
                # Open a new shadow position
                trade_id = uuid.uuid4().hex
                new_trade = ShadowTrade(
                    trade_id=trade_id,
                    candidate_name=candidate_name,
                    symbol=symbol,
                    side=action,
                    qty=qty,
                    entry_price=current_price,
                    status="open"
                )
                session.add(new_trade)
                session.commit()
                logger.info(f"👻 SHADOW ({candidate_name}): Opened {action.upper()} {symbol} @ ${current_price:.2f}")

            elif action == "close" and open_trade:
                # Close the shadow position
                open_trade.status = "closed"
                open_trade.exit_price = current_price
                open_trade.closed_at = datetime.datetime.now(datetime.timezone.utc)
                
                if open_trade.side == "buy":
                    open_trade.realized_pnl = (current_price - open_trade.entry_price) * open_trade.qty
                else:
                    open_trade.realized_pnl = (open_trade.entry_price - current_price) * open_trade.qty
                
                session.commit()
                logger.info(f"👻 SHADOW ({candidate_name}): Closed {symbol} @ ${current_price:.2f} (PnL: ${open_trade.realized_pnl:.2f})")

    except Exception as e:
        logger.error(f"Shadow arena error for {candidate_name} {symbol}: {e}")

def check_shadow_stops(candidate_name: str, symbol: str, current_price: float, stop_loss_pct: float, profit_target_pct: float) -> None:
    """Checks stop loss and profit targets for an open shadow position."""
    try:
        with get_db_session() as session:
            stmt = select(ShadowTrade).where(
                ShadowTrade.candidate_name == candidate_name,
                ShadowTrade.symbol == symbol,
                ShadowTrade.status == "open"
            ).order_by(ShadowTrade.created_at.desc())
            
            open_trade = session.execute(stmt).scalars().first()
            if not open_trade:
                return

            if open_trade.side == "buy":
                pnl_pct = (current_price - open_trade.entry_price) / open_trade.entry_price
            else:
                pnl_pct = (open_trade.entry_price - current_price) / open_trade.entry_price

            if pnl_pct <= -stop_loss_pct or pnl_pct >= profit_target_pct:
                # Close the position
                open_trade.status = "closed"
                open_trade.exit_price = current_price
                open_trade.closed_at = datetime.datetime.now(datetime.timezone.utc)
                open_trade.realized_pnl = (pnl_pct * open_trade.entry_price) * open_trade.qty
                session.commit()
                
                reason = "Stop Loss" if pnl_pct <= -stop_loss_pct else "Profit Target"
                logger.info(f"👻 SHADOW ({candidate_name}): {reason} hit for {symbol} @ ${current_price:.2f} (PnL: ${open_trade.realized_pnl:.2f})")

    except Exception as e:
        pass # Silently fail shadow stops to avoid log spam

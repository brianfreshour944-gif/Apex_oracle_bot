import asyncio
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db import init_db, get_engine, get_db_session, DecisionSnapshot
from src.attribution import analyze_closed_trade
import uuid
from datetime import datetime

async def main():
    print("Testing Causal Attribution Engine...")
    init_db()
    
    # Create a mock closed trade
    mock_id = str(uuid.uuid4())
    
    with get_db_session() as session:
        snap = DecisionSnapshot(
            decision_id=mock_id,
            symbol="BTC/USD",
            regime="bull_volatile",
            final_action="buy",
            confidence=0.85,
            size_multiplier=1.0,
            entry_price=60000.0,
            qty=1.0,
            votes_json='{"transformer": "buy", "llm": "buy", "momentum": "buy", "mean_reversion": "sell"}',
            causal_reasoning_json='{"rsi": 0.45, "macd": 0.8, "funding_rate": -0.2, "vol_of_vol": 0.1}',
            status="closed",
            realized_pnl=1500.0,
            return_pct=2.5,
            created_at=datetime.utcnow()
        )
        session.add(snap)
        session.commit()
        
    print(f"Mock trade {mock_id} inserted. Running attribution...")
    
    res = await analyze_closed_trade(mock_id)
    
    if res:
        print("\n\u2705 Attribution generated successfully!")
        import json
        print(json.dumps(res, indent=2))
    else:
        print("\n\u274c Attribution failed (do you have GROQ_API_KEY set?)")
        
if __name__ == "__main__":
    asyncio.run(main())

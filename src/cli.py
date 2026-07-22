"""Modern Typer CLI for Apex Oracle Bot."""

import sys
import asyncio
import httpx
import typer

from src.logging_config import get_logger
from src.config import settings

logger = get_logger(__name__)

app = typer.Typer(
    name="apex-bot",
    help="Apex Oracle Bot - Production High-Frequency Crypto Trading CLI",
    add_completion=False,
)


@app.command("run")
def run() -> None:
    """Start the live / paper trading bot."""
    from src.bot import run_bot
    typer.secho("🚀 Starting Apex Oracle Bot...", fg=typer.colors.GREEN, bold=True)
    run_bot()


@app.command("backtest")
def backtest(
    symbol: str = typer.Option("BTC/USD", help="Trading pair symbol"),
    bars: int = typer.Option(400, help="Number of historical bars to simulate"),
    equity: float = typer.Option(10000.0, help="Starting account equity in USD"),
    seed: int = typer.Option(7, help="Random seed"),
    regime: str = typer.Option("all", help="Regime to test: all, trending, mean_reverting, volatile"),
    walk_forward: bool = typer.Option(False, "--walk-forward", help="Run walk-forward optimization split"),
) -> None:
    """Run backtest simulation or walk-forward optimization."""
    from src.backtest import run_backtest, run_walk_forward_optimization, print_backtest_summary

    typer.secho(f"📊 Running Backtest Engine for {symbol} ({bars} bars)...", fg=typer.colors.CYAN, bold=True)

    if walk_forward:
        asyncio.run(run_walk_forward_optimization(symbol=symbol, total_bars=bars, seed=seed))
    else:
        regimes_to_run = ["trending", "mean_reverting", "volatile"] if regime == "all" else [regime]
        for reg in regimes_to_run:
            res = asyncio.run(run_backtest(
                symbol=symbol,
                n_bars=bars,
                start_equity=equity,
                seed=seed,
                regime=reg
            ))
            print_backtest_summary(res)


@app.command("status")
def status() -> None:
    """Check live bot status and health endpoint."""
    url = f"http://localhost:{settings.STATUS_PORT}/api/v1/health"
    typer.secho(f"🔍 Checking health status at {url}...", fg=typer.colors.YELLOW)
    try:
        resp = httpx.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            status_color = typer.colors.GREEN if data.get("status") == "healthy" else typer.colors.YELLOW
            typer.secho(f"\nBot Status: {data.get('status')}", fg=status_color, bold=True)
            typer.echo(f"Database:   {data.get('database', {}).get('url')}")
            typer.echo(f"Connected:  {data.get('database', {}).get('connected')}")
            typer.echo(f"Symbols:    {', '.join(data.get('symbols', []))}")
        else:
            typer.secho(f"⚠️ Health check returned HTTP {resp.status_code}: {resp.text}", fg=typer.colors.RED)
    except Exception as e:
        typer.secho(f"❌ Failed to query health endpoint: {e}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()

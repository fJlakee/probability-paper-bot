from __future__ import annotations

import argparse
from pathlib import Path
from .bot import PaperBot
from .config import load_config
from .backtest import WalkForwardBacktester


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance probability paper-trading bot")
    parser.add_argument("command", choices=["once", "run", "backtest"])
    parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    bot = PaperBot(load_config(path), path)
    if args.command == "once":
        bot.cycle()
    elif args.command == "backtest":
        symbols = bot._symbols()
        frames = bot._update_market_data(symbols)
        report_dir = Path(bot.cfg.storage.report_directory)
        if not report_dir.is_absolute():
            report_dir = path.parent / report_dir
        trades, summary = WalkForwardBacktester(bot.cfg, report_dir).run(frames)
        print(f"Trades: {trades}\nSummary: {summary}")
    else:
        bot.run()


if __name__ == "__main__":
    main()

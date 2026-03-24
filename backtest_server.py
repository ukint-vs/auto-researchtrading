"""
Resident backtest evaluator. Keeps data hot in memory, reloads strategy on each run.
Eliminates ~2-3s of startup/import/parquet-load overhead per autoresearch iteration.

Usage:
    uv run backtest_server.py              # Start TCP server on port 9877 (default)
    uv run backtest_server.py --port 9878  # Custom port
    uv run backtest_server.py --stdin      # Read from stdin (pipe mode)
    echo run | nc localhost 9877           # Trigger a backtest from another process
"""

import sys
import time
import json
import socket
import signal as sig
import argparse

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET

DEFAULT_PORT = 9877


def handle_run():
    """Reload strategy, run backtest, return JSON result."""
    # Reload strategy module (picks up autoresearch edits to strategy.py)
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    try:
        from strategy import Strategy
    except Exception as e:
        return json.dumps({"error": f"import: {e}", "score": -999})

    # Set timeout
    def timeout_handler(signum, frame):
        raise TimeoutError("backtest exceeded time budget")

    sig.signal(sig.SIGALRM, timeout_handler)
    sig.alarm(TIME_BUDGET + 10)

    try:
        t_start = time.time()
        strategy = Strategy()
        result = run_backtest(strategy, data)
        score = compute_score(result)
        elapsed = time.time() - t_start

        return json.dumps({
            "score": round(score, 6),
            "sharpe": round(result.sharpe, 6),
            "total_return_pct": round(result.total_return_pct, 6),
            "max_drawdown_pct": round(result.max_drawdown_pct, 6),
            "num_trades": result.num_trades,
            "win_rate_pct": round(result.win_rate_pct, 6),
            "profit_factor": round(result.profit_factor, 6),
            "annual_turnover": round(result.annual_turnover, 2),
            "backtest_seconds": round(result.backtest_seconds, 1),
            "total_seconds": round(elapsed, 1),
        })
    except TimeoutError:
        return json.dumps({"error": "timeout", "score": -999})
    except Exception as e:
        return json.dumps({"error": str(e), "score": -999})
    finally:
        sig.alarm(0)


def main():
    parser = argparse.ArgumentParser(description="Resident backtest evaluator")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (default: 9877)")
    parser.add_argument("--stdin", action="store_true", help="Read commands from stdin instead of TCP")
    args = parser.parse_args()

    if args.stdin:
        run_stdin_mode()
        return

    # Load data ONCE on startup
    t_load = time.time()
    global data
    data = load_data("val")
    load_time = time.time() - t_load

    total_bars = sum(len(df) for df in data.values())
    print(f"Loaded {total_bars} bars across {len(data)} symbols in {load_time:.1f}s", file=sys.stderr)
    print(f"Symbols: {list(data.keys())}", file=sys.stderr)

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)

    print(f"Listening on 127.0.0.1:{args.port}", file=sys.stderr)
    print(f"Client: echo run | nc localhost {args.port}", file=sys.stderr)
    sys.stderr.flush()

    # Handle SIGTERM/SIGINT for clean shutdown
    def shutdown_handler(signum, frame):
        print("\nShutting down.", file=sys.stderr)
        server.close()
        sys.exit(0)

    sig.signal(sig.SIGTERM, shutdown_handler)
    sig.signal(sig.SIGINT, shutdown_handler)

    while True:
        try:
            conn, addr = server.accept()
        except OSError:
            break

        try:
            raw = conn.recv(1024).decode().strip().lower()
            if raw == "run":
                result = handle_run()
                conn.sendall((result + "\n").encode())
            elif raw == "ping":
                conn.sendall(b'{"status": "ok"}\n')
            else:
                conn.sendall(b'{"error": "unknown command, send run or ping"}\n')
        except Exception as e:
            try:
                conn.sendall(json.dumps({"error": str(e), "score": -999}).encode() + b"\n")
            except Exception:
                pass
        finally:
            conn.close()


def run_stdin_mode():
    """Read commands from stdin (for piped input: echo run | uv run backtest_server.py --stdin)."""
    global data
    data = load_data("val")
    total_bars = sum(len(df) for df in data.values())
    print(f"Loaded {total_bars} bars across {len(data)} symbols", file=sys.stderr)
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd == "quit" or cmd == "exit":
            break
        if cmd == "run":
            result = handle_run()
            print(result)
            sys.stdout.flush()


if __name__ == "__main__":
    main()

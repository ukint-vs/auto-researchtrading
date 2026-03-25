"""
Resident backtest evaluator with mandatory dual-gate enforcement.
Keeps data hot in memory, reloads strategy on each run.
Every evaluation runs BOTH gate1 (standard) and gate2 (realistic).
If gate2/gate1 ratio < 0.50, score is forced to -888 (veto).

Usage:
    uv run backtest_server.py              # Start TCP server on port 9877
    uv run backtest_server.py --port 9878  # Custom port

Commands (via TCP):
    run                          # Reload strategy.py, dual-gate backtest, return JSON
    ping                         # Health check
    batch:{"variants":[...]}     # Parallel batch: dual-gate all variants, return sorted
"""

import os
import sys
import time
import json
import socket
import signal as sig
import argparse
import multiprocessing

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import prepare
from prepare import (
    load_data, run_backtest, compute_score, TIME_BUDGET,
    Signal, PortfolioState, BarData,
    TAKER_FEE as STANDARD_TAKER_FEE,
    SLIPPAGE_BPS as STANDARD_SLIPPAGE_BPS,
    REALISTIC_TAKER_FEE, REALISTIC_SLIPPAGE_BPS,
)

DEFAULT_PORT = 9877
GATE2_VETO_RATIO = 0.50

# Global data — loaded once in parent, passed to workers via initializer
data = None

# Per-worker data (set by _worker_data_init in each spawned worker)
_worker_data = None


def _worker_data_init(shared_data):
    """Initializer for Pool workers: store data once per worker."""
    global _worker_data
    _worker_data = shared_data


def _run_dual_gate(strategy_code, namespace):
    """Run a strategy through both gates. Returns (gate1_result, gate2_result)."""
    # Gate 1: standard constants
    prepare.TAKER_FEE = STANDARD_TAKER_FEE
    prepare.SLIPPAGE_BPS = STANDARD_SLIPPAGE_BPS
    prepare.REALISTIC_MODE = False

    strategy1 = namespace["Strategy"]()
    result1 = run_backtest(strategy1, _worker_data)
    score1 = compute_score(result1)

    # Gate 2: realistic constants
    prepare.TAKER_FEE = REALISTIC_TAKER_FEE
    prepare.SLIPPAGE_BPS = REALISTIC_SLIPPAGE_BPS
    prepare.REALISTIC_MODE = True

    strategy2 = namespace["Strategy"]()
    result2 = run_backtest(strategy2, _worker_data)
    score2 = compute_score(result2)

    # Restore standard
    prepare.TAKER_FEE = STANDARD_TAKER_FEE
    prepare.SLIPPAGE_BPS = STANDARD_SLIPPAGE_BPS
    prepare.REALISTIC_MODE = False

    return result1, score1, result2, score2


def _run_variant(args):
    """Worker function: exec strategy code, run dual-gate backtest, return result dict."""
    variant_id, strategy_code = args
    namespace = {
        "np": np,
        "sliding_window_view": sliding_window_view,
        "Signal": Signal,
        "PortfolioState": PortfolioState,
        "BarData": BarData,
    }
    try:
        exec(strategy_code, namespace)
        if "Strategy" not in namespace:
            return {"variant_id": variant_id, "score": -999,
                    "error_type": "runtime", "error": "No Strategy class defined"}

        result1, score1, result2, score2 = _run_dual_gate(strategy_code, namespace)

        ratio = score2 / score1 if score1 > 0 else 0.0
        vetoed = ratio < GATE2_VETO_RATIO and score1 > 0
        effective_score = -888.0 if vetoed else score1

        return {
            "variant_id": variant_id,
            "score": round(effective_score, 6),
            "gate1_score": round(score1, 6),
            "gate2_score": round(score2, 6),
            "ratio": round(ratio, 4),
            "vetoed": vetoed,
            "sharpe": round(result1.sharpe, 6),
            "total_return_pct": round(result1.total_return_pct, 6),
            "max_drawdown_pct": round(result1.max_drawdown_pct, 6),
            "num_trades": result1.num_trades,
            "win_rate_pct": round(result1.win_rate_pct, 6),
            "profit_factor": round(result1.profit_factor, 6),
            "backtest_seconds": round(result1.backtest_seconds + result2.backtest_seconds, 1),
        }
    except SyntaxError as e:
        return {"variant_id": variant_id, "score": -999,
                "error_type": "syntax", "error": str(e)}
    except Exception as e:
        return {"variant_id": variant_id, "score": -999,
                "error_type": "runtime", "error": str(e)}


WORKER_TIMEOUT = (TIME_BUDGET + 30) * 2  # 2x budget for both gates


def handle_batch(batch_json):
    """Run N strategy variants in parallel with dual-gate, return sorted results."""
    try:
        payload = json.loads(batch_json)
        variants = payload["variants"]
    except (json.JSONDecodeError, KeyError) as e:
        return json.dumps({"error": f"invalid batch JSON: {e}"})

    if not variants:
        return json.dumps([])

    workers = min(len(variants), os.cpu_count() or 4, 8)
    work = [(v["id"], v["code"]) for v in variants]

    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_worker_data_init, initargs=(data,)) as pool:
            async_result = pool.map_async(_run_variant, work)
            results = async_result.get(timeout=WORKER_TIMEOUT)
    except multiprocessing.TimeoutError:
        return json.dumps({"error": f"batch timed out after {WORKER_TIMEOUT}s"})
    except Exception as e:
        return json.dumps({"error": f"pool: {e}"})

    results.sort(key=lambda r: r.get("score", -999), reverse=True)
    return json.dumps(results)


def handle_run():
    """Reload strategy, run dual-gate backtest, return JSON result."""
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    try:
        from strategy import Strategy
    except Exception as e:
        return json.dumps({"error": f"import: {e}", "score": -999})

    def timeout_handler(signum, frame):
        raise TimeoutError("backtest exceeded time budget")

    sig.signal(sig.SIGALRM, timeout_handler)
    sig.alarm(TIME_BUDGET * 2 + 30)

    try:
        t_start = time.time()

        # Gate 1: standard
        prepare.REALISTIC_MODE = False
        prepare.TAKER_FEE = STANDARD_TAKER_FEE
        prepare.SLIPPAGE_BPS = STANDARD_SLIPPAGE_BPS
        strategy1 = Strategy()
        result1 = run_backtest(strategy1, data)
        score1 = compute_score(result1)

        # Gate 2: realistic
        prepare.REALISTIC_MODE = True
        prepare.TAKER_FEE = REALISTIC_TAKER_FEE
        prepare.SLIPPAGE_BPS = REALISTIC_SLIPPAGE_BPS
        strategy2 = Strategy()
        result2 = run_backtest(strategy2, data)
        score2 = compute_score(result2)

        # Restore standard
        prepare.REALISTIC_MODE = False
        prepare.TAKER_FEE = STANDARD_TAKER_FEE
        prepare.SLIPPAGE_BPS = STANDARD_SLIPPAGE_BPS

        elapsed = time.time() - t_start

        ratio = score2 / score1 if score1 > 0 else 0.0
        vetoed = ratio < GATE2_VETO_RATIO and score1 > 0
        effective_score = -888.0 if vetoed else score1

        return json.dumps({
            "score": round(effective_score, 6),
            "gate1_score": round(score1, 6),
            "gate2_score": round(score2, 6),
            "ratio": round(ratio, 4),
            "vetoed": vetoed,
            "sharpe": round(result1.sharpe, 6),
            "total_return_pct": round(result1.total_return_pct, 6),
            "max_drawdown_pct": round(result1.max_drawdown_pct, 6),
            "num_trades": result1.num_trades,
            "win_rate_pct": round(result1.win_rate_pct, 6),
            "profit_factor": round(result1.profit_factor, 6),
            "annual_turnover": round(result1.annual_turnover, 2),
            "backtest_seconds": round(result1.backtest_seconds + result2.backtest_seconds, 1),
            "total_seconds": round(elapsed, 1),
        })
    except TimeoutError:
        return json.dumps({"error": "timeout", "score": -999})
    except Exception as e:
        return json.dumps({"error": str(e), "score": -999})
    finally:
        sig.alarm(0)
        prepare.REALISTIC_MODE = False
        prepare.TAKER_FEE = STANDARD_TAKER_FEE
        prepare.SLIPPAGE_BPS = STANDARD_SLIPPAGE_BPS


def recv_all(conn, timeout=300):
    """Read all data from connection until EOF."""
    conn.settimeout(timeout)
    chunks = []
    while True:
        try:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        except socket.timeout:
            break
    return b"".join(chunks).decode().strip()


def main():
    parser = argparse.ArgumentParser(description="Resident backtest evaluator (dual-gate)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (default: 9877)")
    args = parser.parse_args()

    # Load data ONCE on startup
    t_load = time.time()
    global data
    data = load_data("val")
    load_time = time.time() - t_load

    total_bars = sum(len(df) for df in data.values())
    print(f"Loaded {total_bars} bars across {len(data)} symbols in {load_time:.1f}s", file=sys.stderr)
    print(f"Symbols: {list(data.keys())}", file=sys.stderr)
    print(f"Dual-gate enforced: gate2/gate1 < {GATE2_VETO_RATIO} → score=-888", file=sys.stderr)

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)

    print(f"Listening on 127.0.0.1:{args.port}", file=sys.stderr)
    print(f"Commands: run | ping | batch:{{...}}", file=sys.stderr)
    sys.stderr.flush()

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
            raw = recv_all(conn, timeout=300)

            if raw == "run":
                result = handle_run()
                conn.sendall((result + "\n").encode())
            elif raw == "ping":
                conn.sendall(b'{"status": "ok"}\n')
            elif raw.startswith("batch:"):
                batch_json = raw[6:]
                result = handle_batch(batch_json)
                conn.sendall((result + "\n").encode())
            else:
                conn.sendall(b'{"error": "unknown command, send: run | ping | batch:{...}"}\n')
        except Exception as e:
            try:
                conn.sendall(json.dumps({"error": str(e), "score": -999}).encode() + b"\n")
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()

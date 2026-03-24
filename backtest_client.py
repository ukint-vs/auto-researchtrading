"""
Client for the backtest server. Supports single runs and batch mode.

Usage:
    uv run backtest_client.py                          # Single run
    uv run backtest_client.py --batch variants.json    # Batch from file
    echo '{"variants":[...]}' | uv run backtest_client.py --batch -  # Batch from stdin
"""
import socket
import sys
import argparse


def send_command(cmd, port=9877, timeout=600):
    """Send command to server, return response string."""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", port))
        s.sendall(cmd.encode())
        s.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        return data.decode().strip()
    except (ConnectionRefusedError, OSError) as e:
        return f'{{"error": "server not running: {e}", "score": -999}}'
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser(description="Backtest server client")
    parser.add_argument("port", nargs="?", type=int, default=9877, help="Server port (default: 9877)")
    parser.add_argument("--batch", metavar="FILE", help="Batch mode: JSON file with variants (use - for stdin)")
    args = parser.parse_args()

    if args.batch:
        if args.batch == "-":
            batch_json = sys.stdin.read().strip()
        else:
            with open(args.batch) as f:
                batch_json = f.read().strip()
        cmd = f"batch:{batch_json}"
    else:
        cmd = "run"

    result = send_command(cmd, port=args.port)
    print(result)


if __name__ == "__main__":
    main()

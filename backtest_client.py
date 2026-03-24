"""Send 'run' to the backtest server and print the JSON result."""
import socket
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9877
s = socket.socket()
s.settimeout(130)
try:
    s.connect(("127.0.0.1", port))
    s.sendall(b"run")
    s.shutdown(socket.SHUT_WR)
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    print(data.decode().strip())
except (ConnectionRefusedError, OSError) as e:
    print(f'{{"error": "server not running: {e}", "score": -999}}')
finally:
    s.close()

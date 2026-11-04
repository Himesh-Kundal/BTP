#!/usr/bin/env python3
"""
Custom HTTP server to host BTP Documentation Viewer on the local network.
Binds to 0.0.0.0 on port 8080.
"""

import http.server
import socketserver
import os
import sys
import socket

PORT = 8080
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

def get_lan_ip():
    """Find the primary LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually send data
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class DocsHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DOCS_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and disable aggressive caching for seamless viewing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, format, *args):
        # Keep logs clean
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def main():
    lan_ip = get_lan_ip()
    print("=" * 65)
    print("  🚀 BTP Markdown Documentation Viewer Server Active")
    print("=" * 65)
    print(f"  Local Machine URL:   http://localhost:{PORT}/")
    print(f"  Local Machine (IP):  http://127.0.0.1:{PORT}/")
    print(f"  Local Network (LAN): http://{lan_ip}:{PORT}/")
    print("=" * 65)
    print(f"  Serving Directory:   {DOCS_DIR}")
    print("=" * 65)
    sys.stdout.flush()

    with ReusableTCPServer(("0.0.0.0", PORT), DocsHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()

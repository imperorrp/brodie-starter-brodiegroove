"""Serve team/web over https on the LAN so a phone's browser can use its camera (secure context), with a self-signed certificate.

  .venv\Scripts\python team\scripts\serve_web.py [--port 8443] [--ip 10.103.36.77]
Then open https://<laptop-ip>:8443/phone.html?url=...&token=... on the phone (accept the certificate warning once:
Advanced -> proceed). team/scripts/link_local.ps1 prints the full URL. The certificate is generated into team/web/certs/ with
openssl if missing (Git for Windows ships openssl: C:\\Program Files\\Git\\usr\\bin\\openssl.exe).
"""
import argparse, http.server, os, shutil, socket, ssl, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(os.path.dirname(HERE), 'web'); CERTS = os.path.join(WEB, 'certs')


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return '127.0.0.1'


def openssl_bin():
    for c in (shutil.which('openssl'), r'C:\Program Files\Git\usr\bin\openssl.exe', r'C:\Program Files\Git\mingw64\bin\openssl.exe'):
        if c and os.path.exists(c): return c
    return None


def ensure_cert(ip):
    os.makedirs(CERTS, exist_ok=True)
    crt, key = os.path.join(CERTS, 'server.crt'), os.path.join(CERTS, 'server.key')
    if os.path.exists(crt) and os.path.exists(key): return crt, key
    ob = openssl_bin()
    if ob is None: raise SystemExit('openssl not found; install Git for Windows or create team/web/certs/server.{crt,key} yourself')
    san = f'subjectAltName=DNS:localhost,IP:127.0.0.1,IP:{ip}'
    subprocess.run([ob, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', crt, '-days', '30',
                    '-subj', '/CN=brodie-dance-local', '-addext', san], check=True)
    print(f'[serve] self-signed certificate written to {CERTS} ({san})', flush=True)
    return crt, key


GO_FILE = os.path.join(os.path.dirname(WEB), '.go_url')      # written by chrome.ps1 / link_local.ps1; outside the served folder


class Quiet(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=WEB, **k)
    def log_message(self, fmt, *args): print('[serve]', self.address_string(), fmt % args, flush=True)
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store'); super().end_headers()
    def do_GET(self):
        """/go redirects to the last link the scripts wrote, so a phone only has to type https://<ip>:8443/go (the token stays
        out of the typed URL; anyone on the same Wi-Fi who opens /go gets that one-hour, one-room token: fine for a test)."""
        if self.path.split('?')[0] == '/go':
            url = open(GO_FILE, encoding='utf-8').read().strip() if os.path.exists(GO_FILE) else ''
            if not url:
                self.send_response(404); self.end_headers(); self.wfile.write(b'no link yet: run team/scripts/chrome.ps1 first'); return
            self.send_response(302); self.send_header('Location', url); self.end_headers(); return
        super().do_GET()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8443); ap.add_argument('--ip', default=None); a = ap.parse_args()
    ip = a.ip or lan_ip(); crt, key = ensure_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(crt, key)
    httpd = http.server.ThreadingHTTPServer(('0.0.0.0', a.port), Quiet); httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    print(f'[serve] https://{ip}:{a.port}/phone.html  (also https://localhost:{a.port}/phone.html); Ctrl-C to stop', flush=True)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == '__main__':
    sys.exit(main())

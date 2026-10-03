"""
Standalone (separate-process) Brumley-Boneh CRT-RSA victim, timed EXTERNALLY.

Faithful-to-Brumley-Boneh variant of bb_victim.py.  This is a NEW file; it does
not modify bb_victim.py / attack_brumley_boneh.py -- it reuses their helpers.

Differences vs the in-process bb_victim.py:
  * Runs as its OWN OS process and speaks over a localhost TCP socket (the paper
    attacked an SSL *network server*; interprocess/same-machine is one of their
    listed scenarios).  A DLL would share the attacker's process/scheduler/clock
    and could not be independently (un)pinned, so a real process is required.
  * The victim NEVER reports its own timing -- it just decrypts and returns the
    plaintext.  The *attacker* measures the round-trip time (attack_remote.py).
    This is the genuine remote-timing channel (IPC + scheduling + victim-side
    noise), unlike the in-process rdtsc oracle in bb_victim.py.
  * The victim intentionally does NOT pin its CPU / raise its priority (a
    realistic, uncontrolled server).  Only the attacker pins its own core.

To lift the per-decryption leak above IPC/timer noise the victim performs
``--batch`` CRT decryptions of the requested ciphertext per request (the
attacker times the whole batched response); this is the server-side analogue of
the attacker's neighbourhood/sample amplification.

Protocol (line-based, hex):
    server -> "<n_hex> <e_hex>\\n"          (handshake, public params)
    loop:
        client -> "<c_hex>\\n"              (ciphertext)
        server -> "<m_hex>\\n"              (m = c^d mod N, via CRT)
    client -> "BYE\\n"                       (shutdown)

Run (normally spawned by attack_remote.py):
    python bb_victim_server.py --port 50007 --key-file <shared.json> --batch 256
"""

import argparse
import os
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABB = os.path.dirname(_HERE)                 # attack_brumley_boneh/
_ROOT = os.path.dirname(_ABB)                 # montMul/
for _p in (_HERE, _ABB, os.path.join(_ROOT, "attack_refined"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import montmul_c as mc  # noqa: E402
from bb_victim import generate_crt_key, load_crt_key, save_crt_key  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Remote (socket) BB CRT-RSA victim.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--key-file", required=True,
                    help="shared key json (written here so the attacker can "
                         "load n,e and, for evaluation only, verify q/d)")
    ap.add_argument("--key-bits", type=int, default=256)
    ap.add_argument("--key-source", choices=["random", "provable"],
                    default="random")
    ap.add_argument("--batch", type=int, default=256,
                    help="server-side decryptions per request (signal amplifier)")
    ap.add_argument("--reuse-key", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()

    # NOTE: deliberately NO cpu_pin / priority change here -- the victim is an
    # uncontrolled server; only the attacker pins its measurement environment.

    key = None
    if a.reuse_key and os.path.exists(a.key_file):
        try:
            key = load_crt_key(a.key_file)
            if abs(key["n"].bit_length() - a.key_bits) > 1:
                key = None
        except Exception:
            key = None
    if key is None:
        key = generate_crt_key(a.key_bits, source=a.key_source)
    save_crt_key(a.key_file, key)          # share with the attacker

    n, e = key["n"], key["e"]
    p, q = key["p"], key["q"]
    d1, d2, qinv = key["d1"], key["d2"], key["qinv"]
    batch = max(1, a.batch)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(1)
    # Signal readiness only AFTER bind + key persisted, so the attacker never
    # races the handshake or reads a stale key file.
    print("READY %s %d" % (a.host, a.port), flush=True)

    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # no Nagle delay
    rf = conn.makefile("r")
    conn.sendall(("%X %X\n" % (n, e)).encode())

    for line in rf:
        line = line.strip()
        if not line:
            continue
        if line == "BYE":
            break
        try:
            c = int(line, 16) % n
        except ValueError:
            continue
        m1 = m2 = 0
        # batch identical CRT decryptions -> amplify the extra-reduction leak in
        # the measured (external) response time.  do_time=False: the victim does
        # NOT time itself.
        for _ in range(batch):
            m1, _, _ = mc.modexp(c, d1, p, do_time=False, batch=1)
            m2, _, _ = mc.modexp(c, d2, q, do_time=False, batch=1)
        h = (qinv * (m1 - m2)) % p
        m = m2 + h * q
        conn.sendall(("%X\n" % m).encode())

    try:
        conn.close()
        srv.close()
    except OSError:
        pass


if __name__ == "__main__":
    main()

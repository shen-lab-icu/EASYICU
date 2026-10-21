"""Operator commands. Passwords are read only by getpass, never from argv."""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import logging
import os
import re
import secrets
import ssl
import subprocess
import tempfile
from pathlib import Path

from .auth import hash_password
from .config import GatewayConfig, load_config
from .storage import (
    account_name, admin_lock, initialize_secret, instance_socket, private_directory,
    read_private, read_users, socket_identity, write_users,
)


def prompt_password() -> str:
    first = getpass.getpass("Password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second or not first:
        raise ValueError("passwords must be nonempty and match")
    return first


def make_certificate(config: GatewayConfig, host: str) -> None:
    try:
        san = "IP:" + str(ipaddress.ip_address(host))
    except ValueError:
        if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]{0,251}[a-zA-Z0-9])?", host):
            raise ValueError("certificate host must be an IP address or DNS name")
        san = "DNS:" + host
    for target in (config.cert_file, config.key_file):
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        private_directory(target.parent)
        if target.exists() or target.is_symlink():
            raise ValueError("certificate/key already exists; refusing to overwrite")
    with tempfile.TemporaryDirectory(prefix=".certificate-", dir=config.key_file.parent) as temp:
        root = Path(temp)
        # An explicit config works with both OpenSSL and macOS LibreSSL.
        (root / "openssl.cnf").write_text(
            "[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ext\n"
            f"[dn]\nCN={host}\n[ext]\nsubjectAltName={san}\n"
            "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n", encoding="utf-8")
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:3072", "-sha256", "-nodes",
                        "-days", "365", "-config", str(root / "openssl.cnf"),
                        "-keyout", str(root / "key.pem"), "-out", str(root / "cert.pem")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for source, target in ((root / "key.pem", config.key_file), (root / "cert.pem", config.cert_file)):
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(source.read_bytes())


def serve(config: GatewayConfig) -> None:
    import uvicorn

    from .app import create_app

    read_private(config.key_file)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(config.cert_file, config.key_file)
    # Uvicorn/HTTPX default access logs include query strings. Only our redacted
    # logger is enabled; the gateway itself never trusts client proxy headers.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("easyicu.hosted_gateway.access").setLevel(logging.INFO)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    uvicorn.run(create_app(config), host=config.host, port=config.port,
                ssl_certfile=str(config.cert_file), ssl_keyfile=str(config.key_file),
                proxy_headers=False, access_log=False, workers=1)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default error echoes unrecognized argv, possibly a password.
        self.exit(2, "Invalid command arguments; use --help.\n")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("add-user", "reset-password", "remove-user", "list-users",
                    "make-self-signed-cert", "serve"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, default=Path("~/easyicu_dev9/gateway/config.json"))
        if command in {"add-user", "reset-password", "remove-user"}:
            sub.add_argument("name")
        if command == "add-user":
            sub.add_argument("--instance", required=True)
            sub.add_argument("--users-root", type=Path, default=Path("~/easyicu_dev9/users"))
        if command == "make-self-signed-cert":
            sub.add_argument("--host", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "serve":
            serve(config)
            return 0
        if args.command == "make-self-signed-cert":
            make_certificate(config, args.host)
            print("Certificate and key created.")
            return 0
        if args.command == "list-users":
            for name in sorted(read_users(config.users_file)):
                print(name)
            return 0
        name = account_name(args.name)
        socket = (instance_socket(args.users_root, args.instance, config.public_origin)
                  if args.command == "add-user" else None)
        with admin_lock(config.users_file):
            try:
                users = read_users(config.users_file)
            except FileNotFoundError:
                if args.command != "add-user":
                    raise ValueError("no users file")
                users = {}
            if args.command == "add-user":
                if name in users or any(socket_identity(Path(user["socket"])) == socket_identity(socket)
                                        for user in users.values()):
                    raise ValueError("user or instance is already assigned")
                password = hash_password(prompt_password())
                initialize_secret(config.secret_file)
                users[name] = {"socket": str(socket), "password": password,
                               "version": secrets.token_hex(16)}
            else:
                if name not in users:
                    raise ValueError("unknown user")
                if args.command == "reset-password":
                    users[name]["password"] = hash_password(prompt_password())
                    users[name]["version"] = secrets.token_hex(16)
                else:
                    del users[name]
            write_users(config.users_file, users)
        print("User updated.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, subprocess.SubprocessError):
        print("Gateway command failed; check configuration, private file modes and instance assignment.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

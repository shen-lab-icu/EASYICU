"""Manage one private, Unix-socket EasyICU instance per clinician.

Run from the shared checkout (no server is started by create/provision):

  python tools/hosted_instances.py create doctor --display-name 王医生 \
    --data-root /srv/icu --runner-image easyicu-runner:reviewed \
    --public-origin https://doctor.hospital.example --support-contact 'Help desk'
  python tools/hosted_instances.py provision-model doctor --key-file /private/key
  python tools/hosted_instances.py add-source doctor /srv/icu/prepared --label 'ICU data'
  python tools/hosted_instances.py list-sources doctor
  python tools/hosted_instances.py start doctor
  python tools/hosted_instances.py status doctor
  python tools/hosted_instances.py list
  python tools/hosted_instances.py stop doctor

Global --users-root defaults to ~/easyicu_dev9/users. Each instance has home/
and run/ (0700), instance.env (0600, parsed here, never shell-sourced), and its
own socket. --checkout on create defaults to this shared checkout. Configure
the authenticating HTTPS gateway to forward Host/Origin to that socket. Pass
--public-origin on create (or set EASYICU_PUBLIC_ORIGIN for create); it must
match the gateway exactly. Model provisioning uses the existing provider_config
catalog + bounded inference verifier and enables the external-model opt-in
only after it succeeds. Key files must be private regular files; keys and
provider responses are never printed. stop/status use only the recorded PID,
creation time, and exact socket command, with no process or port sweep.
Source commands run in a subprocess with the instance's environment and use
the shared sources owner. add-source accepts only prepared exports under that
instance's data roots; --inactive keeps the current active source. Their output
contains labels, databases, counts and active flags, never source paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import psutil

CHECKOUT = Path(__file__).resolve().parents[1]
if str(CHECKOUT / "src") not in sys.path:
    sys.path.insert(0, str(CHECKOUT / "src"))


def _name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise ValueError("instance name must contain only letters, digits, '_' and '-'")
    return value


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError("instance directory must be owned by this administrator and not a symlink")
    path.chmod(0o700)


def _update_instance_settings(state: Path, patch: dict) -> None:
    from easyicu.webserver import settings

    _private_dir(state)
    previous = os.umask(0o077)
    try:
        path = state / "webserver_settings.json"
        settings.update_settings(patch, config_path=path)
        path.chmod(0o600)
    finally:
        os.umask(previous)


def instance_environment(instance: Path) -> dict[str, str]:
    """Read only the administrator's instance env, never provider credentials."""
    _name(instance.name)
    info = instance.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.geteuid():
        raise ValueError("instance directory must be private and administrator-owned")
    path = instance / "instance.env"
    with path.open(encoding="utf-8") as handle:
        if os.fstat(handle.fileno()).st_mode & 0o077:
            raise ValueError("instance env must be private")
        env = {}
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            key, value = line.rstrip().split("=", 1)
            env[key] = json.loads(value)
    return env


def create_instance(users: Path, args: argparse.Namespace) -> None:
    name = _name(args.name)
    roots = [Path(value) for value in args.data_root]
    if any(not root.is_absolute() or not root.is_dir() or os.pathsep in str(root)
           or root.resolve() == Path(root.anchor) for root in roots):
        raise ValueError("data roots must be existing absolute directories below the filesystem root")
    checkout = Path(args.checkout).resolve()
    if not (checkout / "src/easyicu/webserver/app.py").is_file():
        raise ValueError("checkout does not contain EasyICU")
    origin = args.public_origin or os.environ.get("EASYICU_PUBLIC_ORIGIN", "")
    if origin:
        from urllib.parse import urlsplit

        parsed = urlsplit(origin)
        _ = parsed.port
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("public origin must be an HTTPS origin")
    _private_dir(users)
    instance = users / name
    instance.mkdir(mode=0o700)  # Refuse an existing instance; never overwrite it.
    for child in ("home", "run"):
        _private_dir(instance / child)
    env = {
        "EASYICU_HOSTED": "1",
        "EASYICU_HOME": str(instance / "home"),
        "EASYICU_RUNTIME_DIR": str(instance / "run"),
        "EASYICU_WEB_UNIX_SOCKET": str(instance / "run/web.sock"),
        "EASYICU_USER_DISPLAY_NAME": args.display_name,
        "EASYICU_DATA_ROOTS": os.pathsep.join(str(root.resolve()) for root in roots),
        "EASYICU_RUNNER_IMAGE": args.runner_image,
        "EASYICU_CODE_ROOT": str(checkout),
        "EASYICU_WEB_TRUST_PROXY": "1",
        "EASYICU_PUBLIC_ORIGIN": origin,
    }
    path = instance / "instance.env"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for key, value in env.items():
            handle.write(f"{key}={json.dumps(value, ensure_ascii=False)}\n")
    _update_instance_settings(instance / "home/.easyicu", {
        "support_contact": args.support_contact, "language": "zh", "data_mode": "real",
    })


def provision_model(instance: Path, key_file: Path, *, verifier=None) -> None:
    """Verify and save via the same provider owner as the connection page."""
    from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore

    env = instance_environment(instance)
    # fstat the opened descriptor so a pathname replacement cannot bypass mode checks.
    descriptor = os.open(key_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.geteuid():
            raise ValueError("key file must be a private administrator-owned regular file (0600)")
        key = handle.read(8194).strip()
    if not key or len(key) > 8192 or any(ch.isspace() for ch in key):
        raise ValueError("key file must contain exactly one key")
    state = Path(env["EASYICU_HOME"]) / ".easyicu"
    _private_dir(state)
    store = PiProviderConfigStore(config_path=state / "pi-provider.env",
                                  receipt_path=state / "pi-provider-verification.json")
    store.verify_and_save(provider="deepseek", api_key=key, base_url="https://api.deepseek.com/v1",
                          model="deepseek-chat", api_transport="openai-completions", verifier=verifier)
    _update_instance_settings(state, {"ai_enabled": True})


def _source_metadata(command: str, path: str | None = None, *, label: str | None = None,
                     active: bool = True) -> dict:
    """Run only in the instance subprocess; return registry-safe metadata."""
    from fastapi import HTTPException

    from easyicu.webserver import sources
    from easyicu.webserver.hosted import require_path

    try:
        if command == "add-source":
            # No instance-owned exception: this explicit grant must be in data roots.
            require_path(path)
        _private_dir(Path(os.environ["EASYICU_HOME"]) / ".easyicu")
        registry = (sources.register_source(path, label=label, active=active)
                    if command == "add-source" else sources.load_registry())
        if not registry.get("ok"):
            return {"ok": False, "error": registry["error"]}
        rows = registry.get("sources") or []
        if command == "add-source":
            selected = str(Path(path).expanduser().resolve())
            rows = [row for row in rows if row["path"] == selected]
        return {
            "ok": True,
            "sources": [{
                "label": row.get("label"), "database": row.get("database"),
                "stays": (row.get("summary") or {}).get("stays"),
                "total_rows": (row.get("summary") or {}).get("total_rows"),
                "active": row.get("path") == registry.get("active_path"),
            } for row in rows],
            "registered_source_counts": sources.registered_source_counts(),
        }
    except HTTPException as exc:
        return {"ok": False, "error": exc.detail["error"]}
    except Exception as exc:
        return {"ok": False, "error": getattr(exc, "code", type(exc).__name__)}


def instance_sources(instance: Path, args: argparse.Namespace) -> int:
    """Import sources afresh under the instance environment, never the admin's."""
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "LANG", "TMPDIR", "SYSTEMROOT"} or key.startswith("LC_")}
    env.update(instance_environment(instance))
    checkout = Path(env["EASYICU_CODE_ROOT"])
    env["HOME"] = env["EASYICU_HOME"]
    env["PYTHONPATH"] = str(checkout / "src")
    options = {"command": args.command}
    if args.command == "add-source":
        options.update(path=str(args.path), label=args.label, active=not args.inactive)
    script = (
        "import json, os, sys; os.umask(0o077); "
        "from tools.hosted_instances import _source_metadata; "
        "print(json.dumps(_source_metadata(**json.loads(sys.argv[1])), ensure_ascii=False))"
    )
    result = subprocess.run([sys.executable, "-c", script, json.dumps(options)],
                            cwd=checkout, env=env, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    if not payload.get("ok"):
        print(f"Instance action failed ({payload['error']}).", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _pid_path(instance: Path) -> Path:
    return instance / "run/instance.pid.json"


def _recorded_process(instance: Path):
    try:
        record = json.loads(_pid_path(instance).read_text(encoding="utf-8"))
        process = psutil.Process(int(record["pid"]))
        if abs(process.create_time() - float(record["create_time"])) > 0.01:
            return None
        command = process.cmdline()
        socket = str(instance / "run/web.sock")
        if ("uvicorn" not in command or "easyicu.webserver.app:app" not in command
                or "--uds" not in command or command[command.index("--uds") + 1] != socket):
            return None
        return process
    except (OSError, ValueError, KeyError, IndexError, TypeError, psutil.Error):
        return None


def start_instance(instance: Path) -> None:
    from easyicu.webserver.host_security import require_private_unix_socket_directory

    env = instance_environment(instance)
    if _recorded_process(instance) is not None:
        raise ValueError("instance is already running")
    socket = env["EASYICU_WEB_UNIX_SOCKET"]
    require_private_unix_socket_directory(socket)
    checkout = Path(env["EASYICU_CODE_ROOT"])
    inherited = {key: value for key, value in os.environ.items() if key in {
        "PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT", "SSL_CERT_FILE",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "DOCKER_HOST", "DOCKER_CONFIG",
    } or key.startswith("LC_")}
    inherited.update(env)
    inherited["PYTHONPATH"] = str(checkout / "src")
    command = [sys.executable, "-m", "uvicorn", "easyicu.webserver.app:app", "--uds", socket,
               "--proxy-headers", "--forwarded-allow-ips", "*"]
    with (instance / "run/web.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=checkout, env=inherited, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    identity = psutil.Process(process.pid).create_time()
    if process.poll() is not None:
        raise ValueError("instance failed to start; inspect its private log")
    path = _pid_path(instance)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": process.pid, "create_time": identity}), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def stop_instance(instance: Path) -> None:
    instance_environment(instance)
    process = _recorded_process(instance)
    if process is not None:
        process.terminate()
        process.wait(timeout=15)
    _pid_path(instance).unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users-root", type=Path, default=Path.home() / "easyicu_dev9/users")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("name")
    create.add_argument("--display-name", required=True)
    create.add_argument("--data-root", action="append", required=True)
    create.add_argument("--support-contact", default="")
    create.add_argument("--runner-image", required=True)
    create.add_argument("--checkout", type=Path, default=CHECKOUT)
    create.add_argument("--public-origin", default="")
    for name in ("start", "stop", "status", "provision-model", "add-source", "list-sources"):
        help_text = {"add-source": "Register a prepared export in this instance's data roots",
                     "list-sources": "List source labels, databases, counts and active flags"}.get(name)
        command = commands.add_parser(name, help=help_text)
        command.add_argument("name")
        if name == "provision-model":
            command.add_argument("--key-file", type=Path, required=True)
        elif name == "add-source":
            command.add_argument("path", type=Path)
            command.add_argument("--label")
            command.add_argument("--inactive", action="store_true", help="Keep the current active source")
    commands.add_parser("list")
    args = parser.parse_args(argv)
    try:
        users = args.users_root.expanduser().absolute()
        if args.command == "create":
            create_instance(users, args)
        elif args.command == "list":
            for instance in sorted(users.iterdir()) if users.exists() else []:
                if instance.is_dir() and not instance.is_symlink():
                    instance_environment(instance)
                    print(f"{instance.name}: {'running' if _recorded_process(instance) else 'stopped'}")
        else:
            instance = users / _name(args.name)
            if args.command == "start":
                start_instance(instance)
            elif args.command == "stop":
                stop_instance(instance)
            elif args.command == "provision-model":
                provision_model(instance, args.key_file)
            elif args.command in {"add-source", "list-sources"}:
                return instance_sources(instance, args)
            else:
                instance_environment(instance)
                print(f"{instance.name}: {'running' if _recorded_process(instance) else 'stopped'}")
        return 0
    except Exception as exc:
        # Provider/network exceptions can contain upstream text: report a code,
        # never its message, repr, request, or credential.
        print(f"Instance action failed ({getattr(exc, 'code', type(exc).__name__)}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# Shared metadata-provider capacity

The shared coordinator covers EasyICU's guarded metadata-provider calls from
the **enrolled** homes that all use this version and the same coordinator. It
groups only identical endpoint and credential fingerprints; it cannot identify
different keys or proxy addresses belonging to the same provider account. Other
EasyICU provider paths and external clients are outside this scope.
The coordinator currently requires POSIX ownership and file-permission checks;
it fails closed on Windows until equivalent ACL and locking evidence is available.

An administrator must first inventory every EasyICU home participating in the
deployment, stop those instances, and prevent old or unenrolled instances from
issuing requests to the same provider capacity. If the inventory or retirement
of old binaries cannot be established, do not activate the shared scope.
EasyICU cannot discover arbitrary historical homes or enforce that an old binary
uses the new coordinator.

Create an empty private directory and run the setup in an interactive terminal:

```sh
mkdir -m 700 /absolute/private/coordinator
python -m easyicu.webserver.metadata_provider_shared_scope \
  /absolute/private/coordinator \
  --home /absolute/first/EASYICU_HOME \
  --home /absolute/second/EASYICU_HOME
```

The command shows the coordinator, each exact state directory and its number of
legacy slots before asking the administrator to type `ENROLL`. It takes the
listed Web instance locks and verifies that those slot snapshots have not
changed since the preview. It copies all slots into one newly published shared
database, leaving the original databases and every Provider budget unchanged.
It also writes a private enrollment marker in each listed state directory. If
setup is interrupted after writing a marker, that home stays blocked until the
same scope is completed; it will not fall back to the old local ledger.
An existing scope is never overwritten. A crash before publication leaves no
active partial scope.

Start **every enrolled instance** with the same absolute setting:

```sh
EASYICU_METADATA_PROVIDER_SHARED_ROOT=/absolute/private/coordinator
```

For a hosted instance managed by `tools/hosted_instances.py`, the setting must
be present as a JSON string in that instance's private `instance.env` before it
is started. Never set it for just one instance in the intended shared domain.
The configured coordinator and member state directories must be private,
owned by the running user and free of symlinks. An uninitialized scope, an
unenrolled home, or a changed legacy slot database stops admission without
falling back to an independent ledger.

Imported slots are never automatically released: the current system has no
durable remote-terminal proof suitable for retiring them. In particular, a
local worker exit or elapsed time does not prove a timed-out remote request
ended. An unresolved imported slot may therefore continue to block planning.
Preserve the source ledger and seek a provider-specific terminal receipt and a
verified recovery path; do not delete the slot or increase capacity to make a
test appear to pass.

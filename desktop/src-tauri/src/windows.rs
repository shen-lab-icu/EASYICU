//! Windows storage and a CI launch that does not construct a WebView.

use std::fs::{self, File};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::{
    available_loopback_port, backend_binary, health_check, node_binary, DESKTOP_TOKEN_ENV,
    READY_TIMEOUT,
};
use uuid::Uuid;

pub(super) fn data_root() -> Result<PathBuf, String> {
    let local = std::env::var_os("LOCALAPPDATA").ok_or("LOCALAPPDATA is not set")?;
    let root = PathBuf::from(local);
    if !root.is_absolute() {
        return Err("LOCALAPPDATA must be an absolute path".into());
    }
    // This stable name is separate from the NSIS product's install directory.
    Ok(root.join("EasyICU"))
}

pub(super) fn configure_webview_data() -> Result<(), Box<dyn std::error::Error>> {
    let directory = data_root()?.join("runtime").join("webview2");
    fs::create_dir_all(&directory)?;
    // Set before Tauri constructs the first WebView2 environment.
    std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", directory);
    Ok(())
}

pub(super) fn smoke_test() -> Result<(), Box<dyn std::error::Error>> {
    let root = data_root()?;
    let state = root.join("state");
    let runtime = root.join("runtime");
    let logs = root.join("logs");
    for path in [&state, &runtime, &logs] {
        fs::create_dir_all(path)?;
    }
    let result_path = runtime.join("desktop-smoke-result.json");
    if result_path.exists() {
        fs::remove_file(&result_path)?;
    }
    let executable = std::env::current_exe()?;
    let resources = executable.parent().ok_or("desktop executable has no directory")?;
    let port = available_loopback_port()?;
    let token = Uuid::new_v4().simple().to_string() + &Uuid::new_v4().simple().to_string();
    let log = File::create(logs.join("desktop-smoke.log"))?;
    let mut child = Command::new(backend_binary(resources)?)
        .creation_flags(0x08000000) // CREATE_NO_WINDOW: no console in headless CI.
        .arg("--port").arg(port.to_string())
        .arg("--state-dir").arg(&state)
        .arg("--runtime-dir").arg(&runtime)
        .arg("--parent-pid").arg(std::process::id().to_string())
        .arg("--node-bin").arg(node_binary(resources)?)
        .env(DESKTOP_TOKEN_ENV, &token)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()?;

    // Keep cleanup outside the fallible readiness loop so every outcome reaps
    // the sidecar. The normal backend also watches this process's lifetime.
    let outcome = (|| -> Result<(), Box<dyn std::error::Error>> {
        let deadline = Instant::now() + READY_TIMEOUT;
        while Instant::now() < deadline {
            if let Some(status) = child.try_wait()? {
                return Err(format!("backend exited before readiness: {status}").into());
            }
            if health_check(port, &token) {
                if health_check(port, "invalid-smoke-token") {
                    return Err("backend accepted an unauthenticated readiness request".into());
                }
                fs::write(&result_path, "{\"healthy\":true,\"unauthenticated_rejected\":true}\n")?;
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Err("backend readiness timed out".into())
    })();
    let _ = child.kill();
    child.wait()?;
    outcome
}

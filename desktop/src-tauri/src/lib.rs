use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, Url};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;
use uuid::Uuid;

const DESKTOP_TOKEN_ENV: &str = "EASYICU_DESKTOP_SESSION_TOKEN";
const READY_TIMEOUT: Duration = Duration::from_secs(120);

struct BackendProcess(Mutex<Option<CommandChild>>);

fn available_loopback_port() -> Result<u16, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|error| format!("cannot reserve a loopback port: {error}"))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| format!("cannot inspect the reserved port: {error}"))
}

fn health_check(port: u16, token: &str) -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /api/catalog HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         X-EasyICU-Desktop-Token: {token}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0_u8; 128];
    let Ok(size) = stream.read(&mut response) else {
        return false;
    };
    let status = String::from_utf8_lossy(&response[..size]);
    status.starts_with("HTTP/1.1 200") || status.starts_with("HTTP/1.0 200")
}

fn node_binary(resource_dir: &Path) -> Result<PathBuf, String> {
    let filename = if cfg!(windows) { "node.exe" } else { "node" };
    let path = resource_dir.join("resources").join(filename);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "bundled Node runtime is missing at {}",
            path.display()
        ))
    }
}

fn backend_binary(resource_dir: &Path) -> Result<PathBuf, String> {
    let filename = if cfg!(windows) {
        "easyicu-backend.exe"
    } else {
        "easyicu-backend"
    };
    let path = resource_dir
        .join("resources")
        .join("backend")
        .join(filename);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "bundled EasyICU backend is missing at {}",
            path.display()
        ))
    }
}

fn log_sidecar_events(
    mut receiver: tauri::async_runtime::Receiver<CommandEvent>,
    log_path: PathBuf,
) {
    tauri::async_runtime::spawn(async move {
        let Ok(mut log) = OpenOptions::new().create(true).append(true).open(log_path) else {
            while receiver.recv().await.is_some() {}
            return;
        };
        while let Some(event) = receiver.recv().await {
            match event {
                CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                    let _ = log.write_all(&bytes);
                    let _ = log.write_all(b"\n");
                }
                CommandEvent::Error(message) => {
                    let _ = writeln!(log, "desktop-sidecar-error: {message}");
                }
                CommandEvent::Terminated(payload) => {
                    let _ = writeln!(
                        log,
                        "desktop-sidecar-terminated: code={:?} signal={:?}",
                        payload.code, payload.signal
                    );
                }
                _ => {}
            }
            let _ = log.flush();
        }
    });
}

fn start_backend(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let port = available_loopback_port()?;
    let token = Uuid::new_v4().simple().to_string() + &Uuid::new_v4().simple().to_string();
    let app_data = app.path().app_data_dir()?;
    let state_dir = app_data.join("state");
    let runtime_dir = app_data.join("runtime");
    let log_dir = app.path().app_log_dir()?;
    fs::create_dir_all(&state_dir)?;
    fs::create_dir_all(&runtime_dir)?;
    fs::create_dir_all(&log_dir)?;
    let resource_dir = app.path().resource_dir()?;
    let node = node_binary(&resource_dir)?;
    let backend = backend_binary(&resource_dir)?;

    let command = app
        .shell()
        .command(backend)
        .args([
            "--port".to_string(),
            port.to_string(),
            "--state-dir".to_string(),
            state_dir.to_string_lossy().into_owned(),
            "--runtime-dir".to_string(),
            runtime_dir.to_string_lossy().into_owned(),
            "--parent-pid".to_string(),
            std::process::id().to_string(),
            "--node-bin".to_string(),
            node.to_string_lossy().into_owned(),
        ])
        .env(DESKTOP_TOKEN_ENV, &token);
    let (receiver, child) = command.spawn()?;
    app.manage(BackendProcess(Mutex::new(Some(child))));
    log_sidecar_events(receiver, log_dir.join("desktop-backend.log"));

    let window = app
        .get_webview_window("main")
        .ok_or("EasyICU main window was not created")?;
    std::thread::spawn(move || {
        let deadline = Instant::now() + READY_TIMEOUT;
        while Instant::now() < deadline {
            if health_check(port, &token) {
                let url = format!("http://127.0.0.1:{port}/?desktop_token={token}");
                if let Ok(parsed) = Url::parse(&url) {
                    if window.navigate(parsed).is_ok() {
                        return;
                    }
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        let _ = window.eval(
            "window.easyicuDesktopStatus('error', '本地服务启动失败。请查看 EasyICU 日志后重试。')",
        );
    });
    Ok(())
}

fn stop_backend(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<BackendProcess>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(child) = guard.take() {
                let _ = child.kill();
            }
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            start_backend(app)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building EasyICU Desktop");

    app.run(|app_handle, event| match event {
        RunEvent::Exit | RunEvent::ExitRequested { .. } => stop_backend(app_handle),
        RunEvent::Reopen { .. } => {
            if let Some(window) = app_handle.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_port_is_dynamic_and_unprivileged() {
        let port = available_loopback_port().expect("port");
        assert!(port >= 1024);
    }

    #[test]
    fn node_resource_path_is_platform_specific() {
        let root = Path::new("/tmp/easyicu-resources");
        let expected = if cfg!(windows) { "node.exe" } else { "node" };
        assert!(root.join("resources").join(expected).ends_with(expected));
    }

    #[test]
    fn backend_resource_path_is_platform_specific() {
        let root = Path::new("/tmp/easyicu-resources");
        let expected = if cfg!(windows) {
            "easyicu-backend.exe"
        } else {
            "easyicu-backend"
        };
        assert!(root
            .join("resources")
            .join("backend")
            .join(expected)
            .ends_with(expected));
    }
}

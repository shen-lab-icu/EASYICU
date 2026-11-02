//! App-scoped, non-secret desktop Docker selection.
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DockerRuntime {
    pub(crate) socket: PathBuf,
    pub(crate) image: String,
}

pub(crate) fn load(runtime_dir: &Path) -> Result<Option<DockerRuntime>, &'static str> {
    let config = runtime_dir.join("docker-runner.conf");
    let metadata = match fs::symlink_metadata(&config) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("desktop_docker_config_unreadable"),
    };
    if !metadata.file_type().is_file() || metadata.len() > 4096 {
        return Err("desktop_docker_config_invalid");
    }
    let content = fs::read_to_string(config).map_err(|_| "desktop_docker_config_unreadable")?;
    let mut schema = None;
    let mut socket = None;
    let mut image = None;
    for line in content.lines() {
        let (key, value) = line
            .split_once('=')
            .ok_or("desktop_docker_config_invalid")?;
        let slot = match key {
            "schema_version" => &mut schema,
            "docker_socket" => &mut socket,
            "runner_image" => &mut image,
            _ => return Err("desktop_docker_config_invalid"),
        };
        if slot.replace(value).is_some() || value.is_empty() || value.chars().any(char::is_control)
        {
            return Err("desktop_docker_config_invalid");
        }
    }
    if schema != Some("easyicu.desktop-docker-runner/1") {
        return Err("desktop_docker_config_invalid");
    }
    let socket = PathBuf::from(socket.ok_or("desktop_docker_config_invalid")?);
    if !socket.is_absolute() {
        return Err("desktop_docker_socket_invalid");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        match fs::symlink_metadata(&socket) {
            Ok(metadata) if metadata.file_type().is_socket() => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            _ => return Err("desktop_docker_socket_invalid"),
        }
    }
    #[cfg(not(unix))]
    return Err("desktop_docker_socket_unsupported");
    #[cfg(unix)]
    {
        let image = image.ok_or("desktop_docker_config_invalid")?;
        let digest = image
            .strip_prefix("sha256:")
            .ok_or("desktop_docker_image_invalid")?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("desktop_docker_image_invalid");
        }
        Ok(Some(DockerRuntime {
            socket,
            image: image.to_owned(),
        }))
    }
}

pub(crate) fn child_environment(
    runtime: &DockerRuntime,
    current: impl IntoIterator<Item = (OsString, OsString)>,
) -> Vec<(OsString, OsString)> {
    let blocked = [
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_CERTDIR",
        "EASYICU_RUNNER_IMAGE",
    ];
    let mut result: Vec<_> = current
        .into_iter()
        .filter(|(key, _)| !blocked.iter().any(|name| key == name))
        .collect();
    let mut host = OsString::from("unix://");
    host.push(runtime.socket.as_os_str());
    result.push((OsString::from("DOCKER_HOST"), host));
    result.push((
        OsString::from("EASYICU_RUNNER_IMAGE"),
        OsString::from(&runtime.image),
    ));
    result
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn temp() -> PathBuf {
        let path = PathBuf::from("/tmp").join(format!(
            "ei-rc-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        path
    }
    fn write(dir: &Path, socket: &Path, image: &str) {
        fs::write(dir.join("docker-runner.conf"), format!("schema_version=easyicu.desktop-docker-runner/1\ndocker_socket={}\nrunner_image={}\n", socket.display(), image)).unwrap();
    }
    #[test]
    fn missing_keeps_existing_behavior() {
        let d = temp();
        assert_eq!(load(&d), Ok(None));
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn valid_socket_and_immutable_image_bind_only_child_environment() {
        let d = temp();
        let socket = d.join("docker.sock");
        let _listener = UnixListener::bind(&socket).unwrap();
        let image = format!("sha256:{}", "a".repeat(64));
        write(&d, &socket, &image);
        let runtime = load(&d).unwrap().unwrap();
        let inherited = [
            ("DOCKER_CONTEXT", "wrong"),
            ("DOCKER_TLS_VERIFY", "1"),
            ("DOCKER_HOST", "old"),
            ("EASYICU_RUNNER_IMAGE", "old"),
            ("KEEP_ME", "yes"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()));
        let env = child_environment(&runtime, inherited);
        assert!(env.contains(&("KEEP_ME".into(), "yes".into())));
        assert!(env.contains(&(
            "DOCKER_HOST".into(),
            format!("unix://{}", socket.display()).into()
        )));
        assert!(env.contains(&("EASYICU_RUNNER_IMAGE".into(), image.into())));
        assert!(!env
            .iter()
            .any(|(k, _)| k == "DOCKER_CONTEXT" || k == "DOCKER_TLS_VERIFY"));
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn invalid_config_fails_closed() {
        let d = temp();
        let socket = d.join("docker.sock");
        let _listener = UnixListener::bind(&socket).unwrap();
        for contents in [format!("schema_version=easyicu.desktop-docker-runner/1\ndocker_socket={}\nrunner_image=tag\n",socket.display()), format!("schema_version=easyicu.desktop-docker-runner/1\ndocker_socket=relative\nrunner_image=sha256:{}\n", "a".repeat(64)), format!("schema_version=easyicu.desktop-docker-runner/1\ndocker_socket={}\nrunner_image=sha256:{}\nextra=1\n",socket.display(),"a".repeat(64)), format!("schema_version=easyicu.desktop-docker-runner/1\ndocker_socket={}\nrunner_image=sha256:{}\nrunner_image=sha256:{}\n",socket.display(),"a".repeat(64),"a".repeat(64))] { fs::write(d.join("docker-runner.conf"),contents).unwrap(); assert!(load(&d).is_err()); }
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn absent_socket_retains_selected_endpoint() {
        let d = temp();
        let path = d.join("offline.sock");
        write(&d, &path, &format!("sha256:{}", "a".repeat(64)));
        assert_eq!(load(&d).unwrap().unwrap().socket, path);
        fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn symlink_or_regular_file_is_not_a_socket() {
        let d = temp();
        let file = d.join("not-a-socket");
        fs::write(&file, b"").unwrap();
        write(&d, &file, &format!("sha256:{}", "a".repeat(64)));
        assert!(load(&d).is_err());
        let socket = d.join("socket");
        let _listener = UnixListener::bind(&socket).unwrap();
        let link = d.join("alias");
        std::os::unix::fs::symlink(&socket, &link).unwrap();
        write(&d, &link, &format!("sha256:{}", "a".repeat(64)));
        assert!(load(&d).is_err());
        fs::remove_dir_all(d).unwrap();
    }
}

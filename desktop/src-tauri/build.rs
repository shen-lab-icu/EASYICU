fn main() {
    #[cfg(windows)]
    tauri_build::try_build(tauri_build::Attributes::new().windows_attributes(
        tauri_build::WindowsAttributes::new()
            .app_manifest(include_str!("../windows-backend.manifest")),
    ))
    .expect("cannot build Windows desktop resources");
    #[cfg(not(windows))]
    tauri_build::build()
}

"""File system utilities (R ricu utils-file.R).

Functions for file and directory operations, configuration management,
and data directory handling.
"""

import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Generator
import platform

# ============================================================================
# System utilities
# ============================================================================

def sys_name() -> str:
    """Get system name (R ricu sys_name).
    
    Returns:
        System name: 'Linux', 'Darwin' (macOS), 'Windows', etc.
    """
    return platform.system()

def sys_env(var: str, unset: str = "") -> str:
    """Get environment variable (R ricu sys_env).
    
    Args:
        var: Environment variable name
        unset: Value to return if not set
        
    Returns:
        Environment variable value or unset value
    """
    return os.environ.get(var, unset)

# ============================================================================
# Data directory management
# ============================================================================

def data_dir(subdir: Optional[str] = None, create: bool = True) -> str:
    """Get data directory path (R ricu data_dir).
    
    Determines the location where to place data meant to persist between
    individual sessions.
    
    For data, the default location depends on the operating system as:
    - Linux: ~/.local/share/ricu
    - macOS: ~/Library/Application Support/ricu
    - Windows: %LOCALAPPDATA%/ricu
    
    The environment variable RICU_DATA_PATH can be used to overwrite the
    default location.
    
    Args:
        subdir: Subdirectory to create below data directory
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        Path to data directory
        
    Examples:
        >>> data_dir()
        '/home/user/.local/share/ricu'
        >>> data_dir('mimic')
        '/home/user/.local/share/ricu/mimic'
    """
    res = sys_env("RICU_DATA_PATH", unset="")
    
    if not res:
        system = sys_name()
        
        if system == "Darwin":  # macOS
            base = sys_env("XDG_DATA_HOME", os.path.expanduser("~/Library/Application Support"))
        elif system == "Windows":
            base = sys_env("LOCALAPPDATA", sys_env("APPDATA", os.path.expanduser("~")))
        else:  # Linux and others
            base = sys_env("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        
        res = os.path.join(base, "ricu")
    
    if subdir:
        res = os.path.join(res, subdir)
    
    if create:
        res = ensure_dirs(res)
    
    return os.path.normpath(res)

def src_data_dir(src: Union[str, List[str]]) -> Union[str, List[str]]:
    """Get source data directory (R ricu src_data_dir).
    
    Args:
        src: Data source name(s)
        
    Returns:
        Path(s) to source data directory
    """
    if isinstance(src, list):
        return [src_data_dir(s) for s in src]
    
    # For demo datasets, check for package installation
    # For now, just use data_dir
    return data_dir(src, create=False)

# ============================================================================
# Directory operations
# ============================================================================

def ensure_dirs(paths: Union[str, List[str]]) -> Union[str, List[str]]:
    """Ensure directories exist, creating them if necessary (R ricu ensure_dirs).
    
    Args:
        paths: Path or list of paths to ensure
        
    Returns:
        Same path(s) after ensuring they exist
        
    Raises:
        RuntimeError: If directory creation fails
    """
    if isinstance(paths, list):
        return [ensure_dirs(p) for p in paths]
    
    path = paths
    
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise RuntimeError(f"Path exists but is not a directory: {path}")
    else:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Could not create directory {path}: {e}")
    
    return path

def is_dir(path: str) -> bool:
    """Check if path is a directory (R ricu is.dir).
    
    Args:
        path: Path to check
        
    Returns:
        True if path is a directory
    """
    return os.path.isdir(path)

def dir_exists(path: str) -> bool:
    """Check if directory exists (R ricu dir.exists).
    
    Args:
        path: Path to check
        
    Returns:
        True if directory exists
    """
    return os.path.isdir(path)

def dir_create(path: str, recursive: bool = True) -> bool:
    """Create directory (R ricu dir.create).
    
    Args:
        path: Path to create
        recursive: Create parent directories if needed
        
    Returns:
        True if successful
    """
    try:
        os.makedirs(path, exist_ok=True) if recursive else os.mkdir(path)
        return True
    except OSError:
        return False

# ============================================================================
# File operations
# ============================================================================

def file_size_format(size: int, binary: bool = True) -> str:
    """Format file size in human-readable format (R ricu file_size_format).
    
    Args:
        size: Size in bytes
        binary: Use binary (1024) or decimal (1000) units
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    base = 1024 if binary else 1000
    units = ["B", "KB", "MB", "GB", "TB"] if not binary else ["B", "KiB", "MiB", "GiB", "TiB"]
    
    if size < base:
        return f"{size} {units[0]}"
    
    for i, unit in enumerate(units[1:], 1):
        size /= base
        if size < base or i == len(units) - 1:
            return f"{size:.1f} {unit}"
    
    return f"{size:.1f} {units[-1]}"

def file_copy_safe(src: str, dst: str, overwrite: bool = False) -> bool:
    """Safely copy file (R ricu file_copy_safe).
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Allow overwriting existing file
        
    Returns:
        True if successful
    """
    import shutil
    
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"Destination exists and overwrite=False: {dst}")
    
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False

def read_lines_chunked(file_path: str, chunk_size: int = 10000) -> Generator[List[str], None, None]:
    """Read large file in chunks (R ricu read_lines_chunked).
    
    Args:
        file_path: Path to file
        chunk_size: Number of lines per chunk
        
    Yields:
        Chunks of lines
    """
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line.rstrip('\n'))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

# ============================================================================
# Configuration management
# ============================================================================

def default_config_path() -> str:
    """Get default config path (R ricu default_config_path).
    
    Returns:
        Path to default config directory
    """
    # In Python package, configs are in pyricu/data/
    from importlib import resources
    try:
        # Python 3.9+
        with resources.as_file(resources.files('pyricu').joinpath('data')) as path:
            return str(path)
    except AttributeError:
        # Python 3.7-3.8
        import pkg_resources
        return pkg_resources.resource_filename('pyricu', 'data')

def user_config_path() -> Optional[List[str]]:
    """Get user config path from environment (R ricu user_config_path).
    
    Returns:
        List of user config paths or None
    """
    res = sys_env("RICU_CONFIG_PATH", unset="")
    
    if not res:
        return None
    
    return [p.strip() for p in res.split(",")]

def config_paths() -> List[str]:
    """Get all config paths (R ricu config_paths).
    
    Returns:
        List of config directory paths
    """
    user_paths = user_config_path()
    default_path = default_config_path()
    
    if user_paths:
        return user_paths + [default_path]
    else:
        return [default_path]

def get_config(name: str, cfg_dirs: Optional[List[str]] = None,
               combine_fun: Optional[Callable] = None, **kwargs) -> Any:
    """Get configuration from JSON files (R ricu get_config).
    
    Iterates over config directories and reads JSON files with the specified
    name, combining results with combine_fun.
    
    Args:
        name: Config file name (without .json extension)
        cfg_dirs: Config directories to search (default: config_paths())
        combine_fun: Function to combine results from multiple files
        **kwargs: Additional arguments for json.load
        
    Returns:
        Combined configuration data
        
    Examples:
        >>> cfg = get_config("concept-dict")
        >>> cfg = get_config("data-sources")
    """
    if cfg_dirs is None:
        cfg_dirs = config_paths()
    
    if combine_fun is None:
        combine_fun = lambda x, y: {**x, **y} if isinstance(x, dict) and isinstance(y, dict) else y
    
    results = []
    
    for cfg_dir in cfg_dirs:
        file_path = os.path.join(cfg_dir, f"{name}.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f, **kwargs)
                    results.append(data)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    
    if not results:
        return None
    
    # Combine results
    if combine_fun is None:
        return results
    
    from functools import reduce
    return reduce(combine_fun, results)

def read_json(path: str, simplify_vector: bool = True, 
              simplify_dataframe: bool = False,
              simplify_matrix: bool = False, **kwargs) -> Any:
    """Read JSON file (R ricu read_json).
    
    Args:
        path: File path
        simplify_vector: Simplify to vector (not used in Python)
        simplify_dataframe: Simplify to DataFrame (not used in Python)
        simplify_matrix: Simplify to matrix (not used in Python)
        **kwargs: Additional arguments for json.load
        
    Returns:
        Parsed JSON data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f, **kwargs)

def set_config(x: Any, name: str, dir: Optional[str] = None, **kwargs) -> None:
    """Write configuration to JSON file (R ricu set_config).
    
    Args:
        x: Object to write
        name: Config file name (without .json extension)
        dir: Directory to write to
        **kwargs: Additional arguments for json.dump
    """
    file_path = f"{name}.json"
    
    if dir is not None:
        ensure_dirs(dir)
        file_path = os.path.join(dir, file_path)
    
    write_json(x, file_path, **kwargs)

def write_json(x: Any, path: str, indent: int = 2, **kwargs) -> None:
    """Write JSON file (R ricu write_json).
    
    Args:
        x: Object to write
        path: File path
        indent: Indentation level
        **kwargs: Additional arguments for json.dump
    """
    # Ensure directory exists
    dir_path = os.path.dirname(path)
    if dir_path:
        ensure_dirs(dir_path)
    
    with open(path, 'w') as f:
        json.dump(x, f, indent=indent, **kwargs)

# ============================================================================
# Auto-attach sources
# ============================================================================

def auto_attach_srcs() -> List[str]:
    """Get list of sources to auto-attach (R ricu auto_attach_srcs).
    
    Returns:
        List of source names
    """
    res = sys_env("RICU_SRC_LOAD", unset="")
    
    if not res:
        return ["mimic", "mimic_demo", "eicu", "eicu_demo", "hirid", "aumc", "miiv", "sic"]
    else:
        return [s.strip() for s in res.split(",")]

# ============================================================================
# Utility functions
# ============================================================================

def file_exists(path: str) -> bool:
    """Check if file exists (pyricu utility).
    
    Args:
        path: Path to check
        
    Returns:
        True if file exists
    """
    return os.path.isfile(path)

def get_file_size(path: str) -> int:
    """Get file size in bytes (pyricu utility).
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(path)

def list_files(directory: str, pattern: Optional[str] = None, 
               recursive: bool = False) -> List[str]:
    """List files in directory (pyricu utility).
    
    Args:
        directory: Directory path
        pattern: File pattern (e.g., "*.csv")
        recursive: Search recursively
        
    Returns:
        List of file paths
    """
    from glob import glob
    
    if pattern is None:
        pattern = "*"
    
    search_pattern = os.path.join(directory, "**" if recursive else "", pattern)
    return glob(search_pattern, recursive=recursive)

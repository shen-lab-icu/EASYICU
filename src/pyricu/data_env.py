"""Data environment management for lazy loading (R ricu data-env.R).

Provides SrcEnv class for managing lazy-loaded data sources and tables,
corresponding to R ricu's data environment system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import weakref

import pandas as pd

from .config import SrcConfig
from .file_utils import ensure_dirs_exist


class SrcEnv:
    """Source environment for lazy loading tables (R ricu src_env).
    
    Manages a collection of tables from a data source, loading them
    on demand and optionally caching in memory.
    
    Attributes:
        name: Data source name (e.g., 'mimic', 'eicu')
        config: Source configuration
        tables: Dictionary of loaded tables
        data_dir: Directory containing data files
        
    Examples:
        >>> env = SrcEnv('mimic', config)
        >>> # Table loaded on first access
        >>> patients = env.load_table('patients')
        >>> # Cached for subsequent access
        >>> patients2 = env.load_table('patients')
        >>> assert patients is patients2
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[SrcConfig] = None,
        data_dir: Optional[Path] = None,
        auto_attach: bool = True,
    ):
        """Initialize source environment.
        
        Args:
            name: Data source name
            config: Source configuration (loaded if None)
            data_dir: Data directory (from config if None)
            auto_attach: Whether to auto-attach on creation
        """
        self.name = name
        self.config = config
        self._tables: Dict[str, pd.DataFrame] = {}
        self._table_refs: Dict[str, weakref.ref] = {}
        self.data_dir = data_dir
        
        if auto_attach:
            self.attach()
    
    def attach(self) -> None:
        """Attach the source environment (make available)."""
        # Register this environment globally
        _attached_envs[self.name] = self
    
    def detach(self) -> None:
        """Detach the source environment."""
        if self.name in _attached_envs:
            del _attached_envs[self.name]
    
    def is_attached(self) -> bool:
        """Check if environment is attached."""
        return self.name in _attached_envs
    
    def load_table(
        self,
        table_name: str,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Load a table (lazy loading with caching).
        
        Args:
            table_name: Name of table to load
            force_reload: Whether to reload even if cached
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If table not found
        """
        # Check cache
        if not force_reload and table_name in self._tables:
            return self._tables[table_name]
        
        # Load from disk
        table_path = self._get_table_path(table_name)
        
        if not table_path.exists():
            raise ValueError(
                f"Table '{table_name}' not found for source '{self.name}' "
                f"at path: {table_path}"
            )
        
        # Load based on file format
        if table_path.suffix == '.fst':
            from .fst_reader import read_fst
            df = read_fst(str(table_path))
        elif table_path.suffix == '.csv':
            df = pd.read_csv(table_path)
        elif table_path.suffix == '.parquet':
            df = pd.read_parquet(table_path)
        else:
            raise ValueError(f"Unsupported file format: {table_path.suffix}")
        
        # Cache
        self._tables[table_name] = df
        
        return df
    
    def unload_table(self, table_name: str) -> None:
        """Unload a table from memory.
        
        Args:
            table_name: Name of table to unload
        """
        if table_name in self._tables:
            del self._tables[table_name]
    
    def unload_all(self) -> None:
        """Unload all tables from memory."""
        self._tables.clear()
    
    def list_tables(self) -> List[str]:
        """List available tables for this source.
        
        Returns:
            List of table names
        """
        if self.config and hasattr(self.config, 'tables'):
            return list(self.config.tables.keys())
        
        # Fallback: scan data directory
        if self.data_dir and self.data_dir.exists():
            tables = []
            for file_path in self.data_dir.glob('*'):
                if file_path.suffix in ['.fst', '.csv', '.parquet']:
                    tables.append(file_path.stem)
            return sorted(tables)
        
        return []
    
    def list_loaded(self) -> List[str]:
        """List currently loaded tables.
        
        Returns:
            List of loaded table names
        """
        return list(self._tables.keys())
    
    def _get_table_path(self, table_name: str) -> Path:
        """Get path to table file.
        
        Args:
            table_name: Table name
            
        Returns:
            Path to table file
        """
        if self.data_dir:
            base_dir = Path(self.data_dir)
        elif self.config:
            base_dir = Path(self.config.data_dir)
        else:
            from .resources import data_dir as default_data_dir
            base_dir = Path(default_data_dir()) / self.name
        
        # Try different file formats
        for suffix in ['.fst', '.csv', '.parquet']:
            table_path = base_dir / f"{table_name}{suffix}"
            if table_path.exists():
                return table_path
        
        # Default to .fst
        return base_dir / f"{table_name}.fst"
    
    def __repr__(self) -> str:
        """String representation."""
        loaded = len(self.list_loaded())
        available = len(self.list_tables())
        status = "attached" if self.is_attached() else "detached"
        
        return (
            f"<SrcEnv '{self.name}' ({status}): "
            f"{loaded}/{available} tables loaded>"
        )
    
    def __contains__(self, table_name: str) -> bool:
        """Check if table is available."""
        return table_name in self.list_tables()


# Global registry of attached environments
_attached_envs: Dict[str, SrcEnv] = {}


def attached_srcs() -> List[str]:
    """List attached source names (R ricu attached_srcs).
    
    Returns:
        List of currently attached source names
        
    Examples:
        >>> attach_src('mimic')
        >>> attach_src('eicu')
        >>> attached_srcs()
        ['mimic', 'eicu']
    """
    return list(_attached_envs.keys())


def get_src_env(name: str) -> Optional[SrcEnv]:
    """Get a source environment by name.
    
    Args:
        name: Source name
        
    Returns:
        SrcEnv if attached, None otherwise
    """
    return _attached_envs.get(name)


def attach_src(
    name: str,
    config: Optional[SrcConfig] = None,
    data_dir: Optional[Path] = None,
) -> SrcEnv:
    """Attach a source environment (R ricu attach_src).
    
    Args:
        name: Data source name
        config: Source configuration (loaded if None)
        data_dir: Data directory override
        
    Returns:
        Attached SrcEnv
        
    Examples:
        >>> env = attach_src('mimic')
        >>> patients = env.load_table('patients')
    """
    # Check if already attached
    if name in _attached_envs:
        return _attached_envs[name]
    
    # Create and attach new environment
    env = SrcEnv(name, config=config, data_dir=data_dir, auto_attach=True)
    return env


def detach_src(name: str) -> None:
    """Detach a source environment (R ricu detach_src).
    
    Args:
        name: Source name to detach
        
    Examples:
        >>> detach_src('mimic')
    """
    if name in _attached_envs:
        env = _attached_envs[name]
        env.detach()


def detach_all_srcs() -> None:
    """Detach all source environments."""
    for name in list(_attached_envs.keys()):
        detach_src(name)


def src_env_available(name: str) -> bool:
    """Check if a source environment is available.
    
    Args:
        name: Source name
        
    Returns:
        True if source can be attached
    """
    try:
        from .resources import src_config_paths
        paths = src_config_paths()
        
        # Check if config exists
        for path in paths:
            if Path(path).exists():
                import json
                with open(path) as f:
                    configs = json.load(f)
                    if name in configs:
                        return True
        
        return False
    except Exception:
        return False


def is_tbl_avail(tbl: str, env: Union[str, SrcEnv]) -> bool:
    """Check if table is available in source (R ricu is_tbl_avail).
    
    Args:
        tbl: Table name
        env: Source environment or name
        
    Returns:
        True if table is available
    """
    if isinstance(env, str):
        env = get_src_env(env)
    
    if env is None:
        return False
    
    return tbl in env


def src_tbl_avail(env: Union[str, SrcEnv], tbls: Optional[List[str]] = None) -> Dict[str, bool]:
    """Check availability of multiple tables (R ricu src_tbl_avail).
    
    Args:
        env: Source environment or name
        tbls: List of table names (None = all tables)
        
    Returns:
        Dict mapping table names to availability status
    """
    if isinstance(env, str):
        env = get_src_env(env)
    
    if env is None:
        return {}
    
    if tbls is None:
        tbls = env.list_tables()
    
    return {tbl: is_tbl_avail(tbl, env) for tbl in tbls}


def src_data_avail(src: Union[str, List[str], SrcEnv] = None) -> pd.DataFrame:
    """Get data availability report (R ricu src_data_avail).
    
    Args:
        src: Source name(s), SrcEnv, or None for all attached
        
    Returns:
        DataFrame with columns: name, available, tables, total
    """
    import pandas as pd
    
    # Handle different input types
    if src is None:
        src = attached_srcs()
    elif isinstance(src, str):
        src = [src]
    elif isinstance(src, SrcEnv):
        src = [src.name]
    
    results = []
    for s in src:
        env = get_src_env(s) if isinstance(s, str) else s
        
        if env is None:
            results.append({
                'name': s,
                'available': False,
                'tables': 0,
                'total': 0
            })
        else:
            tbl_status = src_tbl_avail(env)
            results.append({
                'name': env.name,
                'available': all(tbl_status.values()) if tbl_status else False,
                'tables': sum(tbl_status.values()),
                'total': len(tbl_status)
            })
    
    return pd.DataFrame(results)


def is_data_avail(src: Union[str, List[str]] = None) -> Dict[str, bool]:
    """Check if all data is available for sources (R ricu is_data_avail).
    
    Args:
        src: Source name(s) or None for all attached
        
    Returns:
        Dict mapping source names to availability status
    """
    df = src_data_avail(src)
    return dict(zip(df['name'], df['available']))


class DataEnv:
    """Data environment manager (R ricu data_env).
    
    Higher-level interface for managing multiple source environments
    and providing unified access to data.
    
    Examples:
        >>> env = DataEnv(['mimic', 'eicu'])
        >>> mimic_patients = env.load('mimic', 'patients')
        >>> eicu_patients = env.load('eicu', 'patient')
    """
    
    def __init__(self, sources: Optional[List[str]] = None):
        """Initialize data environment.
        
        Args:
            sources: List of source names to attach (None = all available)
        """
        self.sources: Dict[str, SrcEnv] = {}
        
        if sources:
            for src in sources:
                self.attach(src)
        else:
            # Attach all available sources
            self.attach_available()
    
    def attach(self, name: str, **kwargs) -> SrcEnv:
        """Attach a source.
        
        Args:
            name: Source name
            **kwargs: Additional arguments passed to attach_src
            
        Returns:
            Attached SrcEnv
        """
        env = attach_src(name, **kwargs)
        self.sources[name] = env
        return env
    
    def detach(self, name: str) -> None:
        """Detach a source.
        
        Args:
            name: Source name
        """
        if name in self.sources:
            detach_src(name)
            del self.sources[name]
    
    def attach_available(self) -> None:
        """Attach all available sources."""
        # Try common sources
        for src in ['mimic', 'mimic_demo', 'eicu', 'eicu_demo', 
                    'hirid', 'aumc', 'picdb']:
            if src_env_available(src):
                try:
                    self.attach(src)
                except Exception:
                    pass  # Skip unavailable sources
    
    def load(self, source: str, table: str, **kwargs) -> pd.DataFrame:
        """Load a table from a source.
        
        Args:
            source: Source name
            table: Table name
            **kwargs: Additional arguments passed to load_table
            
        Returns:
            Loaded DataFrame
        """
        if source not in self.sources:
            self.attach(source)
        
        return self.sources[source].load_table(table, **kwargs)
    
    def list_sources(self) -> List[str]:
        """List attached sources."""
        return list(self.sources.keys())
    
    def list_tables(self, source: str) -> List[str]:
        """List tables in a source.
        
        Args:
            source: Source name
            
        Returns:
            List of table names
        """
        if source not in self.sources:
            raise ValueError(f"Source '{source}' not attached")
        
        return self.sources[source].list_tables()
    
    def __repr__(self) -> str:
        """String representation."""
        n_sources = len(self.sources)
        sources = ', '.join(self.sources.keys())
        return f"<DataEnv with {n_sources} sources: {sources}>"


# Convenience functions

def new_src_env(name: str, **kwargs) -> SrcEnv:
    """Create a new source environment (R ricu new_src_env).
    
    Args:
        name: Source name
        **kwargs: Additional arguments passed to SrcEnv
        
    Returns:
        New SrcEnv instance
    """
    return SrcEnv(name, **kwargs)


def src_env_objects() -> Dict[str, SrcEnv]:
    """Get all attached source environment objects.
    
    Returns:
        Dictionary mapping source names to SrcEnv objects
    """
    return _attached_envs.copy()


def is_src_env(obj: Any) -> bool:
    """Check if object is a SrcEnv.
    
    Args:
        obj: Object to check
        
    Returns:
        True if obj is a SrcEnv
    """
    return isinstance(obj, SrcEnv)

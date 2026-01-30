"""PyRICU Web 应用组件模块。

将 app.py 中的功能模块化，提高代码可维护性。

模块列表:
- constants: 特征组定义和显示名称
- utils: 系统资源检测和并行配置
- data_validation: 数据库路径验证和状态检查
"""

from .constants import (
    CONCEPT_GROUPS_INTERNAL,
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_DISPLAY,
    get_concept_groups,
)

from .utils import (
    get_system_resources,
    get_optimal_parallel_config,
)

from .data_validation import (
    check_data_status,
    validate_database_path,
    find_database_path,
    get_database_info,
    REQUIRED_PARQUET_TABLES,
    REQUIRED_CSV_FILES,
    DATABASE_NAMES,
    DATABASE_PATTERNS,
)

__all__ = [
    # Constants
    'CONCEPT_GROUPS_INTERNAL',
    'CONCEPT_GROUP_NAMES', 
    'CONCEPT_GROUPS_DISPLAY',
    'get_concept_groups',
    # Utils
    'get_system_resources',
    'get_optimal_parallel_config',
    # Data validation
    'check_data_status',
    'validate_database_path',
    'find_database_path',
    'get_database_info',
    'REQUIRED_PARQUET_TABLES',
    'REQUIRED_CSV_FILES',
    'DATABASE_NAMES',
    'DATABASE_PATTERNS',
]

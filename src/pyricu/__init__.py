"""pyricu - Python utilities for working with ICU datasets.

This package adapts the ideas of the R ``ricu`` package to provide a Pythonic
interface for loading intensive care datasets, working with configuration
metadata, and resolving clinical concepts across heterogeneous sources.
"""

from .config import DataSourceConfig, DataSourceRegistry
from .concept import ConceptDictionary, ConceptResolver
from .datasource import FilterOp, FilterSpec, ICUDataSource, load_table
from .resources import load_data_sources, load_dictionary, package_path
from .table import (
    ICUTable, 
    IdTbl, 
    TsTbl, 
    WinTbl, 
    PvalTbl,
    rbind_tbl, 
    cbind_tbl, 
    merge_lst,
    is_id_tbl,
    is_ts_tbl,
    is_win_tbl,
    is_icu_tbl,
    has_time_cols,
    validate_tbl_structure,
    id_vars,
    index_var,
    dur_var,
    meta_vars,
    data_vars,
    upgrade_id as table_upgrade_id,
    downgrade_id,
    change_id,
    rbind_lst,
    rename_cols,
    rm_cols,
)

# 高层便捷API - 推荐使用的主API
try:
    from .api import (
        # 主API - 智能默认值
        load_concepts as _load_concepts_api,  # 先导入为内部名称
        load_concept,  # 别名
        # Easy API - 便捷函数
        load_sofa,
        load_sofa2,
        load_sepsis3,
        load_vitals,
        load_labs,
        # 工具函数
        list_available_concepts,
        list_available_sources,
        get_concept_info,
    )
    # 将新API作为主要的load_concepts
    load_concepts = _load_concepts_api
    _HAS_API = True
except ImportError as e:
    print(f"Warning: Failed to import api module: {e}")
    _HAS_API = False

# 快速启动 API - DEPRECATED（保留向后兼容）
try:
    from .quickstart import (
        ICUQuickLoader,
        get_patient_ids,
        # 向后兼容的别名
        MIMICQuickLoader,
        load_mimic_sofa,
        load_mimic_sepsis3,
        load_mimic_vitals,
        load_mimic_labs,
    )
    _HAS_QUICKSTART = True
except ImportError:
    _HAS_QUICKSTART = False

# 从 load_concepts 模块导入（保留向后兼容）
# 注意：这会覆盖上面的load_concepts，所以我们在最后重新设置
try:
    from .load_concepts import ConceptLoader
    from .load_concepts import load_concepts as _load_concepts_old
    _HAS_LOAD_CONCEPTS = True
except ImportError:
    _HAS_LOAD_CONCEPTS = False

# 确保新API优先
if _HAS_API:
    load_concepts = _load_concepts_api  # 使用新API
    # load_concept已经从api.py导入，是load_concepts的别名

# 增强API - 支持缓存和时间对齐
try:
    from .api_enhanced import (
        load_concept_cached,
        align_to_icu_admission,
        load_sofa_with_score,
    )
    _HAS_ENHANCED_API = True
except ImportError:
    _HAS_ENHANCED_API = False

# Optional imports with availability checks
try:
    from .assertions import (
        assert_that,
        is_string,
        is_flag,
        is_scalar,
        is_number,
        is_count,
        no_na,
        has_length,
        has_rows,
        has_cols,
        are_in,
        is_unique,
        is_sorted,
        validate_data_frame,
        validate_id_tbl,
        validate_ts_tbl,
        validate_win_tbl,
    )
    _HAS_ASSERTIONS = True
except ImportError:
    _HAS_ASSERTIONS = False

try:
    from .utils import (
        round_to,
        is_val,
        not_val,
        is_true,
        is_false,
        first_elem,
        last_elem,
        coalesce,
        concat,
        agg_or_na,
        mean_or_na,
        sum_or_na,
        min_or_na,
        max_or_na,
    )
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False
try:
    from .download import download_src, download_sources
    _HAS_DOWNLOAD = True
except ImportError:
    _HAS_DOWNLOAD = False

try:
    from .import_data import import_src, import_sources
    _HAS_IMPORT = True
except ImportError:
    _HAS_IMPORT = False

try:
    from .ts_utils import (
        change_interval,
        expand_intervals,
        expand,
        collapse,
        fill_gaps,
        replace_na,
        slide,
        slide_index,
        hop,
        has_gaps,
        is_regular,
        hours,
        minutes,
        mins,
        days,
        secs,
        weeks,
        slide_windows,
        stay_windows,
        locf as ts_locf,
        locb as ts_locb,
        calc_dur,
        remove_gaps,
        merge_ranges,
        group_measurements,
        create_intervals,
    )
    _HAS_TS_UTILS = True
except ImportError:
    _HAS_TS_UTILS = False

try:
    from .data_utils import (
        add_column,
        aggregate_table,
        change_id_type,
        drop_columns,
        filter_table,
        merge_tables,
        pivot_table,
        rename_columns,
        select_columns,
        sort_table,
        stay_windows as stay_windows_utils,
        id_windows,
        id_origin,
        upgrade_id,
    )
    _HAS_DATA_UTILS = True
except ImportError:
    _HAS_DATA_UTILS = False

try:
    from .data_env import (
        SrcEnv,
        DataEnv,
        attached_srcs,
        get_src_env,
        attach_src,
        detach_src,
        detach_all_srcs,
        src_env_available,
        new_src_env,
        src_env_objects,
        is_src_env,
    )
    _HAS_DATA_ENV = True
except ImportError:
    _HAS_DATA_ENV = False

try:
    from .callbacks import (
        sofa_score,
        sofa_resp,
        sofa_coag,
        sofa_liver,
        sofa_cardio,
        sofa_cns,
        sofa_renal,
        sirs_score,
        qsofa_score,
        apache_ii_score,
        news_score,
        mews_score,
        pafi,
        safi,
    )
    _HAS_CALLBACKS = True
except ImportError:
    _HAS_CALLBACKS = False

try:
    from .sepsis import (
        sep3,
        susp_inf,
        delta_cummin,
        delta_start,
        delta_min,
    )
    _HAS_SEPSIS = True
except ImportError:
    _HAS_SEPSIS = False

# SOFA-2 & Sepsis-2
try:
    from .sofa2 import (
        sofa2_score,
        sofa2_resp,
        sofa2_coag,
        sofa2_liver,
        sofa2_cardio,
        sofa2_cns,
        sofa2_renal,
    )
    from .sepsis2 import (
        sep2,
    )
    _HAS_SOFA2 = True
except ImportError:
    _HAS_SOFA2 = False

try:
    from .export import (
        write_psv,
        read_psv,
        export_wide_format,
        export_long_format,
        export_summary,
        export_cohort_info,
        export_parquet,
        export_feather,
        export_json,
        export_data,
        data_quality_report,
    )
    _HAS_EXPORT = True
except ImportError:
    _HAS_EXPORT = False

try:
    from .file_utils import (
        data_dir,
        src_data_dir,
        ensure_dirs,
        is_dir,
        dir_exists,
        dir_create,
        file_size_format,
        file_copy_safe,
        config_paths,
        get_config,
        set_config,
        read_json,
        write_json,
        auto_attach_srcs,
        file_exists,
    )
    _HAS_FILE_UTILS = True
except ImportError:
    _HAS_FILE_UTILS = False

try:
    from .cli_utils import (
        is_interactive,
        progress_init,
        progress_tick,
        with_progress,
        msg_ricu,
        warn_ricu,
        stop_ricu,
        msg_progress,
        fmt_msg,
        bullet,
        big_mark,
        quote_bt,
        enbraket,
        concat as cli_concat,
        prcnt,
        ask_yes_no,
        cli_rule,
    )
    _HAS_CLI_UTILS = True
except ImportError:
    _HAS_CLI_UTILS = False

try:
    from .callback_utils import (
        transform_fun,
        binary_op,
        comp_na,
        set_val,
        apply_map,
        convert_unit,
        combine_callbacks,
        fwd_concept,
        ts_to_win_tbl,
        locf,
        locb,
        aggregate_fun,
        eicu_age,
        mimic_age,
        percent_as_numeric,
        distribute_amount,
        dex_to_10,
        mimv_rate,
        grp_mount_to_rate,
        grp_amount_to_rate,
        padded_capped_diff,
    )
    _HAS_CALLBACK_UTILS = True
except ImportError:
    _HAS_CALLBACK_UTILS = False

try:
    from .unit_conversion import (
        UnitConverter,
        convert_unit as convert_unit_value,
        celsius_to_fahrenheit,
        fahrenheit_to_celsius,
        glucose_mg_to_mmol,
        glucose_mmol_to_mg,
        creatinine_mg_to_umol,
        creatinine_umol_to_mg,
    )
    _HAS_UNIT_CONV = True
except ImportError:
    _HAS_UNIT_CONV = False

try:
    from .scores import (
        sirs_score as sirs_score_func,
        qsofa_score as qsofa_score_func,
        news_score as news_score_func,
        mews_score as mews_score_func,
    )
    _HAS_SCORES = True
except ImportError:
    _HAS_SCORES = False

try:
    from .data_quality import (
        DataQualityValidator,
        validate_data_quality,
        print_quality_summary,
    )
    _HAS_DATA_QUALITY = True
except ImportError:
    _HAS_DATA_QUALITY = False

# 数据源工具
try:
    from .src_utils import (
        src_name,
        src_prefix,
        src_extra_cfg,
        src_data_avail,
        src_tbl_avail,
        is_data_avail,
        is_tbl_avail,
        is_src_tbl,
    )
    _HAS_SRC_UTILS = True
except ImportError:
    _HAS_SRC_UTILS = False

# 表元数据访问器
try:
    from .table_meta import (
        id_var,
        id_col,
        index_col,
        dur_col,
        dur_unit,
        data_var,
        data_col,
        time_unit,
        time_step,
        interval as table_interval,
        id_var_opts,
        default_vars,
    )
    _HAS_TABLE_META = True
except ImportError:
    _HAS_TABLE_META = False

# 底层数据加载
try:
    from .data_load import (
        load_src,
        load_difftime,
        load_id,
        load_ts,
        load_win,
    )
    _HAS_DATA_LOAD = True
except ImportError:
    _HAS_DATA_LOAD = False

# 类型转换
try:
    from .table_convert import (
        reclass_tbl,
        unclass_tbl,
        as_col_cfg,
        as_id_cfg,
        as_src_cfg,
        as_tbl_cfg,
        as_src_tbl,
        as_ptype,
    )
    _HAS_TABLE_CONVERT = True
except ImportError:
    _HAS_TABLE_CONVERT = False

# 概念管理
try:
    from .concept_utils import (
        add_concept,
        concept_availability,
        explain_dictionary,
        subset_src,
    )
    _HAS_CONCEPT_UTILS = True
except ImportError:
    _HAS_CONCEPT_UTILS = False

# 临床工具
try:
    from .clinical_utils import (
        avpu,
        bmi,
        gcs,
        norepi_equiv,
        supp_o2,
        urine24,
        vaso60,
        vaso_ind,
        vent_ind,
    )
    _HAS_CLINICAL_UTILS = True
except ImportError:
    _HAS_CLINICAL_UTILS = False

# 数据工具
try:
    from .data_tools import (
        unmerge,
        rm_na,
        change_dur_unit,
        has_no_gaps,
        load_src_cfg,
    )
    _HAS_DATA_TOOLS = True
except ImportError:
    _HAS_DATA_TOOLS = False

# ID映射系统
try:
    from .id_mapping import (
        id_map,
        id_map_helper,
        id_origin,
        id_orig_helper,
        id_windows,
        id_win_helper,
        as_src_env as id_as_src_env,
    )
    _HAS_ID_MAPPING = True
except ImportError:
    _HAS_ID_MAPPING = False

# 回调系统
try:
    from .callback_system import (
        do_callback,
        do_itm_load,
        set_callback,
        prepare_query,
        add_weight,
        get_target,
        set_target,
        get_itm_var,
    )
    _HAS_CALLBACK_SYSTEM = True
except ImportError:
    _HAS_CALLBACK_SYSTEM = False

# 概念构建
try:
    from .concept_builder import (
        new_concept,
        new_item,
        new_itm,
        new_cncpt,
        init_cncpt,
        init_itm,
        is_cncpt,
        is_concept,
        is_item,
        is_itm,
        new_src_tbl,
    )
    _HAS_CONCEPT_BUILDER = True
except ImportError:
    _HAS_CONCEPT_BUILDER = False

__all__ = [
    # === 推荐使用的API ===
    # 主API（智能默认值，完全灵活）
    "load_concepts",
    "load_concept",  # 别名
    # Easy API（预定义便捷函数）
    "load_sofa",
    "load_sofa2",
    "load_sepsis3",
    "load_vitals",
    "load_labs",
    # 工具函数
    "list_available_concepts",
    "list_available_sources",
    "get_concept_info",
    
    # === 核心类 ===
    "ConceptDictionary",
    "ConceptResolver",
    "DataSourceConfig",
    "DataSourceRegistry",
    "FilterOp",
    "FilterSpec",
    "ICUDataSource",
    "ICUTable",
    "IdTbl",
    "TsTbl",
    "WinTbl",
    "PvalTbl",
    
    # === 表操作 ===
    "rbind_tbl",
    "cbind_tbl",
    "merge_lst",
    "is_id_tbl",
    "is_ts_tbl",
    "is_win_tbl",
    "is_icu_tbl",
    "has_time_cols",
    "validate_tbl_structure",
    "id_vars",
    "index_var",
    "dur_var",
    "meta_vars",
    "data_vars",
    "table_upgrade_id",
    "downgrade_id",
    "change_id",
    "rbind_lst",
    "rename_cols",
    "rm_cols",
    
    # === 资源加载 ===
    "load_table",
    "load_data_sources",
    "load_dictionary",
    "package_path",
]

# Add optional exports
if _HAS_API:
    # API已在上面添加到__all__中
    pass

if _HAS_LOAD_CONCEPTS:
    __all__.extend([
        "ConceptLoader",  # 只导出类，不导出load_concepts函数
    ])

if _HAS_ENHANCED_API:
    __all__.extend([
        "load_concept_cached",
        "align_to_icu_admission",
        "load_sofa_with_score",
    ])

if _HAS_QUICKSTART:
    # Deprecated - 保留向后兼容
    __all__.extend([
        "ICUQuickLoader",  # DEPRECATED
        "get_patient_ids",
        # 向后兼容的别名
        "MIMICQuickLoader",  # DEPRECATED
        "load_mimic_sofa",  # DEPRECATED
        "load_mimic_sepsis3",  # DEPRECATED
        "load_mimic_vitals",  # DEPRECATED
        "load_mimic_labs",  # DEPRECATED
    ])

if _HAS_ASSERTIONS:
    __all__.extend([
        "assert_that",
        "is_string",
        "is_flag",
        "is_scalar",
        "is_number",
        "is_count",
        "no_na",
        "has_length",
        "has_rows",
        "has_cols",
        "are_in",
        "is_unique",
        "is_sorted",
        "validate_data_frame",
        "validate_id_tbl",
        "validate_ts_tbl",
        "validate_win_tbl",
    ])

if _HAS_UTILS:
    __all__.extend([
        "round_to",
        "is_val",
        "not_val",
        "is_true",
        "is_false",
        "first_elem",
        "last_elem",
        "coalesce",
        "concat",
        "agg_or_na",
        "mean_or_na",
        "sum_or_na",
        "min_or_na",
        "max_or_na",
    ])
if _HAS_DOWNLOAD:
    __all__.extend(["download_src", "download_sources"])

if _HAS_IMPORT:
    __all__.extend(["import_src", "import_sources"])

if _HAS_TS_UTILS:
    __all__.extend([
        "change_interval",
        "expand_intervals",
        "expand",
        "collapse",
        "fill_gaps",
        "replace_na",
        "slide",
        "slide_index",
        "hop",
        "has_gaps",
        "is_regular",
        "hours",
        "minutes",
        "mins",
        "days",
        "secs",
        "weeks",
        "slide_windows",
        "stay_windows",
        "ts_locf",
        "ts_locb",
        "calc_dur",
        "remove_gaps",
        "merge_ranges",
        "group_measurements",
        "create_intervals",
    ])

if _HAS_DATA_UTILS:
    __all__.extend([
        "add_column",
        "aggregate_table",
        "change_id_type",
        "drop_columns",
        "filter_table",
        "merge_tables",
        "pivot_table",
        "rename_columns",
        "select_columns",
        "sort_table",
        "stay_windows_utils",
        "id_windows",
        "id_origin",
        "upgrade_id",
    ])

if _HAS_DATA_ENV:
    __all__.extend([
        "SrcEnv",
        "DataEnv",
        "attached_srcs",
        "get_src_env",
        "attach_src",
        "detach_src",
        "detach_all_srcs",
        "src_env_available",
        "new_src_env",
        "src_env_objects",
        "is_src_env",
    ])

if _HAS_CALLBACKS:
    __all__.extend([
        "sofa_score",
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cardio",
        "sofa_cns",
        "sofa_renal",
        "sirs_score",
        "qsofa_score",
        "apache_ii_score",
        "news_score",
        "mews_score",
        "pafi",
        "safi",
    ])

if _HAS_SEPSIS:
    __all__.extend([
        "sep3",
        "susp_inf",
        "delta_cummin",
        "delta_start",
        "delta_min",
    ])

if _HAS_SOFA2:
    __all__.extend([
        "sofa2_score",
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
        "sep2",
    ])

if _HAS_EXPORT:
    __all__.extend([
        "write_psv",
        "read_psv",
        "export_wide_format",
        "export_long_format",
        "export_summary",
        "export_cohort_info",
        "export_parquet",
        "export_feather",
        "export_json",
        "export_data",
        "data_quality_report",
    ])

if _HAS_FILE_UTILS:
    __all__.extend([
        "data_dir",
        "src_data_dir",
        "ensure_dirs",
        "is_dir",
        "dir_exists",
        "dir_create",
        "file_size_format",
        "file_copy_safe",
        "config_paths",
        "get_config",
        "set_config",
        "read_json",
        "write_json",
        "auto_attach_srcs",
        "file_exists",
    ])

if _HAS_CLI_UTILS:
    __all__.extend([
        "is_interactive",
        "progress_init",
        "progress_tick",
        "with_progress",
        "msg_ricu",
        "warn_ricu",
        "stop_ricu",
        "msg_progress",
        "fmt_msg",
        "bullet",
        "big_mark",
        "quote_bt",
        "enbraket",
        "prcnt",
        "ask_yes_no",
        "cli_rule",
    ])

if _HAS_CALLBACK_UTILS:
    __all__.extend([
        "transform_fun",
        "binary_op",
        "comp_na",
        "set_val",
        "apply_map",
        "convert_unit",
        "combine_callbacks",
        "fwd_concept",
        "ts_to_win_tbl",
        "locf",
        "locb",
        "aggregate_fun",
        "eicu_age",
        "mimic_age",
        "percent_as_numeric",
        "distribute_amount",
        "dex_to_10",
        "mimv_rate",
        "grp_mount_to_rate",
        "grp_amount_to_rate",
        "padded_capped_diff",
    ])

if _HAS_UNIT_CONV:
    __all__.extend([
        "UnitConverter",
        "convert_unit_value",
        "celsius_to_fahrenheit",
        "fahrenheit_to_celsius",
        "glucose_mg_to_mmol",
        "glucose_mmol_to_mg",
        "creatinine_mg_to_umol",
        "creatinine_umol_to_mg",
    ])

if _HAS_SCORES:
    __all__.extend([
        "sirs_score_func",
        "qsofa_score_func",
        "news_score_func",
        "mews_score_func",
    ])

if _HAS_DATA_QUALITY:
    __all__.extend([
        "DataQualityValidator",
        "validate_data_quality",
        "print_quality_summary",
    ])

if _HAS_SRC_UTILS:
    __all__.extend([
        "src_name",
        "src_prefix",
        "src_extra_cfg",
        "src_data_avail",
        "src_tbl_avail",
        "is_data_avail",
        "is_tbl_avail",
        "is_src_tbl",
    ])

if _HAS_TABLE_META:
    __all__.extend([
        "id_var",
        "id_col",
        "index_col",
        "dur_col",
        "dur_unit",
        "data_var",
        "data_col",
        "time_unit",
        "time_step",
        "table_interval",
        "id_var_opts",
        "default_vars",
    ])

if _HAS_DATA_LOAD:
    __all__.extend([
        "load_src",
        "load_difftime",
        "load_id",
        "load_ts",
        "load_win",
    ])

if _HAS_TABLE_CONVERT:
    __all__.extend([
        "reclass_tbl",
        "unclass_tbl",
        "as_col_cfg",
        "as_id_cfg",
        "as_src_cfg",
        "as_tbl_cfg",
        "as_src_tbl",
        "as_ptype",
    ])

if _HAS_CONCEPT_UTILS:
    __all__.extend([
        "add_concept",
        "concept_availability",
        "explain_dictionary",
        "subset_src",
    ])

if _HAS_CLINICAL_UTILS:
    __all__.extend([
        "avpu",
        "bmi",
        "gcs",
        "norepi_equiv",
        "supp_o2",
        "urine24",
        "vaso60",
        "vaso_ind",
        "vent_ind",
    ])

if _HAS_DATA_TOOLS:
    __all__.extend([
        "unmerge",
        "rm_na",
        "change_dur_unit",
        "has_no_gaps",
        "load_src_cfg",
    ])

if _HAS_ID_MAPPING:
    __all__.extend([
        "id_map",
        "id_map_helper",
        "id_origin",
        "id_orig_helper",
        "id_windows",
        "id_win_helper",
        "id_as_src_env",
    ])

if _HAS_CALLBACK_SYSTEM:
    __all__.extend([
        "do_callback",
        "do_itm_load",
        "set_callback",
        "prepare_query",
        "add_weight",
        "get_target",
        "set_target",
        "get_itm_var",
    ])

if _HAS_CONCEPT_BUILDER:
    __all__.extend([
        "new_concept",
        "new_item",
        "new_itm",
        "new_cncpt",
        "init_cncpt",
        "init_itm",
        "is_cncpt",
        "is_concept",
        "is_item",
        "is_itm",
        "new_src_tbl",
    ])

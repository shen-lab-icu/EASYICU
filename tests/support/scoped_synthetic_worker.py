"""Worker for the synthetic scoped-producer integration test only."""
import sys

from easyicu.webserver.dataio import make_export_runner


class Job:
    def emit(self, payload):
        pass


source, destination = sys.argv[1:]
make_export_runner(
    data_path=source, database="sic",
    modules=["outcome", "blood_gas", "sic_case_landmark"],
    concepts={"outcome": ["los_icu"], "blood_gas": ["lact"],
              "sic_case_landmark": ["care_unit_to_case_closure_days",
                                    "death_time_during_case_hours", "death_during_case"]},
    export_format="parquet", out_dir=destination, include_feature_definitions=False,
)(Job())

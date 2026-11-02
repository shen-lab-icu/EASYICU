# AUMC `drugitems.fluidin` volume source correction

The [AmsterdamUMCdb legacy `drugitems` field definition](https://github.com/AmsterdamUMC/AmsterdamUMCdb/wiki/drugitems) defines `fluidin` as administered fluid volume in **mL, including the solution**. It defines `doseunit` separately as the unit of the prescribed medication dose. Consequently, a row with `fluidin=100` and `doseunit=mg` contributes 100 mL of fluid, not an unknown volume.

Before this correction, `total_input_ml` selected `fluidin` but inherited the table default `unit_var=doseunit`. `distribute_volume_hourly` then passed the medication dose unit to `normalize_volume_to_ml`, which discarded rows whose dose units were not volume units. The source also selected the maximum of `fluidin` and `solutionadministered`, even though the field definition says `fluidin` already includes the solution.

The candidate source now declares `value_unit=mL` for the AUMC `fluidin` source and uses `fluidin` alone. The generic interval-volume allocator and explicit-unit conversion for MIMIC sources are unchanged. A focused regression exercises `mg` and `ml` dose-unit rows and the solution-inclusive volume rule.

This changes source-code semantics only. It does **not** alter any sealed EasyICU release or retrospectively certify analyses using `total_input_ml` or its derived `fluid_balance` fields. Those consumers require a separately named candidate extraction and numerical comparison before a release decision.

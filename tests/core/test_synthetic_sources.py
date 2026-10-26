"""Converted synthetic sources preserve each database's physical contract."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_sources_are_deterministic_and_have_native_identifiers(tmp_path, database):
    from easyicu.synthetic_sources import generate_source, verify_source

    first = generate_source(tmp_path / "first", database, stays=12)
    second = generate_source(tmp_path / "second", database, stays=12)
    assert first == second
    assert first["source_kind"] == "synthetic"
    assert first["contains_patient_data"] is False
    assert verify_source(tmp_path / "first") == first
    files = sorted((tmp_path / "first").rglob("*.parquet"))
    assert files
    for path in files:
        assert path.read_bytes() == (tmp_path / "second" / path.relative_to(tmp_path / "first")).read_bytes()
    name, identifier = {
        "hirid": ("general", "patientid"),
        "aumc": ("admissions", "admissionid"),
        "sic": ("cases", "CaseID"),
    }[database]
    frame = pd.read_parquet(tmp_path / "first" / f"{name}.parquet")
    assert len(frame) == frame[identifier].nunique() == 12
    assert pd.api.types.is_integer_dtype(frame[identifier])
    if database == "hirid":
        assert set(frame) == {"patientid", "admissiontime", "sex", "age", "discharge_status"}
        assert frame.age.mod(5).eq(0).all() and frame.age.le(90).all()
        observations = pd.read_parquet(tmp_path / "first/observations/1.parquet")
        weight = observations.loc[observations.variableid.eq(10000400), "value"].dropna()
        assert not weight.empty and weight.mod(5).eq(0).all()
        assert (tmp_path / "first/observations/1.parquet").is_file()
    elif database == "aumc":
        assert frame.patientid.nunique() < len(frame)
        assert set(frame.agegroup.dropna()) <= {"18-39", "40-49", "50-59", "60-69", "70-79", "80+"}
        assert frame.admittedat.ne(0).all()
        assert (frame.dischargedat - frame.admittedat).min() >= 48 * 3_600_000
        assert frame.dateofdeath.iloc[0] == frame.dischargedat.iloc[0]
        assert frame.dateofdeath.iloc[1] == frame.dischargedat.iloc[1] + 240 * 3_600_000
    else:
        assert frame.PatientID.nunique() < len(frame)
        assert frame.ICUOffset.gt(0).all()
        assert (frame.TimeOfStay - frame.ICUOffset).min() >= 48 * 3600
        assert frame.OffsetOfDeath.isna().any()
        assert set(frame.HospitalDischargeType) == {2026, 2028}
        assert ((frame.HospitalDischargeType != 2028) & frame.OffsetOfDeath.notna()).any()
        assert frame.OffsetOfDeath.iloc[0] == frame.TimeOfStay.iloc[0]
        assert frame.OffsetOfDeath.iloc[1] == frame.TimeOfStay.iloc[1] + 240 * 3600
        assert frame.WeightOnAdmission.between(45000, 125000).all()
        assert frame.WeightOnAdmission.mod(5000).eq(0).all()
        assert frame.AgeOnAdmission.mod(5).eq(0).all() and frame.AgeOnAdmission.le(90).all()


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_default_sources_have_four_hundred_stays(tmp_path, database):
    from easyicu.synthetic_sources import generate_source

    manifest = generate_source(tmp_path / database, database)
    assert manifest["stays"] == 400
    assert manifest["seed"] == 20261026
    assert "not clinical findings" in manifest["notice"]


@pytest.mark.parametrize("defect", ["value", "extra", "missing", "symlink", "manifest", "forged_hash", "readme", "same_size"])
def test_synthetic_claim_needs_generator_content_proof(tmp_path, defect):
    from easyicu.synthetic_sources import MANIFEST, generate_source, verify_source

    root = tmp_path / "source"
    generate_source(root, "sic", stays=12)
    member = root / "cases.parquet"
    if defect in {"value", "forged_hash"}:
        frame = pd.read_parquet(member)
        frame.loc[0, "AgeOnAdmission"] += 1
        frame.to_parquet(member, index=False)
        if defect == "forged_hash":
            import hashlib

            manifest = json.loads((root / MANIFEST).read_text())
            manifest["files"]["cases.parquet"]["sha256"] = hashlib.sha256(member.read_bytes()).hexdigest()
            manifest["files"]["cases.parquet"]["size_bytes"] = member.stat().st_size
            (root / MANIFEST).write_text(json.dumps(manifest))
    elif defect == "extra":
        (root / "extra.parquet").write_bytes(member.read_bytes())
    elif defect == "missing":
        member.unlink()
    elif defect == "symlink":
        moved = tmp_path / "outside.parquet"
        member.rename(moved)
        member.symlink_to(moved)
    elif defect == "readme":
        (root / "README.md").write_text("Clinical report")
    elif defect == "same_size":
        content = bytearray(member.read_bytes())
        content[10] ^= 1
        member.write_bytes(content)
    else:
        manifest = json.loads((root / MANIFEST).read_text())
        manifest["database"] = "hirid"
        (root / MANIFEST).write_text(json.dumps(manifest))
    assert verify_source(root) is None


def test_generator_refuses_to_overwrite_existing_data(tmp_path):
    from easyicu.synthetic_sources import generate_source

    (tmp_path / "keep.txt").write_text("existing fixture")
    with pytest.raises(FileExistsError):
        generate_source(tmp_path, "hirid")
    assert (tmp_path / "keep.txt").read_text() == "existing fixture"


def test_converted_schema_uses_physical_sic_column_names(tmp_path):
    from easyicu.synthetic_sources import generate_source

    generate_source(tmp_path / "sic", "sic", stays=12)
    config = json.loads((Path(__file__).parents[2] / "src/easyicu/data/data-sources.json").read_text())
    tables = next(source for source in config if source["name"] == "sic")["tables"]
    for name, table in tables.items():
        assert pq.read_schema(tmp_path / "sic" / f"{name}.parquet").names == [col["name"] for col in table["cols"].values()]


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_native_clocks_types_and_quality_sentinels(tmp_path, database):
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / database
    generate_source(root, database, stays=12)
    table, itemcol, item, idcol, clock = {
        "hirid": ("observations", "variableid", 110, "patientid", "datetime"),
        "aumc": ("numericitems", "itemid", 6642, "admissionid", "measuredat"),
        "sic": ("data_float_h.parquet", "DataID", 703, "CaseID", "Offset"),
    }[database]
    frame = pd.read_parquet(root / table)
    rows = frame.loc[frame[itemcol].eq(item)]
    value = "Val" if database == "sic" else "value"
    if database == "hirid":
        value = "value"
        assert pa.types.is_timestamp(pq.read_schema(root / "observations/1.parquet").field(clock).type)
        assert pq.read_schema(root / "general.parquet").field("admissiontime").type == pa.string()
        assert (root / "ordinal_vars_ref.parquet").is_file()
        assert (root / "hirid_variable_reference.parquet").is_file()
    elif database == "aumc":
        assert pd.api.types.is_float_dtype(rows[clock])
    else:
        assert pd.api.types.is_integer_dtype(rows[clock])
    assert rows[value].isna().any()
    assert rows.loc[rows[idcol].eq(1), value].max() == 400
    assert rows.loc[rows[idcol].eq(4), value].isna().any()
    by_stay = rows.loc[rows[idcol].eq(2)].sort_values(clock)
    delta = by_stay[clock].diff().dropna().iloc[0]
    assert delta == {"hirid": pd.Timedelta(hours=1), "aumc": 3600000, "sic": 3600}[database]
    if database == "aumc":
        assert pq.read_schema(root / "drugitems.parquet").field("doseunitid").type == pa.string()
        from easyicu.io.data_converter import PARTITIONING_CONFIG
        for name in ("numericitems", "listitems"):
            breaks = PARTITIONING_CONFIG["aumc"][name]["breaks"]
            assert len(list((root / name).glob("*.parquet"))) == len(breaks) + 1
            for number in range(len(breaks) + 1):
                shard = pd.read_parquet(root / name / f"{number + 1}.parquet")
                if number:
                    assert shard.itemid.gt(breaks[number - 1]).all()
                if number < len(breaks):
                    assert shard.itemid.le(breaks[number]).all()
                assert list(zip(shard.itemid, shard.admissionid)) == sorted(zip(shard.itemid, shard.admissionid))


def test_core_measurements_share_a_plausible_invented_severity_distribution():
    from easyicu.synthetic_source_values import population, numeric_values

    pop = population(400)
    dictionary = json.loads((Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json").read_text())
    index = np.arange(400)
    rng = np.random.default_rng(7)
    values = {name: numeric_values(name, dictionary[name], pop, index, np.zeros(400), rng)[1:]
              for name in ("map", "lact", "crea", "urine")}
    assert np.corrcoef(values["map"], values["lact"])[0, 1] < -.8
    assert np.corrcoef(values["crea"], values["lact"])[0, 1] > .8
    assert np.corrcoef(values["urine"], values["lact"])[0, 1] < -.8
    assert .05 < pop["dead"].mean() < .4
    assert pop["length"][pop["dead"]].mean() > pop["length"][~pop["dead"]].mean()
    for _, group in pd.DataFrame(pop).groupby("patient"):
        assert group.age.nunique() == group.male.nunique() == group.weight.nunique() == 1
        assert not group.dead.iloc[:-1].any()


@pytest.mark.parametrize("database,stays", [("miiv", 12), ("hirid", 0), ("sic", 601), ("aumc", True)])
def test_generator_rejects_unsupported_parameters(tmp_path, database, stays):
    from easyicu.synthetic_sources import generate_source

    with pytest.raises(ValueError):
        generate_source(tmp_path / "bad", database, stays=stays)
    assert not (tmp_path / "bad").exists()

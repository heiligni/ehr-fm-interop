"""This script is mostly copied from https://github.com/Medical-Event-Data-Standard/meds_etl/blob/0.1.2/src/meds_etl/mimic/__init__.py
For this work we extract additional data that represents the reference values for the lab and other tests.
"""

import argparse
import json
import os
import shutil
import subprocess
from importlib.resources import files
from typing import Iterable

import jsonschema
import meds
import polars as pl

import meds_etl
import meds_etl.flat

MIMIC_VERSION = "2.2"

MIMIC_TIME_FORMATS: Iterable[str] = ("%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d")


def add_dot(code, position):
    return (
        pl.when(code.str.len_chars() > position)
        .then(code.str.slice(0, position) + "." + code.str.slice(position))
        .otherwise(code)
    )


def normalize_icd_diagnosis(icd_version, icd_code):
    icd9_code = (
        pl.when(icd_code.str.starts_with("E"))
        .then(add_dot(icd_code, 4))
        .otherwise(add_dot(icd_code, 3))
    )

    icd10_code = add_dot(icd_code, 3)

    return (
        pl.when(icd_version == "9")
        .then("ICD9CM/" + icd9_code)
        .otherwise("ICD10CM/" + icd10_code)
    )


def normalize_icd_procedure(icd_version, icd_code):
    icd9_code = add_dot(icd_code, 2)
    icd10_code = icd_code

    return (
        pl.when(icd_version == "9")
        .then("ICD9Proc/" + icd9_code)
        .otherwise("ICD10PCS/" + icd10_code)
    )


def main():
    parser = argparse.ArgumentParser(
        prog="meds_etl_mimic", description="Performs an ETL from MIMIC_IV to MEDS"
    )
    parser.add_argument("src_mimic", type=str)
    parser.add_argument("destination", type=str)
    parser.add_argument("--num_shards", type=int, default=100)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--backend", type=str, default="polars")
    args = parser.parse_args()

    if not os.path.exists(args.src_mimic):
        raise ValueError(
            f'The source MIMIC_IV folder ("{args.src_mimic}") does not seem to exist?'
        )

    src_mimic_version = os.path.join(args.src_mimic, MIMIC_VERSION)

    if not os.path.exists(src_mimic_version):
        raise ValueError(
            f'The source MIMIC_IV folder does not contain a version 2.2 subfolder ("{src_mimic_version}")'
        )

    os.makedirs(args.destination)

    all_tables = {
        # Explicitly ignore icustays as it is redunant with hosp/transfers
        "icu/icustays": None,
        # Ignore these tables as they don't generate events
        "icu/caregiver": None,
        "icu/d_items": None,
        "icu/datetimeevents": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid"),
            "time": pl.col("storetime"),
            "value": pl.col("value"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("caregiver_id"),
            },
        },
        "icu/inputevents": {
            "code": "MIMIC_IV_ITEM/"
            + pl.col("itemid")
            + "/"
            + pl.col("ordercategorydescription"),
            "time": pl.col("storetime"),
            "value": pl.col("rate"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("caregiver_id"),
                "start": pl.col("starttime"),
                "end": pl.col("endtime"),
                "unit": pl.col("rateuom"),
            },
        },
        "icu/chartevents": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid"),
            "time": pl.coalesce(pl.col("storetime"), pl.col("charttime")),
            "value": pl.col("value"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("caregiver_id"),
                "unit": pl.col("valueuom"),
            },
        },
        "icu/procedureevents": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid"),
            "time": pl.col("storetime"),
            "value": pl.col("value"),
            "metadata": {
                "location": pl.col("location"),
                "location_category": pl.col("locationcategory"),
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("caregiver_id"),
                "unit": pl.col("valueuom"),
                "start": pl.col("starttime"),
                "end": pl.col("endtime"),
            },
        },
        "icu/ingredientevents": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid"),
            "time": pl.col("storetime"),
            "value": pl.col("rate"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("caregiver_id"),
                "unit": pl.col("rateuom"),
                "start": pl.col("starttime"),
                "end": pl.col("endtime"),
            },
        },
        "icu/outputevents": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid"),
            "time": pl.col("storetime"),
            "value": pl.col("value"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("caregiver_id"),
                "unit": pl.col("valueuom"),
            },
        },
        "hosp/admissions": [
            {
                "code": "MIMIC_IV_Admission/" + pl.col("admission_type"),
                "time": pl.col("admittime"),
                "metadata": {
                    "visit_id": pl.col("hadm_id"),
                    "caregiver_id": pl.col("admit_provider_id"),
                    "end": pl.col("dischtime"),
                },
            },
            {
                "code": "MIMIC_IV_Admission_Location/" + pl.col("admission_location"),
                "time": pl.col("admittime"),
                "metadata": {
                    "visit_id": pl.col("hadm_id"),
                },
            },
            {
                "code": "MIMIC_IV_Discharge_Location/" + pl.col("discharge_location"),
                "time": pl.col("dischtime"),
                "metadata": {
                    "visit_id": pl.col("hadm_id"),
                },
                "possibly_null_code": True,
            },
            {
                "code": "MIMIC_IV_Insurance/" + pl.col("insurance"),
                "time": pl.col("admittime"),
                "metadata": {
                    "visit_id": pl.col("hadm_id"),
                },
            },
            {
                "code": "MIMIC_IV_Language/" + pl.col("language"),
                "time": pl.col("admittime"),
                "metadata": {
                    "visit_id": pl.col("hadm_id"),
                },
            },
            {
                "code": "MIMIC_IV_Marital_Status/" + pl.col("marital_status"),
                "time": pl.col("admittime"),
                "metadata": {
                    "visit_id": pl.col("hadm_id"),
                },
                "possibly_null_code": True,
            },
            {
                "code": "MIMIC_IV_Race/" + pl.col("race"),
                "time": pl.col("admittime"),
                "metadata": {
                    "visit_id": pl.col("hadm_id"),
                },
            },
        ],
        # No events are generated from these tables
        "hosp/d_hcpcs": None,
        "hosp/d_icd_diagnoses": None,
        "hosp/d_icd_procedures": None,
        "hosp/d_labitems": None,
        "hosp/diagnoses_icd": {
            "code": normalize_icd_diagnosis(pl.col("icd_version"), pl.col("icd_code")),
            "time": pl.col("dischtime"),
            "requires_admission_join": True,
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "seq_num": pl.col("seq_num"),
            },
        },
        "hosp/drgcodes": {
            "code": pl.col("drg_type") + "/" + pl.col("drg_code"),
            "time": pl.col("dischtime"),
            "requires_admission_join": True,
            "metadata": {
                "visit_id": pl.col("hadm_id"),
            },
        },
        # It's unclear how to best process these tables.
        # TODO: Fix these
        "hosp/emar": None,
        "hosp/emar_detail": None,
        "hosp/hcpcsevents": {
            "code": "HCPCS/" + pl.col("hcpcs_cd"),
            "time": pl.col("chartdate"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "seq_num": pl.col("seq_num"),
            },
        },
        "hosp/labevents": {
            "code": "MIMIC_IV_LABITEM/" + pl.col("itemid"),
            "time": pl.coalesce(pl.col("storetime"), pl.col("charttime")),
            "value": pl.col("value"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("order_provider_id"),
                "unit": pl.col("valueuom"),
                "priority": pl.col("priority"),
                "comments": pl.col("comments"),
                "ref_range_lower": pl.col("ref_range_lower"),
                "ref_range_upper": pl.col("ref_range_upper"),
            },
        },
        # This transformation is ignoring a ton of data within the table.
        # TODO: Improve this
        "hosp/microbiologyevents": {
            "code": "MIMIC_IV_MicrobiologyTest/" + pl.col("test_name"),
            "time": pl.coalesce(
                pl.col("storetime"),
                pl.col("storedate"),
                pl.col("charttime"),
                pl.col("chartdate"),
            ),
            "value": pl.col("org_name"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "caregiver_id": pl.col("order_provider_id"),
                "specimen_name": "MIMIC_IV_MicrobiologySpecimin/"
                + pl.col("spec_type_desc"),
                "ab_name": "MIMIC_IV_MicrobiologyAntiBiotic/" + pl.col("ab_name"),
                "dilution_text": pl.col("dilution_text"),
                "interpretation": pl.col("interpretation"),
                "comments": pl.col("comments"),
            },
        },
        "hosp/omr": {
            "code": "MIMIC_IV_OMR/" + pl.col("result_name"),
            "time": pl.col("chartdate"),
            "value": pl.col("result_value"),
        },
        "hosp/patients": [
            {
                # Birth
                "code": pl.lit(meds.birth_code),
                "time": pl.datetime(
                    pl.col("anchor_year").cast(int) - pl.col("anchor_age").cast(int),
                    1,
                    1,
                ).cast(str),
            },
            {
                "code": "MIMIC_IV_Gender/" + pl.col("gender"),
                "time": pl.datetime(
                    pl.col("anchor_year").cast(int) - pl.col("anchor_age").cast(int),
                    1,
                    1,
                ).cast(str),
            },
            {
                # Death
                "code": pl.lit(meds.death_code),
                "time": pl.col("dod"),
                "possibly_null_time": True,
            },
        ],
        # It's unclear how to process this table.
        # TODO: Fix this
        "hosp/pharmacy": None,
        # The POE table contains the same information as other tables, so not needed
        "hosp/poe": None,
        "hosp/poe_detail": None,
        "hosp/prescriptions": {
            "code": pl.coalesce(
                "NDC/" + pl.when(pl.col("ndc") != "0").then(pl.col("ndc")),
                "MIMIC_IV_Drug/" + pl.col("drug"),
            ),
            "time": pl.col("starttime"),
            "value": pl.col("dose_val_rx"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "route": pl.col("route"),
                "unit": pl.col("dose_unit_rx"),
            },
            "possibly_null_time": True,
        },
        "hosp/procedures_icd": {
            "code": normalize_icd_procedure(pl.col("icd_version"), pl.col("icd_code")),
            "time": pl.col("chartdate"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
                "seq_num": pl.col("seq_num"),
            },
        },
        # Doesn't generate events
        "hosp/provider": None,
        "hosp/services": {
            "code": "MIMIC_IV_Service/" + pl.col("curr_service"),
            "time": pl.col("transfertime"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
            },
        },
        "hosp/transfers": {
            "code": "MIMIC_IV_Transfer/" + pl.col("careunit"),
            "time": pl.col("intime"),
            "metadata": {
                "visit_id": pl.col("hadm_id"),
            },
            "possibly_null_code": True,
        },
    }

    admission_table = pl.read_csv(
        os.path.join(src_mimic_version, "hosp", "admissions" + ".csv.gz"),
        infer_schema_length=0,
    ).lazy()

    decompressed_dir = os.path.join(args.destination, "decompressed")
    os.mkdir(decompressed_dir)

    temp_dir = os.path.join(args.destination, "temp")
    os.mkdir(temp_dir)

    os.mkdir(os.path.join(temp_dir, "flat_data"))

    print("Processing tables into " + temp_dir)

    for table_name, mapping_codes in all_tables.items():
        print("Processing", table_name)
        if mapping_codes is None:
            continue

        uncompressed_path = os.path.join(
            decompressed_dir, table_name.replace("/", "_") + ".csv"
        )
        compressed_path = os.path.join(src_mimic_version, table_name + ".csv.gz")

        with open(uncompressed_path, "w") as f:
            subprocess.run(["gunzip", "-c", compressed_path], stdout=f)

        if not isinstance(mapping_codes, list):
            mapping_codes = [mapping_codes]

        for map_index, mapping_code in enumerate(mapping_codes):
            reader = pl.read_csv_batched(
                uncompressed_path,
                infer_schema_length=0,
                batch_size=10_000_000,
            )

            patient_id = pl.col("subject_id").cast(pl.Int64)
            time = meds_etl.flat.parse_time(mapping_code["time"], MIMIC_TIME_FORMATS)

            code = mapping_code["code"]
            if "value" in mapping_code:
                value = mapping_code["value"]
            else:
                value = pl.lit(None, dtype=str)

            d, n, t = meds_etl.flat.convert_generic_value_to_specific(value)

            if "metadata" not in mapping_code:
                mapping_code["metadata"] = {}

            mapping_code["metadata"]["table"] = pl.lit(table_name)

            batch_index = 0
            while True:
                batch_index += 1
                table_csv = reader.next_batches(1)
                if table_csv is None:
                    break

                table_csv = table_csv[0].lazy()

                if mapping_code.get("requires_admission_join", False):
                    table_csv = table_csv.join(admission_table, on="hadm_id")

                if mapping_code.get("possibly_null_code", False):
                    table_csv = table_csv.filter(code.is_not_null())

                if mapping_code.get("possibly_null_time", False):
                    table_csv = table_csv.filter(time.is_not_null())

                columns = {
                    "patient_id": patient_id,
                    "time": time,
                    "code": code,
                }

                columns["datetime_value"] = d
                columns["numeric_value"] = n
                columns["text_value"] = t

                for k, v in mapping_code["metadata"].items():
                    columns[k] = v.alias(k)

                try:
                    event_data = table_csv.select(**columns).collect()
                except Exception as e:
                    print("Could not process", table_name, mapping_codes)
                    print(columns)
                    raise e

                fname = os.path.join(
                    temp_dir,
                    "flat_data",
                    f'{table_name.replace("/", "_")}_{map_index}_{batch_index}.parquet',
                )
                event_data.write_parquet(fname)

        os.remove(uncompressed_path)

    shutil.rmtree(decompressed_dir)

    print("Processing each shard")

    with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
        f.write("{}\n")

    meds_etl.flat.convert_flat_to_meds(
        temp_dir,
        os.path.join(args.destination, "result"),
        num_shards=args.num_shards,
        num_proc=args.num_proc,
        backend=args.backend,
    )

    shutil.move(
        os.path.join(args.destination, "result", "data"),
        os.path.join(args.destination, "data"),
    )
    shutil.rmtree(os.path.join(args.destination, "result"))

    shutil.rmtree(temp_dir)

    code_metadata = {}

    code_metadata_files = {
        "d_labitems_to_loinc": {
            "code": "MIMIC_IV_LABITEM/" + pl.col("itemid (omop_source_code)"),
            "descr": pl.col("label"),
            "parent": pl.col("omop_vocabulary_id") + "/" + pl.col("omop_concept_code"),
        },
        "inputevents_to_rxnorm": {
            "code": "MIMIC_IV_ITEM/"
            + pl.col("itemid (omop_source_code)")
            + "/"
            + pl.col("ordercategorydescription"),
            "descr": pl.col("label"),
            "parent": pl.col("omop_vocabulary_id") + "/" + pl.col("omop_concept_code"),
        },
        "meas_chartevents_main": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid (omop_source_code)"),
            "descr": pl.col("label"),
            "parent": pl.col("omop_vocabulary_id") + "/" + pl.col("omop_concept_code"),
        },
        "outputevents_to_loinc": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid (omop_source_code)"),
            "descr": pl.col("label"),
            "parent": pl.col("omop_vocabulary_id") + "/" + pl.col("omop_concept_code"),
        },
        "proc_datetimeevents": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid (omop_source_code)"),
            "descr": pl.col("label"),
            "parent": pl.col("omop_vocabulary_id") + "/" + pl.col("omop_concept_code"),
        },
        "proc_itemid": {
            "code": "MIMIC_IV_ITEM/" + pl.col("itemid (omop_source_code)"),
            "descr": pl.col("label"),
            "parent": pl.col("omop_vocabulary_id") + "/" + pl.col("omop_concept_code"),
        },
    }

    for fname, mapping_info in code_metadata_files.items():
        with files("meds_etl.mimic.concept_map").joinpath(fname + ".csv").open(
            "rb"
        ) as f:
            table = pl.read_csv(f.read(), infer_schema_length=0)
            code_and_value = table.select(
                code=mapping_info["code"],
                descr=mapping_info["descr"],
                parent=mapping_info["parent"],
            )
            for code, descr, value in code_and_value.iter_rows():
                if code in code_metadata:
                    assert (
                        value == code_metadata[code].get("parent_codes", [None])[0]
                    ), f"{code} {descr} {value} {code_metadata[code]}"
                result = {}

                if descr is not None:
                    result["description"] = descr

                if value is not None:
                    result["parent_codes"] = [value]

                code_metadata[code] = result

    metadata = {
        "dataset_name": "MIMIC-IV",
        "dataset_version": MIMIC_VERSION,
        "etl_name": "meds_etl.mimic",
        "etl_version": meds_etl.__version__,
        "code_metadata": code_metadata,
    }

    jsonschema.validate(instance=metadata, schema=meds.dataset_metadata)

    with open(os.path.join(args.destination, "metadata.json"), "w") as f:
        json.dump(metadata, f)


# Add direct entry point to this file
if __name__ == "__main__":
    main()

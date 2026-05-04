from src.manual_close_reading import (
    build_manual_close_reading_rows,
    build_manual_close_reading_row,
    find_invalid_iso2_rows,
    normalize_country_iso2,
    sanitize_manual_rows,
)


def test_build_manual_close_reading_row_maps_core_fields() -> None:
    row = build_manual_close_reading_row(
        {
            "firm_name": "TestCo",
            "origin_track": "in_denmark",
            "first_legal_entity_name": "TestCo ApS",
            "first_cvr": "12345678",
            "founding_year": 2012,
            "move_year": 2019,
            "moved_to_city": "London",
            "moved_to_country_iso": "GBR",
            "moved_to_first_legal_entity_name": "TestCo Ltd",
            "hq_today_city": "London",
            "hq_today_country_iso": "GBR",
            "status_today": "active",
            "acquirer": "BuyerCo",
            "acq_date": "2020-05-01",
            "validation_label": "true",
            "confidence": "high",
            "validation_reason": "Valid.",
        },
        industry_index={"testco": "software"},
    )

    assert row["firm"] == "TestCo"
    assert row["name_today"] == "TestCo Ltd"
    assert row["name_first"] == "TestCo ApS"
    assert row["cvr"] == "12345678"
    assert row["move_to"] == "GB"
    assert row["industry"] == "tech"
    assert row["founding_year"] == "2012"
    assert row["move_year"] == "2019"
    assert row["real_move_to_country"] == "GB"
    assert row["location_today_country"] == ""
    assert row["source_total"] == ""
    assert row["1_founded_dk"] == ""
    assert row["1_relocation"] == ""
    assert row["1_ma_co_occ"] == ""
    assert row["link"] == ""
    assert row["deal_year"] == "2020"
    assert row["deal_source"] == ""
    assert row["Column1"] == "unclear"


def test_build_manual_close_reading_row_blanks_today_hq_when_same_as_latest() -> None:
    row = build_manual_close_reading_row(
        {
            "firm_name": "SameHQCo",
            "origin_track": "in_denmark",
            "founding_year": 2014,
            "move_year": 2020,
            "moved_to_city": "London",
            "moved_to_country_iso": "GBR",
            "hq_today_city": "London",
            "hq_today_country_iso": "GBR",
            "status_today": "active",
            "validation_label": "true",
            "confidence": "high",
            "validation_reason": "Valid.",
        }
    )

    assert row["real_move_to_city"] == "London"
    assert row["real_move_to_country"] == "GB"
    assert row["move_to"] == "GB"
    assert row["location_today_city"] == ""
    assert row["location_today_country"] == ""


def test_build_manual_close_reading_row_ignores_suffix_only_name_change() -> None:
    row = build_manual_close_reading_row(
        {
            "firm_name": "Acarix AB",
            "first_legal_entity_name": "Acarix A/S",
            "moved_to_first_legal_entity_name": "Acarix AB",
            "origin_track": "in_denmark",
            "moved_to_country_iso": "SWE",
            "sources_status_today": ["https://example.com/about"],
        }
    )

    assert row["name_first"] == "Acarix A/S"
    assert row["name_today"] == "Acarix AB"
    assert row["name_change_source"] == ""


def test_normalize_country_iso2_converts_iso3_and_keeps_iso2() -> None:
    assert normalize_country_iso2("DNK") == "DK"
    assert normalize_country_iso2("US") == "US"
    assert normalize_country_iso2("") == ""


def test_sanitize_manual_rows_normalizes_country_columns() -> None:
    rows = sanitize_manual_rows(
        [
            {
                "index": "",
                "firm": "TestCo",
                "group": "",
                "name_first": "TestCo",
                "name_today": "TestCo",
                "name_change_source": "",
                "move_to": "",
                "founded_dk": "true",
                "relocation": "unclear",
                "ma_co_occ": "false",
                "ma_type": "",
                "relocation_OR_dk_founder_young": "",
                "entre_intra": "",
                "Duplicate": "",
                "comment_final": "",
                "dk_emp_2024": "",
                "total_emp_2024": "",
                "linkedin_cat": "",
                "source_total": "",
                "dk_gr_profit_2024": "",
                "foreign_entity_gr_profit_2024": "",
                "total_gr_profit_2024": "",
                "emp_comment": "",
                "annotator": "",
                "founding_year": "2012",
                "young": "true",
                "founding_source": "",
                "founding_unsure": "false",
                "real_move_to_country": "GBR",
                "real_move_to_city": "",
                "deal_year": "",
                "guesstimate": "",
                "guesstimate_comment": "",
                "relocation_source": "",
                "relocation_unsure": "true",
                "status_today_manual": "",
                "location_today_country": "DNK",
                "location_today_city": "",
                "industry": "tech",
                "today_source": "",
                "today_unsure": "false",
                "additional_comment": "",
                "cvr": "",
                "1_comment": "",
                "2_comment": "",
                "1_founded_dk": "true",
                "1_relocation": "unclear",
                "1_ma_co_occ": "false",
                "1_annotator": "",
                "approve_1st_ann": "",
                "2_founded_dk": "",
                "2_relocation": "",
                "2_ma_co_occ": "",
                "2_annotator": "",
                "date": "",
                "link": "",
                "acq_type": "",
                "acquiror_real_name": "",
                "acq_iso": "USA",
                "acquiror_type": "",
                "deal value in th USD": "",
                "deal_year": "",
                "deal_value_th_usd": "",
                "deal_source": "",
                "deal_number": "",
                "acquiror": "",
                "acquiror_country": "",
                "target": "",
                "target_country": "",
                "deal_type": "",
                "deal_status": "",
                "deal_date": "",
                "method": "LLM pipeline",
                "Column1": "",
            }
        ]
    )
    assert rows[0]["real_move_to_country"] == "GB"
    assert rows[0]["location_today_country"] == "DK"
    assert rows[0]["acq_iso"] == "US"
    assert rows[0]["Column1"] == "unclear"


def test_find_invalid_iso2_rows_reports_non_iso2_values() -> None:
    invalid = find_invalid_iso2_rows(
        [
            {
                "firm": "BadCo",
                "real_move_to_country": "DNKX",
                "location_today_country": "DK",
                "acq_iso": "",
            }
        ]
    )
    assert invalid == [("BadCo", "real_move_to_country", "DNKX")]
def test_build_manual_close_reading_rows_filters_to_non_danish_hq_signal() -> None:
    rows = build_manual_close_reading_rows(
        [
            {
                "firm_name": "KeepCo",
                "origin_track": "in_denmark",
                "validation_label": "false",
                "moved_hq_abroad": "false",
                "founding_country_iso": "DNK",
                "sector_if_known": "enterprise software",
                "moved_to_country_iso": "GBR",
                "hq_today_country_iso": "GBR",
            },
            {
                "firm_name": "DropNonDanishFounding",
                "origin_track": "abroad_danish_founders",
                "validation_label": "true",
                "moved_hq_abroad": "uncertain",
                "founding_country_iso": "USA",
                "moved_to_country_iso": "GBR",
                "hq_today_country_iso": "GBR",
            },
            {
                "firm_name": "DropDenmark",
                "origin_track": "in_denmark",
                "validation_label": "true",
                "moved_hq_abroad": "uncertain",
                "founding_country_iso": "DNK",
                "moved_to_country_iso": "DNK",
                "hq_today_country_iso": "DNK",
            },
            {
                "firm_name": "DropMissing",
                "origin_track": "in_denmark",
                "validation_label": "true",
                "moved_hq_abroad": "uncertain",
                "founding_country_iso": "DNK",
                "moved_to_country_iso": "",
                "hq_today_country_iso": "GBR",
            },
        ]
    )

    assert [row["firm"] for row in rows] == ["DropMissing", "KeepCo"]


def test_build_manual_close_reading_row_blanks_deal_fields_without_deal_evidence() -> None:
    row = build_manual_close_reading_row(
        {
            "firm_name": "NoDealCo",
            "origin_track": "in_denmark",
            "founding_year": 2011,
            "moved_to_city": "Berlin",
            "moved_to_country_iso": "DEU",
            "hq_today_city": "Berlin",
            "hq_today_country_iso": "DEU",
            "ma_after_or_during_move": "false",
            "sources_ma": ["https://example.com/context"],
        }
    )

    assert row["acq_type"] == ""
    assert row["acquiror_real_name"] == ""
    assert row["deal_year"] == ""
    assert row["deal_source"] == ""

import csv
from pathlib import Path

from src.seed_list import build_core_relocation_names, build_exclusion_list, load_seed_firms, select_discovery_prompt_firms


def write_seed_csv(path: Path) -> None:
    rows = [
        {
            "name": "Zendesk",
            "founding_origin": "in Denmark",
            "industry": "software",
            "founded": "2007",
            "moved": "yes",
            "latest_hq_city": "San Francisco",
            "latest_hq_country": "US",
            "today_hq_city": "San Francisco",
            "today_hq_country": "US",
            "status_today": "active",
            "employment_total": "",
            "employment_dk": "",
            "acquiror": "",
            "acquiror_country": "",
            "deal_value_th_usd": "",
            "deal_year": "",
            "method": "manual",
        },
        {
            "name": "Unity",
            "founding_origin": "abroad (Danish founders)",
            "industry": "gaming",
            "founded": "2004",
            "moved": "",
            "latest_hq_city": "",
            "latest_hq_country": "",
            "today_hq_city": "San Francisco",
            "today_hq_country": "US",
            "status_today": "active",
            "employment_total": "",
            "employment_dk": "",
            "acquiror": "",
            "acquiror_country": "",
            "deal_value_th_usd": "",
            "deal_year": "",
            "method": "manual",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_seed_list_loading_and_distinction(tmp_path: Path) -> None:
    path = tmp_path / "preliminary.csv"
    write_seed_csv(path)

    firms = load_seed_firms(path)

    assert [firm.name for firm in firms] == ["Zendesk", "Unity"]
    assert firms[0].founding_origin == "in Denmark"
    assert firms[1].founding_origin == "abroad (Danish founders)"


def test_seed_list_exclusions_and_core_subset(tmp_path: Path) -> None:
    path = tmp_path / "preliminary.csv"
    write_seed_csv(path)

    firms = load_seed_firms(path)

    assert build_exclusion_list(firms) == ["Unity", "Zendesk"]
    assert build_core_relocation_names(firms) == ["Zendesk"]


def test_select_discovery_prompt_firms_filters_29_04_file_to_borsen(tmp_path: Path) -> None:
    path = tmp_path / "preliminary_data_29_04.csv"
    rows = [
        {
            "name": "APM Terminals",
            "founding_origin": "in Denmark",
            "industry": "Transportation",
            "founded": "2001",
            "moved": "2004",
            "latest_hq_city": "The Hague",
            "latest_hq_country": "NL",
            "today_hq_city": "",
            "today_hq_country": "",
            "status_today": "Active",
            "employment_total": "",
            "employment_dk": "6",
            "acquiror": "",
            "acquiror_country": "",
            "deal_value_th_usd": "",
            "deal_year": "",
            "method": "Børsen",
        },
        {
            "name": "Other Manual Firm",
            "founding_origin": "in Denmark",
            "industry": "Tech",
            "founded": "2010",
            "moved": "",
            "latest_hq_city": "",
            "latest_hq_country": "",
            "today_hq_city": "",
            "today_hq_country": "",
            "status_today": "Active",
            "employment_total": "",
            "employment_dk": "",
            "acquiror": "",
            "acquiror_country": "",
            "deal_value_th_usd": "",
            "deal_year": "",
            "method": "manual",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    firms = load_seed_firms(path)
    prompt_firms = select_discovery_prompt_firms(path, firms)

    assert [firm.name for firm in prompt_firms] == ["APM Terminals"]


def test_select_discovery_prompt_firms_keeps_non_29_04_files_unfiltered(tmp_path: Path) -> None:
    path = tmp_path / "preliminary_data_28_04.csv"
    write_seed_csv(path)

    firms = load_seed_firms(path)

    assert select_discovery_prompt_firms(path, firms) == firms

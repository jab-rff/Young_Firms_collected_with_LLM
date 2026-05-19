"""Streamlit app for manual founder/firm LinkedIn review."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_DATA_PATH = Path(
    r"C:\Users\jab\Documents\Young_Firms_collected_with_LLM\data\cumulative\final_review_master_abroad_danish_founders_unique_firms_with_diffbot_review_true_founders_linkedin.csv"
)
DEFAULT_JSON_BACKUP_PATH = Path("data/dk_founders/final_review_master_abroad_danish_founders_unique_firms_with_diffbot_review_true_founders_linkedin.json")

REVIEWED_COLUMN = "manual_founder_linkedin_reviewed"
FOUNDED_BY_DANE_COLUMN = "founded_by_dane_manual"
FOUNDING_YEAR_COLUMN = "founding_year_manual"
FOUNDING_COUNTRY_COLUMN = "founding_country_iso_manual"
FOUNDING_CITY_COLUMN = "founding_city_manual"
STATUS_TODAY_COLUMN = "status_today_manual"
COUNTRY_TODAY_COLUMN = "country_today_manual"
CITY_TODAY_COLUMN = "city_today_manual"
EMPLOYEES_DK_COLUMN = "employees_dk_manual"
EMPLOYEES_TOTAL_COLUMN = "employees_total_manual"
DK_FOUNDER_SOURCE_COLUMN = "dk_founder_source_manual"
DK_FOUNDER_REASONING_COLUMN = "dk_founder_reasoning_manual"
FOUNDED_OTHER_FIRMS_COLUMN = "founded_other_firms_manual"

REVIEW_COLUMNS = [
    FOUNDED_BY_DANE_COLUMN,
    FOUNDING_YEAR_COLUMN,
    FOUNDING_COUNTRY_COLUMN,
    FOUNDING_CITY_COLUMN,
    STATUS_TODAY_COLUMN,
    COUNTRY_TODAY_COLUMN,
    CITY_TODAY_COLUMN,
    EMPLOYEES_DK_COLUMN,
    EMPLOYEES_TOTAL_COLUMN,
    DK_FOUNDER_SOURCE_COLUMN,
    DK_FOUNDER_REASONING_COLUMN,
    FOUNDED_OTHER_FIRMS_COLUMN,
    REVIEWED_COLUMN,
]

FOUNDED_BY_DANE_OPTIONS = ["unclear", "true", "false"]
STATUS_TODAY_OPTIONS = ["", "acquired", "active", "inactive"]
STATUS_TODAY_MODEL_MAP = {
    "acquired": "acquired",
    "merged": "acquired",
    "active": "active",
    "inactive": "inactive",
    "closed": "inactive",
    "uncertain": "",
}
URL_PATTERN = re.compile(r"https?://[^\s|]+")


def main() -> None:
    st.set_page_config(
        page_title="Founder LinkedIn Review",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_layout_css()
    st.title("Founder LinkedIn Review")

    data_path = DEFAULT_DATA_PATH
    if not data_path.exists():
        st.error(f"CSV file not found: {data_path}")
        return

    rows = load_rows(data_path)
    ensure_review_columns(rows)
    if not rows:
        st.success("No rows found.")
        return

    if "founder_review_index" not in st.session_state:
        st.session_state.founder_review_index = first_unreviewed_index(rows)

    current_index = st.session_state.get("founder_review_index", 0)
    current_index = max(0, min(current_index, len(rows) - 1))
    st.session_state.founder_review_index = current_index
    current_row = dict(rows[current_index])
    current_key = build_row_key(current_row)

    top_left, top_right = st.columns([0.9, 3.1])
    with top_left:
        st.caption(f"Remaining: {len(rows)}")
        st.caption(f"Position: {current_index + 1}")
    with top_right:
        st.markdown(f"### {str(current_row.get('firm_name') or '')}")
        st.caption(
            f"Source: {current_row.get('source', '')} | "
            f"Track: {current_row.get('origin_track', '')} | "
            f"Validation: {current_row.get('validation_label', '')}"
        )

    render_top_linkedin(current_row)

    info_left, info_right = st.columns([1, 1.3])
    with info_left:
        st.markdown("**Model Values**")
        st.write(f"Founding year: `{current_row.get('founding_year', '')}`")
        st.write(f"Founding city: `{current_row.get('founding_city', '')}`")
        st.write(f"Founding country: `{current_row.get('founding_country_iso', '')}`")
        st.write(f"Status today: `{current_row.get('status_today', '')}`")
        st.write(f"Danish founders abroad: `{current_row.get('danish_founders_abroad', '')}`")
    with info_right:
        evidence_columns = st.columns(2)
        with evidence_columns[0]:
            st.text_area(
                "Founder Danish context",
                value=str(current_row.get("founder_danish_context") or ""),
                height=180,
                disabled=True,
            )
        with evidence_columns[1]:
            st.text_area(
                "Evidence summary",
                value=str(current_row.get("evidence_summary") or ""),
                height=180,
                disabled=True,
            )

    with st.form("review_form"):
        edited: dict[str, str] = {}

        form_left, form_mid, form_right = st.columns([1, 1, 1.3])
        with form_left:
            edited[FOUNDED_BY_DANE_COLUMN] = st.selectbox(
                "Founded by dane",
                options=FOUNDED_BY_DANE_OPTIONS,
                index=FOUNDED_BY_DANE_OPTIONS.index(default_founded_by_dane_value(current_row)),
                key=f"{current_key}:{FOUNDED_BY_DANE_COLUMN}",
            )
            edited[FOUNDING_YEAR_COLUMN] = st.text_input(
                "Founding year",
                value=default_manual_value(current_row, FOUNDING_YEAR_COLUMN, "founding_year"),
                key=f"{current_key}:{FOUNDING_YEAR_COLUMN}",
            )
            edited[FOUNDING_COUNTRY_COLUMN] = st.text_input(
                "Founding country",
                value=default_manual_value(current_row, FOUNDING_COUNTRY_COLUMN, "founding_country_iso"),
                key=f"{current_key}:{FOUNDING_COUNTRY_COLUMN}",
            )
            edited[FOUNDING_CITY_COLUMN] = st.text_input(
                "Founding city",
                value=default_manual_value(current_row, FOUNDING_CITY_COLUMN, "founding_city"),
                key=f"{current_key}:{FOUNDING_CITY_COLUMN}",
            )

        with form_mid:
            edited[STATUS_TODAY_COLUMN] = st.selectbox(
                "Status today",
                options=STATUS_TODAY_OPTIONS,
                index=STATUS_TODAY_OPTIONS.index(default_status_value(current_row)),
                key=f"{current_key}:{STATUS_TODAY_COLUMN}",
            )
            edited[COUNTRY_TODAY_COLUMN] = st.text_input(
                "Country today",
                value=default_manual_value(current_row, COUNTRY_TODAY_COLUMN, "hq_today_country_iso"),
                key=f"{current_key}:{COUNTRY_TODAY_COLUMN}",
            )
            edited[CITY_TODAY_COLUMN] = st.text_input(
                "City today",
                value=default_manual_value(current_row, CITY_TODAY_COLUMN, "hq_today_city"),
                key=f"{current_key}:{CITY_TODAY_COLUMN}",
            )
            edited[EMPLOYEES_DK_COLUMN] = st.text_input(
                "Nr of employees in DK",
                value=default_manual_value(current_row, EMPLOYEES_DK_COLUMN),
                key=f"{current_key}:{EMPLOYEES_DK_COLUMN}",
            )
            edited[EMPLOYEES_TOTAL_COLUMN] = st.text_input(
                "Nr of employees in total",
                value=default_manual_value(current_row, EMPLOYEES_TOTAL_COLUMN),
                key=f"{current_key}:{EMPLOYEES_TOTAL_COLUMN}",
            )
            founded_other_firms_default = str(current_row.get(FOUNDED_OTHER_FIRMS_COLUMN) or "").strip().casefold() == "true"
            edited[FOUNDED_OTHER_FIRMS_COLUMN] = "true" if st.checkbox(
                "Founded other firms",
                value=founded_other_firms_default,
                key=f"{current_key}:{FOUNDED_OTHER_FIRMS_COLUMN}",
            ) else "false"

        with form_right:
            edited[DK_FOUNDER_SOURCE_COLUMN] = st.text_area(
                "DK founder source",
                value=default_manual_value(current_row, DK_FOUNDER_SOURCE_COLUMN),
                height=90,
                key=f"{current_key}:{DK_FOUNDER_SOURCE_COLUMN}",
            )
            edited[DK_FOUNDER_REASONING_COLUMN] = st.text_area(
                "DK founder reasoning",
                value=default_manual_value(current_row, DK_FOUNDER_REASONING_COLUMN),
                height=90,
                key=f"{current_key}:{DK_FOUNDER_REASONING_COLUMN}",
            )

        action_columns = st.columns(2)
        with action_columns[0]:
            save_only = st.form_submit_button("Save")
        with action_columns[1]:
            save_next = st.form_submit_button("Save and next")

    nav_columns = st.columns(2)
    with nav_columns[0]:
        if st.button("Previous", disabled=current_index == 0):
            st.session_state.founder_review_index = max(0, current_index - 1)
            st.rerun()
    with nav_columns[1]:
        if st.button("Next", disabled=current_index >= len(rows) - 1):
            st.session_state.founder_review_index = min(len(rows) - 1, current_index + 1)
            st.rerun()

    if save_only or save_next:
        updated_rows = []
        target_key = build_row_key(current_row)
        for row in load_rows(data_path):
            if build_row_key(row) == target_key:
                merged = dict(row)
                for column, value in edited.items():
                    merged[column] = value
                merged[REVIEWED_COLUMN] = "true"
                updated_rows.append(merged)
            else:
                updated_rows.append(row)
        write_rows(updated_rows, data_path, DEFAULT_JSON_BACKUP_PATH)
        if save_next:
            st.session_state.founder_review_index = next_unreviewed_index(updated_rows, current_index + 1)
        st.rerun()

    st.markdown("---")
    founder_columns = st.columns(5)
    for slot, column in enumerate(founder_columns, start=1):
        name = str(current_row.get(f"dk_founder_{slot}") or "").strip()
        with column:
            st.caption(f"Founder {slot}")
            st.caption(name or "[empty]")

    st.markdown("**Links**")
    render_bottom_links(current_row)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = pd.read_csv(path).where(pd.notna, "").to_dict(orient="records")
    rows.sort(
        key=lambda row: (
            str(row.get("dk_founder_1_linkedin") or "").strip() == "",
            str(row.get("firm_name") or "").casefold(),
        )
    )
    return rows


def ensure_review_columns(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for column in REVIEW_COLUMNS:
            row.setdefault(column, "")


def first_unreviewed_index(rows: list[dict[str, Any]]) -> int:
    for index, row in enumerate(rows):
        if not is_reviewed(row):
            return index
    return 0


def next_unreviewed_index(rows: list[dict[str, Any]], start_index: int) -> int:
    for index in range(start_index, len(rows)):
        if not is_reviewed(rows[index]):
            return index
    return min(start_index, len(rows) - 1)


def is_reviewed(row: dict[str, Any]) -> bool:
    return str(row.get(REVIEWED_COLUMN, "") or "").strip().casefold() == "true"


def default_founded_by_dane_value(row: dict[str, Any]) -> str:
    manual = str(row.get(FOUNDED_BY_DANE_COLUMN) or "").strip().casefold()
    if manual in FOUNDED_BY_DANE_OPTIONS:
        return manual
    model_value = str(row.get("danish_founders_abroad") or "").strip().casefold()
    if model_value in {"true", "false"}:
        return model_value
    return "unclear"


def default_status_value(row: dict[str, Any]) -> str:
    manual = str(row.get(STATUS_TODAY_COLUMN) or "").strip().casefold()
    if manual in STATUS_TODAY_OPTIONS:
        return manual
    model_value = str(row.get("status_today") or "").strip().casefold()
    mapped_value = STATUS_TODAY_MODEL_MAP.get(model_value, "")
    if mapped_value in STATUS_TODAY_OPTIONS:
        return mapped_value
    return ""


def default_manual_value(row: dict[str, Any], manual_column: str, fallback_column: str | None = None) -> str:
    manual = str(row.get(manual_column) or "").strip()
    if manual:
        return manual
    if fallback_column is None:
        return ""
    return str(row.get(fallback_column) or "").strip()


def build_row_key(row: dict[str, Any]) -> str:
    return " | ".join(
        [
            str(row.get("source") or "").strip().casefold(),
            str(row.get("firm_name") or "").strip().casefold(),
            str(row.get("first_legal_entity_name") or "").strip().casefold(),
            str(row.get("founding_year") or "").strip().casefold(),
        ]
    )


def render_top_linkedin(row: dict[str, Any]) -> None:
    founder = str(row.get("dk_founder_1") or "").strip() or "Founder 1"
    linkedin = str(row.get("dk_founder_1_linkedin") or "").strip()
    if linkedin:
        render_popup_links([(f"{founder} LinkedIn", linkedin)], empty_text="No founder 1 LinkedIn.")
    else:
        st.caption(f"{founder}: no LinkedIn URL")


def render_bottom_links(row: dict[str, Any]) -> None:
    items: list[tuple[str, str]] = []

    firm_linkedin = str(row.get("firm_linkedin_url_search") or "").strip()
    if firm_linkedin:
        items.append(("Firm LinkedIn", firm_linkedin))

    for slot in range(2, 6):
        founder = str(row.get(f"dk_founder_{slot}") or "").strip()
        founder_linkedin = str(row.get(f"dk_founder_{slot}_linkedin") or "").strip()
        if founder_linkedin:
            items.append((founder or f"Founder {slot}", founder_linkedin))

    for column in (
        "sources_founding",
        "sources_founder_identity",
        "sources_relocation",
        "sources_status_today",
        DK_FOUNDER_SOURCE_COLUMN,
    ):
        for index, url in enumerate(extract_urls(str(row.get(column) or "")), start=1):
            items.append((f"{column} #{index}", url))

    render_popup_links(items, empty_text="No additional links.")


def extract_urls(text: str) -> list[str]:
    urls = []
    seen: set[str] = set()
    for match in URL_PATTERN.findall(text):
        normalized = match.rstrip(".,);]")
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def short_label(url: str) -> str:
    if len(url) <= 70:
        return url
    return f"{url[:67]}..."


def render_popup_links(items: list[tuple[str, str]], empty_text: str) -> None:
    if not items:
        st.caption(empty_text)
        return
    rows = []
    for index, (label, url) in enumerate(items):
        safe_label = html.escape(label)
        safe_url = html.escape(url, quote=True)
        short = html.escape(short_label(url))
        rows.append(
            f"""
            <div style="margin: 0 0 8px 0;">
              <button
                onclick="window.open('{safe_url}', '_blank', 'noopener,noreferrer,width=1400,height=1000,left=80,top=60'); return false;"
                style="cursor:pointer;padding:6px 10px;border:1px solid #bbb;border-radius:6px;background:#f8f8f8;">
                {safe_label}
              </button>
              <span style="margin-left:8px;color:#555;">{short}</span>
            </div>
            """
        )
    components.html("".join(rows), height=min(900, 44 * len(items) + 18), scrolling=True)


def write_rows(rows: list[dict[str, Any]], path: Path, json_backup_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for column in REVIEW_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json_backup(rows, json_backup_path)


def save_json_backup(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def inject_layout_css() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {display: none !important;}
        .block-container {
            max-width: 100% !important;
            padding-top: 0.8rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

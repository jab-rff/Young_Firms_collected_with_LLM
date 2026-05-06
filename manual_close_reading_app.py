"""Streamlit app for case-by-case manual close reading."""

from __future__ import annotations

from pathlib import Path
import re

import streamlit as st

from src.manual_close_reading import (
    DEFAULT_FINAL_DATASET_PATH,
    HUMAN_VALIDATION_COLUMN,
    HUMAN_VALIDATION_OPTIONS,
    MANUAL_REVIEWED_COLUMN,
    DEFAULT_MANUAL_REVIEW_PATH,
    DEFAULT_VALIDATED_MASTER_PATH,
    EDITABLE_COLUMNS,
    FINAL_COLUMNS,
    REASONING_COLUMNS,
    build_manual_close_reading_rows,
    find_invalid_iso2_rows,
    load_existing_manual_rows,
    load_jsonl,
    sanitize_manual_rows,
    save_manual_close_reading_rows,
)

TRI_STATE_FIELDS = {
    "founded_dk",
    "relocation",
    "ma_co_occ",
    "young",
    "founding_unsure",
    "relocation_unsure",
    "today_unsure",
    "1_founded_dk",
    "1_relocation",
    "1_ma_co_occ",
    "2_founded_dk",
    "2_relocation",
    "2_ma_co_occ",
}

TEXT_AREA_FIELDS = {
    "comment_final",
    "emp_comment",
    "guesstimate_comment",
    "additional_comment",
    "1_comment",
    "2_comment",
}

COMMENTARY_FIELDS = {
    "group",
    "Duplicate",
    "annotator",
    "1_annotator",
    "2_annotator",
    "approve_1st_ann",
    "method",
    "acquiror_type",
    "linkedin_cat",
    "entre_intra",
    "cvr",
}

URL_PATTERN = re.compile(r"https?://[^\s|]+")


def main() -> None:
    st.set_page_config(page_title="Manual Close Reading", layout="wide")
    st.title("Manual Close Reading")

    validated_path = Path(
        st.sidebar.text_input("Validated master JSONL", value=str(DEFAULT_VALIDATED_MASTER_PATH))
    )
    working_sheet_path = Path(
        st.sidebar.text_input("Working review CSV", value=str(DEFAULT_MANUAL_REVIEW_PATH))
    )
    final_export_path = Path(
        st.sidebar.text_input("Final dataset CSV", value=str(DEFAULT_FINAL_DATASET_PATH))
    )

    if not validated_path.exists():
        st.error(f"Validated master file not found: {validated_path}")
        return

    validated_records = load_jsonl(validated_path)
    existing_rows = load_existing_manual_rows(working_sheet_path)
    rows = build_manual_close_reading_rows(validated_records, existing_rows=existing_rows)
    if not rows:
        st.warning("No cases were found.")
        return

    validation_options = ["all", *HUMAN_VALIDATION_OPTIONS]
    filter_human_validation = st.sidebar.selectbox("Human validation", validation_options, index=0)

    filtered_rows = [
        row
        for row in rows
        if str(row.get(MANUAL_REVIEWED_COLUMN, "false")).strip().lower() != "true"
        and (
            filter_human_validation == "all"
            or row.get(HUMAN_VALIDATION_COLUMN, "unclear") == filter_human_validation
        )
    ]
    if not filtered_rows:
        st.warning("No cases match the current filters.")
        return

    if "manual_case_filter" not in st.session_state:
        st.session_state.manual_case_filter = filter_human_validation
    if "manual_case_index" not in st.session_state or st.session_state.manual_case_filter != filter_human_validation:
        st.session_state.manual_case_index = _first_unreviewed_index(filtered_rows)
        st.session_state.manual_case_filter = filter_human_validation
    st.session_state.manual_case_index = max(0, min(st.session_state.manual_case_index, len(filtered_rows) - 1))

    current_index = st.session_state.manual_case_index
    current_row = dict(filtered_rows[current_index])
    current_case_key = _build_case_key(current_row, current_index)

    st.write(f"Case {current_index + 1} of {len(filtered_rows)}")
    st.subheader(current_row.get("firm", ""))
    st.caption(
        f"Track: {current_row.get('origin_track', '')} | "
        f"Pipeline validation: {current_row.get('validation_label', '')} | "
        f"Human validation: {current_row.get(HUMAN_VALIDATION_COLUMN, 'unclear')}"
    )
    _render_clickable_links(current_row)

    with st.form("case_form"):
        edited = {}
        left, right = st.columns(2)
        form_columns = [
            column for column in EDITABLE_COLUMNS if column not in {HUMAN_VALIDATION_COLUMN, MANUAL_REVIEWED_COLUMN}
        ]
        split_index = (len(form_columns) + 1) // 2
        left_fields = form_columns[:split_index]
        right_fields = form_columns[split_index:]

        with left:
            _render_fields(left_fields, current_row, edited, current_case_key)

        with right:
            _render_fields(right_fields, current_row, edited, current_case_key)

        st.markdown("**Reasoning**")
        for column in REASONING_COLUMNS:
            st.text_area(column, value=current_row.get(column, ""), height=80, disabled=True)

        validation_options = HUMAN_VALIDATION_OPTIONS
        current_validation = str(current_row.get(HUMAN_VALIDATION_COLUMN, "") or "unclear")
        if current_validation not in validation_options:
            current_validation = "unclear"

        action_columns = st.columns([1, 1, 1.4])
        with action_columns[0]:
            save_only = st.form_submit_button("Save")
        with action_columns[1]:
            save_next = st.form_submit_button("Save and next case")
        with action_columns[2]:
            edited[HUMAN_VALIDATION_COLUMN] = st.selectbox(
                HUMAN_VALIDATION_COLUMN,
                options=validation_options,
                index=validation_options.index(current_validation),
                key=f"{current_case_key}:{HUMAN_VALIDATION_COLUMN}",
            )

    controls = st.columns(2)
    with controls[0]:
        if st.button("Previous case", disabled=current_index == 0):
            st.session_state.manual_case_index = max(0, current_index - 1)
            st.rerun()
    with controls[1]:
        if st.button("Next case", disabled=current_index >= len(filtered_rows) - 1):
            st.session_state.manual_case_index = min(len(filtered_rows) - 1, current_index + 1)
            st.rerun()

    if save_only or save_next:
        updated_rows = []
        target_key = current_row.get("firm", "")
        for row in rows:
            row_key = row.get("firm", "")
            if row_key == target_key:
                merged = dict(row)
                for column in EDITABLE_COLUMNS:
                    merged[column] = edited.get(column, row.get(column, ""))
                merged[MANUAL_REVIEWED_COLUMN] = "true"
                updated_rows.append(merged)
            else:
                updated_rows.append(row)
        sanitized = sanitize_manual_rows(updated_rows)
        save_manual_close_reading_rows(
            sanitized,
            manual_review_path=working_sheet_path,
            final_dataset_path=final_export_path,
        )
        invalid_rows = find_invalid_iso2_rows(sanitized)
        if invalid_rows:
            st.warning(f"Saved, but {len(invalid_rows)} country fields are not valid ISO-2 codes.")
        else:
            st.success("Saved.")
        if save_next and current_index < len(filtered_rows) - 1:
            st.session_state.manual_case_index = _next_unreviewed_index(filtered_rows, current_index + 1)
            st.rerun()

    st.markdown("**Final dataset columns**")
    st.code(", ".join(FINAL_COLUMNS))

def _render_fields(
    columns: list[str],
    current_row: dict[str, str],
    edited: dict[str, str],
    case_key: str,
) -> None:
    for column in columns:
        current_value = str(current_row.get(column, "") or "")
        widget_key = f"{case_key}:{column}"
        if column in TRI_STATE_FIELDS:
            options = ["true", "false", "unclear", ""]
            value = current_value if current_value in options else ""
            edited[column] = st.selectbox(column, options=options, index=options.index(value), key=widget_key)
        elif column in TEXT_AREA_FIELDS:
            edited[column] = st.text_area(column, value=current_value, height=100, key=widget_key)
        else:
            edited[column] = st.text_input(column, value=current_value, key=widget_key)


def _build_case_key(current_row: dict[str, str], current_index: int) -> str:
    firm = str(current_row.get("firm", "") or "").strip()
    return f"case:{current_index}:{firm}"


def _first_unreviewed_index(rows: list[dict[str, str]]) -> int:
    for index, row in enumerate(rows):
        if str(row.get(MANUAL_REVIEWED_COLUMN, "false") or "false").strip().lower() != "true":
            return index
    return 0


def _next_unreviewed_index(rows: list[dict[str, str]], start_index: int) -> int:
    for index in range(start_index, len(rows)):
        if str(rows[index].get(MANUAL_REVIEWED_COLUMN, "false") or "false").strip().lower() != "true":
            return index
    return min(start_index, len(rows) - 1)


def _render_clickable_links(current_row: dict[str, str]) -> None:
    link_groups = _collect_clickable_links(current_row)
    if not link_groups:
        return
    st.markdown("**Links**")
    for column, urls in link_groups:
        rendered_urls = " | ".join(f"[{_short_link_label(url)}]({url})" for url in urls)
        st.markdown(f"`{column}`: {rendered_urls}")


def _collect_clickable_links(current_row: dict[str, str]) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    seen_urls: set[str] = set()
    for column in [*EDITABLE_COLUMNS, *REASONING_COLUMNS]:
        value = str(current_row.get(column, "") or "").strip()
        if not value:
            continue
        column_urls: list[str] = []
        for url in URL_PATTERN.findall(value):
            normalized_url = url.rstrip(".,);]")
            if normalized_url and normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                column_urls.append(normalized_url)
        if column_urls:
            groups.append((column, column_urls))
    return groups


def _short_link_label(url: str) -> str:
    if len(url) <= 60:
        return url
    return f"{url[:57]}..."


if __name__ == "__main__":
    main()

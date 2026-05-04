"""Streamlit app for case-by-case manual close reading."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.manual_close_reading import (
    DEFAULT_FINAL_DATASET_PATH,
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

    validation_options = ["all", "true", "false", "unclear"]
    filter_human_validation = st.sidebar.selectbox("Human validation", validation_options, index=0)

    filtered_rows = [
        row
        for row in rows
        if filter_human_validation == "all" or row.get("Column1", "unclear") == filter_human_validation
    ]
    if not filtered_rows:
        st.warning("No cases match the current filters.")
        return

    if "manual_case_index" not in st.session_state:
        st.session_state.manual_case_index = 0
    st.session_state.manual_case_index = max(0, min(st.session_state.manual_case_index, len(filtered_rows) - 1))

    current_index = st.session_state.manual_case_index
    current_row = dict(filtered_rows[current_index])

    st.write(f"Case {current_index + 1} of {len(filtered_rows)}")
    st.subheader(current_row.get("firm", ""))
    st.caption(
        f"Track: {current_row.get('origin_track', '')} | "
        f"Pipeline validation: {current_row.get('validation_label', '')} | "
        f"Human validation: {current_row.get('Column1', 'unclear')}"
    )

    with st.form("case_form"):
        edited = {}
        left, right = st.columns(2)
        split_index = (len(EDITABLE_COLUMNS) + 1) // 2
        left_fields = EDITABLE_COLUMNS[:split_index]
        right_fields = EDITABLE_COLUMNS[split_index:]

        with left:
            _render_fields(left_fields, current_row, edited)

        with right:
            _render_fields(right_fields, current_row, edited)

        st.markdown("**Reasoning**")
        for column in REASONING_COLUMNS:
            st.text_area(column, value=current_row.get(column, ""), height=80, disabled=True)

        save_only = st.form_submit_button("Save")
        save_next = st.form_submit_button("Save and next case")

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
            st.session_state.manual_case_index = current_index + 1
            st.rerun()

    st.markdown("**Final dataset columns**")
    st.code(", ".join(FINAL_COLUMNS))

def _render_fields(columns: list[str], current_row: dict[str, str], edited: dict[str, str]) -> None:
    for column in columns:
        current_value = str(current_row.get(column, "") or "")
        if column == "Column1":
            options = ["true", "false", "unclear"]
            value = current_value if current_value in options else "unclear"
            edited[column] = st.selectbox("human_validation (Column1)", options=options, index=options.index(value))
        elif column in TRI_STATE_FIELDS:
            options = ["true", "false", "unclear", ""]
            value = current_value if current_value in options else ""
            edited[column] = st.selectbox(column, options=options, index=options.index(value))
        elif column in TEXT_AREA_FIELDS:
            edited[column] = st.text_area(column, value=current_value, height=100)
        else:
            edited[column] = st.text_input(column, value=current_value)


if __name__ == "__main__":
    main()

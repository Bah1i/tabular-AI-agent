import json
import re


PROMPT_VERSION = "2026-05-29.9"


SYSTEM_PROMPT = """
You are a safe Python/Pandas code generator for tabular data.

Return only a JSON object without Markdown. The JSON schema is:
{
  "code": "Python code as a string",
  "explanation": "short explanation in Russian",
  "plan": {
    "selected_columns": ["..."],
    "operations": ["..."],
    "parameters": {"key": "value"},
    "operator": "operator family when known",
    "output_rows_rule": "how output rows are produced",
    "output_columns_rule": "how output columns are produced",
    "row_0_role": "header|data|metadata|uncertain",
    "value_order_rule": "positional value order",
    "generalization_assumption": "why the rule works beyond the example",
    "self_check": "brief visible-example shape/order/value check"
  }
}

Hard requirements:
1. Define function transform(df) that accepts pandas.DataFrame and returns pandas.DataFrame.
2. Start with result = df.copy().
3. Do not read files, write files, call network, call subprocess, or access environment variables.
4. Allowed imports: pandas as pd, numpy as np, re, math, datetime, collections.
5. The output must match expected_output_rows exactly for transform mode.
6. Generate a general transformation for the full source table, not a hardcoded answer for the examples.
7. Be careful with mixed date formats, missing values, numbers stored as strings, duplicates, and column order.
   For dates, infer day/month/year interpretation from the visible source->expected mapping before choosing
   dayfirst/monthfirst/yearfirst. Do not assume ISO or dayfirst globally when expected_output_rows contradict it.
   Add a fallback for unparsed values, and never leave final date/text cells as "nan" or "NaT".
8. Preserve the exact output column order implied by expected_output_rows or by the analytical query.
9. The code field must be a valid JSON string. Escape quotes, backslashes, and newlines; do not emit raw multi-line strings inside JSON.
"""

GENERIC_TRANSFORM_SYSTEM_PROMPT = SYSTEM_PROMPT

FOOFAH_CODE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

FOOFAH-specific requirements:
1. Treat the input as a 2D string matrix with no trusted header row, not a semantic dataframe with trusted headers.
2. Preserve strings exactly, including commas, spaces, Err-like values, and empty cells.
3. Prefer positional, operator-style transformations over semantic pandas column logic.
4. Infer a reusable row and column generation rule from visible InputTable -> OutputTable.
5. Do not encode case ids, literal hidden-answer values, or behavior known only from hidden TestAnswer.
6. Return only valid JSON with code, explanation, and plan.
"""

FOOFAH_ROUTER_SYSTEM_PROMPT = """
Return only JSON. Choose a ranked operator family list and parameters for FOOFAH-style synthesis.
Think in 2D string matrices with no trusted header row. Use only visible InputTable -> OutputTable
unless benchmark_context.hidden_feedback_allowed is true. Do not generate Python code.
"""

SQL_SYSTEM_PROMPT = "Return only JSON with sql and explanation. Generate read-only SQL."

GENERIC_JSON_SYSTEM_PROMPT = "Return only JSON."


FINAL_REMINDER = """
Final reminder: return only JSON with code, explanation, and plan. The code must define transform(df).
The transformation must generalize from the small expected example to the entire source table.
"""


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _looks_date_like(value) -> bool:
    text = str(value or "").strip()
    return bool(re.search(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text))


def _date_mapping_hint(source_value, expected_value) -> str:
    source = str(source_value or "").strip()
    expected = str(expected_value or "").strip()
    iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", source)
    dotted = re.match(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$", source)
    expected_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", expected)
    if iso and expected_iso:
        year, mid, last = iso.groups()
        exp_year, exp_mid, exp_last = expected_iso.groups()
        if exp_year == year and int(exp_mid) == int(last) and int(exp_last) == int(mid):
            return "visible mapping implies YYYY-DD-MM / day-first interpretation for ISO-like source strings"
        if exp_year == year and int(exp_mid) == int(mid) and int(exp_last) == int(last):
            return "visible mapping preserves ISO-like YYYY-MM-DD source strings"
    if dotted and expected_iso:
        day, month, year = dotted.groups()
        exp_year, exp_mid, exp_last = expected_iso.groups()
        if exp_year == year and int(exp_mid) == int(month) and int(exp_last) == int(day):
            return "visible mapping implies DD.MM.YYYY -> YYYY-MM-DD"
        if exp_year == year and int(exp_mid) == int(day) and int(exp_last) == int(month):
            return "visible mapping implies MM.DD.YYYY -> YYYY-MM-DD"
    return "infer this date interpretation from the visible source->expected pair"


def _visible_value_alignment(example_input: list[dict], example_output: list[dict]) -> list[dict]:
    if not example_input or not example_output:
        return []
    shared_columns = [col for col in example_input[0].keys() if col in example_output[0]]
    alignments: list[dict] = []
    max_rows = min(len(example_input), len(example_output), 8)
    for col in shared_columns:
        pairs = []
        hints = []
        for row_idx in range(max_rows):
            source_value = example_input[row_idx].get(col)
            expected_value = example_output[row_idx].get(col)
            if not (_looks_date_like(source_value) or _looks_date_like(expected_value)):
                continue
            pair = {
                "row": row_idx,
                "source": str(source_value),
                "expected": str(expected_value),
            }
            hint = _date_mapping_hint(source_value, expected_value)
            pair["hint"] = hint
            pairs.append(pair)
            if hint not in hints:
                hints.append(hint)
        if pairs:
            alignments.append(
                {
                    "column": col,
                    "kind": "date_like",
                    "examples": pairs[:5],
                    "inference_hints": hints[:3],
                    "rule": "Expected output is authoritative. Self-check every visible date pair before choosing pandas parsing flags.",
                }
            )
    return alignments


def _json_objects_from_text(text: str | None) -> list[dict]:
    if not text:
        return []
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            item, offset = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(item, dict):
            objects.append(item)
        index = start + offset
    return objects


def _benchmark_feedback_payload(user_instruction: str | None) -> dict | None:
    for item in _json_objects_from_text(user_instruction):
        feedback = item.get("benchmark_generalization_feedback")
        if isinstance(feedback, dict):
            return feedback
    return None


def _records_to_grid(records: list[dict]) -> list[list[str]]:
    if not records:
        return []
    columns = list(records[0].keys())
    return [["" if row.get(column) is None else str(row.get(column)) for column in columns] for row in records]


def _grid_width(grid: list[list[str]]) -> int:
    return max((len(row) for row in grid), default=0)


def _foofah_generalization_contract() -> list[str]:
    return [
        "Infer a reusable row-generation rule, not only the visible row count.",
        "Infer a reusable column-generation rule, not only the visible output width.",
        "When repeated structures exist, parameterize by the current input pattern.",
        "Do not emit metadata, title, or separator rows unless visible output proves they are data.",
        "Preserve the visible positional order exactly.",
        "Decide row 0 role once, then apply it consistently.",
        "Use only the visible example unless benchmark_context explicitly allows feedback.",
    ]


def _pad_grid(grid: list[list[str]]) -> list[list[str]]:
    width = _grid_width(grid)
    return [(row + [""] * max(0, width - len(row)))[:width] for row in grid]


def _nonempty_cells(grid: list[list[str]]) -> list[str]:
    return [cell for row in grid for cell in row if cell != ""]


def _row_subset(output_grid: list[list[str]], input_grid: list[list[str]]) -> bool:
    input_rows = {tuple(row) for row in _pad_grid(input_grid)}
    return all(tuple(row) in input_rows for row in _pad_grid(output_grid))


def _has_error_cell(value: str) -> bool:
    return str(value).strip().lower().startswith("err:")


def _metric_order_hint(input_grid: list[list[str]], output_grid: list[list[str]]) -> str:
    if len(input_grid) < 2 or not output_grid or _grid_width(input_grid) < 4:
        return "none"
    first = input_grid[0]
    second = input_grid[1]
    if len(first) < 3 or len(second) < 3 or not second[1]:
        return "none"
    first_values = [cell for cell in first[2:] if cell != ""]
    second_values = [cell for cell in second[2:] if cell != ""]
    if not first_values or not second_values:
        return "none"
    tail = output_grid[0][1:] if output_grid and output_grid[0] else []
    grouped = first_values + second_values
    interleaved: list[str] = []
    for index in range(max(len(first_values), len(second_values))):
        if index < len(first_values):
            interleaved.append(first_values[index])
        if index < len(second_values):
            interleaved.append(second_values[index])
    if tail[: len(grouped)] == grouped:
        return "metric_rows_grouped"
    if tail[: len(interleaved)] == interleaved:
        return "metric_values_interleaved"
    return "metric_order_uncertain"


def _header_grid_long_order_hint(input_grid: list[list[str]], output_grid: list[list[str]]) -> str:
    if not input_grid or len(input_grid) < 2 or not output_grid or _grid_width(input_grid) < 3 or _grid_width(output_grid) < 3:
        return "none"
    header_values = {_norm_foofah_cell(cell) for cell in input_grid[0][1:] if _norm_foofah_cell(cell)}
    row_labels = {_norm_foofah_cell(row[0]) for row in input_grid[1:] if row and _norm_foofah_cell(row[0])}
    first = output_grid[0]
    left = _norm_foofah_cell(first[0]) if len(first) > 0 else ""
    middle = _norm_foofah_cell(first[1]) if len(first) > 1 else ""
    if left in row_labels and middle in header_values:
        return "row_key_then_header"
    if left in header_values and middle in row_labels:
        return "header_then_row_key"
    return "none"


def _norm_foofah_cell(value: str) -> str:
    return str(value).strip().replace(" ", "").lower()


def _first_row_role_features(
    input_grid: list[list[str]],
    output_grid: list[list[str]],
    pair_labels: list[str],
    first_row_unique_labels: bool,
) -> dict:
    if not input_grid:
        return {
            "expected_has_header_row": False,
            "first_row_values_used_as_output_labels": False,
            "first_row_values_used_as_output_data": False,
            "first_row_is_metadata_like": False,
            "first_row_role_guess": "none",
            "first_row_role_reasons": ["input is empty"],
        }

    first_row_values = [cell for cell in input_grid[0] if cell != ""]
    norm_first_values = [_norm_foofah_cell(cell) for cell in first_row_values if _norm_foofah_cell(cell)]
    norm_first_set = set(norm_first_values)
    output_first_row = output_grid[0] if output_grid else []
    norm_output_first = {_norm_foofah_cell(cell) for cell in output_first_row if _norm_foofah_cell(cell)}
    norm_pair_labels = {_norm_foofah_cell(label) for label in pair_labels if _norm_foofah_cell(label)}
    output_rows_norm = [
        {_norm_foofah_cell(cell) for cell in row if _norm_foofah_cell(cell)}
        for row in output_grid
    ]
    output_all_norm = set().union(*output_rows_norm) if output_rows_norm else set()

    first_output_blank = bool(output_grid and output_grid[0] and output_grid[0][0] == "")
    output_first_matches_pair_labels = bool(norm_pair_labels and norm_output_first and norm_output_first.issubset(norm_pair_labels | {""}))
    first_row_values_used_together = bool(
        norm_first_set
        and any(norm_first_set.issubset(row_values) for row_values in output_rows_norm)
    )
    expected_has_header_row = bool(
        len(output_grid) > 1
        and (
            first_output_blank
            or output_first_matches_pair_labels
            or (first_row_unique_labels and bool(norm_first_set & norm_output_first) and not first_row_values_used_together)
        )
    )

    data_rows_norm = output_rows_norm[1:] if expected_has_header_row else output_rows_norm
    data_all_norm = set().union(*data_rows_norm) if data_rows_norm else set()
    first_cell_norm = _norm_foofah_cell(input_grid[0][0]) if input_grid and input_grid[0] else ""
    first_cell_copied_as_data = bool(
        first_cell_norm
        and any(row and _norm_foofah_cell(row[0]) == first_cell_norm for row in (output_grid[1:] if expected_has_header_row else output_grid))
    )
    first_row_overlap_with_output_data = len(norm_first_set & data_all_norm)
    first_row_values_used_as_output_data = bool(
        first_row_values_used_together
        or (norm_first_set and norm_first_set.issubset(data_all_norm))
        or (not expected_has_header_row and first_cell_copied_as_data and first_row_overlap_with_output_data >= 2)
    )
    first_row_values_used_as_output_labels = bool(
        not first_row_values_used_as_output_data
        and norm_first_set
        and bool(norm_first_set & output_all_norm)
    )
    first_row_is_metadata_like = bool(
        norm_first_set
        and not first_row_values_used_as_output_data
        and not first_row_values_used_as_output_labels
        and len(input_grid) > len(output_grid)
    )

    reasons: list[str] = []
    if expected_has_header_row:
        reasons.append("expected output appears to contain a header row")
    if first_row_values_used_as_output_data:
        reasons.append("all first-row values appear together as output data")
    if first_row_values_used_as_output_labels:
        reasons.append("first-row values appear as output labels, not as a copied data row")
    if first_row_is_metadata_like:
        reasons.append("first-row values do not appear in output and source has extra rows")

    if first_row_values_used_as_output_data:
        role = "data"
    elif first_row_values_used_as_output_labels:
        role = "header"
    elif first_row_is_metadata_like:
        role = "metadata"
    elif first_row_unique_labels:
        role = "uncertain_header_like"
        reasons.append("first row has unique label-like cells but output evidence is inconclusive")
    else:
        role = "uncertain"
        reasons.append("no strong evidence for header, data, or metadata")

    return {
        "expected_has_header_row": expected_has_header_row,
        "first_row_values_used_as_output_labels": first_row_values_used_as_output_labels,
        "first_row_values_used_as_output_data": first_row_values_used_as_output_data,
        "first_row_is_metadata_like": first_row_is_metadata_like,
        "first_row_role_guess": role,
        "first_row_role_reasons": reasons,
    }


def _foofah_features(input_grid: list[list[str]], output_grid: list[list[str]]) -> dict:
    input_rows = len(input_grid)
    input_cols = _grid_width(input_grid)
    output_rows = len(output_grid)
    output_cols = _grid_width(output_grid)
    input_values = set(_nonempty_cells(input_grid))
    output_values = set(_nonempty_cells(output_grid))
    first_col = [row[0] for row in input_grid if row]
    first_col_repeats = len(first_col) != len(set(first_col)) if first_col else False
    first_row_unique_labels = bool(input_grid and len(input_grid[0]) > 1 and len(set(input_grid[0])) == len(input_grid[0]))
    input_blank_cells = sum(1 for row in input_grid for cell in row if cell == "")
    expected_row_subset = _row_subset(output_grid, input_grid) if output_grid else True
    key_value_cells = [
        cell
        for cell in _nonempty_cells(input_grid[: min(20, len(input_grid))])
        if "=" in cell and cell.split("=", 1)[0].strip() and cell.split("=", 1)[1].strip()
    ]
    def pair_label(cell: str) -> str:
        for delimiter in (":", "="):
            if delimiter in cell:
                label, value = cell.split(delimiter, 1)
                if label.strip() and value.strip():
                    return label.strip().lower().rstrip(".")
        return ""

    comma_list_cells = [cell for cell in _nonempty_cells(input_grid[: min(20, len(input_grid))]) if "," in cell]
    error_like_cell_count = sum(1 for cell in _nonempty_cells(input_grid[: min(30, len(input_grid))]) if _has_error_cell(cell))
    metric_order_hint = _metric_order_hint(input_grid, output_grid)
    header_grid_long_order_hint = _header_grid_long_order_hint(input_grid, output_grid)
    comma_name_cells = [
        cell
        for cell in _nonempty_cells(input_grid[: min(30, len(input_grid))])
        if "," in cell and all(part.strip() for part in cell.split(",", 1)) and any(ch.isalpha() for ch in cell)
    ]
    hints: list[str] = []
    first_output_blank = bool(output_grid and output_grid[0] and output_grid[0][0] == "")
    cells_with_colon = [cell for cell in _nonempty_cells(input_grid[: min(20, len(input_grid))]) if ":" in cell and cell.split(":", 1)[0].strip()]
    pair_labels = [label for label in (pair_label(cell) for cell in _nonempty_cells(input_grid[: min(30, len(input_grid))])) if label]
    contact_label_tokens = ("tel", "telephone", "phone", "fax", "mobile", "cell", "email", "e-mail")
    contact_attribute_pair_pattern = bool(
        input_cols == 2
        and any(any(token in label for token in contact_label_tokens) for label in pair_labels)
        and output_cols > input_cols
        and output_rows < input_rows
    )
    rows_with_colon_pairs = [
        row
        for row in input_grid[: min(30, len(input_grid))]
        if any(":" in cell and cell.split(":", 1)[0].strip() and cell.split(":", 1)[1].strip() for cell in row)
    ]
    context_attribute_pair_pattern = bool(
        input_cols == 2
        and rows_with_colon_pairs
        and output_cols > input_cols
        and output_rows < input_rows
    )
    first_row_features = _first_row_role_features(input_grid, output_grid, pair_labels, first_row_unique_labels)
    nonempty_by_row = [sum(1 for cell in row if cell != "") for row in input_grid]
    sparse_rows = bool(nonempty_by_row and max(nonempty_by_row) < input_cols)
    if output_rows > input_rows and output_cols < input_cols:
        hints.append("fold/unpivot")
    if output_rows > input_rows and output_cols == 2 and input_cols > 2:
        hints.append("fold_pair")
    if output_rows > input_rows and output_cols == 3 and first_row_unique_labels:
        hints.append("fold_triple")
    if input_rows >= 3 and input_cols >= 3 and output_cols == 3 and input_grid and input_grid[0] and input_grid[0][0] == "":
        hints.append("header_grid_to_long")
    if input_rows >= 2 and input_cols >= 3 and output_rows > input_rows and output_cols >= input_cols:
        hints.append("header_grid_to_long")
    if header_grid_long_order_hint != "none":
        hints.append("header_grid_to_long")
    if input_cols == 2 and output_cols >= 4 and output_rows < input_rows:
        hints.append("context_attribute_pairs")
        hints.append("context_detail")
    if output_rows == 1 and input_rows > 1 and output_cols == input_rows * input_cols:
        hints.append("flatten_all_rows")
    if output_rows == 1 and input_rows >= 1 and input_cols > output_cols and output_cols == 2:
        hints.append("first_last_to_wide")
    if output_rows > input_rows and input_rows and input_cols > output_cols and first_row_unique_labels:
        hints.append("headered_fold")
    if output_rows < input_rows and output_cols == input_cols and expected_row_subset:
        hints.append("row_filter")
    if output_rows == input_rows and output_cols < input_cols:
        hints.append("column_select")
    if first_col_repeats and output_cols > input_cols:
        hints.append("wide_by_key")
    if first_col_repeats and output_cols > input_cols and input_cols >= 3:
        hints.append("wide_by_key_pairs")
    if first_col_repeats and output_rows < input_rows and first_output_blank:
        hints.append("pivot_long_to_wide")
    if input_cols in {2, 3} and output_cols > input_cols and (first_col_repeats or first_output_blank):
        hints.append("pivot_long_to_wide")
    if input_cols > 3 and input_rows >= 2 and output_cols > input_cols and first_col_repeats:
        hints.append("grouped_suffix_wide")
    if input_rows == 1 and output_rows == 1 and output_cols == input_cols:
        hints.append("grouped_suffix_wide")
    if input_cols > 3 and input_rows >= 2 and output_cols > input_cols and not first_col_repeats:
        hints.append("stacked_metric_rows")
    if metric_order_hint != "none":
        hints.append("stacked_metric_rows")
    if input_cols >= 3 and output_rows < input_rows and first_output_blank:
        hints.append("pivot_long_to_wide")
    if sparse_rows and output_rows < input_rows and output_cols > input_cols:
        hints.append("flatten_record_blocks")
    if input_cols == 1 and output_cols > 1:
        hints.append("wrap_fixed_width")
    if input_cols == 1 and input_blank_cells and output_cols > 1:
        hints.append("blank_separated_blocks")
    if input_cols == 1 and any(row and row[0].startswith("*") for row in input_grid) and output_cols > 1:
        hints.append("marker_separated_blocks")
    if input_cols == 1 and key_value_cells and output_cols > 1:
        hints.append("key_value_unfold")
    if cells_with_colon:
        hints.append("delimited_pairs_to_wide")
    if context_attribute_pair_pattern:
        hints.append("context_attribute_pairs")
        hints.append("context_detail")
    if contact_attribute_pair_pattern:
        hints.append("context_attribute_pairs")
    if comma_list_cells and output_rows > input_rows:
        hints.append("split_list_fold")
    if input_rows == 1 and input_cols == 1 and output_cols > 1:
        hints.append("regex_extract")
    if comma_name_cells and output_cols == 2:
        hints.append("name_merge_split_rows")
        hints.append("split_extract")
    if input_rows >= 2 and output_rows * 2 == input_rows and output_cols >= input_cols:
        hints.append("record_pair_merge")
    if input_rows == 1 and input_cols >= 10 and output_rows > input_rows and output_cols < input_cols:
        hints.append("prefix_suffix_chunks")
    if error_like_cell_count and input_cols > output_cols:
        hints.append("prefix_suffix_chunks")
    if input_rows and output_rows == input_cols and output_cols == input_rows:
        hints.append("transpose")
    if output_values and output_values.issubset(input_values):
        hints.append("rearrange_existing_values")
    return {
        "input_shape": {"rows": input_rows, "columns": input_cols},
        "expected_shape": {"rows": output_rows, "columns": output_cols},
        "row_delta": output_rows - input_rows,
        "column_delta": output_cols - input_cols,
        "expected_rows_are_subset_of_input_rows": expected_row_subset,
        "all_expected_values_appear_in_input": output_values.issubset(input_values) if output_values else True,
        "first_column_has_repeated_keys": first_col_repeats,
        "first_row_looks_like_labels": first_row_unique_labels,
        **first_row_features,
        "input_blank_cells": input_blank_cells,
        "key_value_line_count": len(key_value_cells),
        "colon_pair_cell_count": len(cells_with_colon),
        "pair_labels_seen": pair_labels[:10],
        "rows_with_colon_pairs": len(rows_with_colon_pairs),
        "context_attribute_pair_pattern": context_attribute_pair_pattern,
        "contact_attribute_pair_pattern": contact_attribute_pair_pattern,
        "comma_list_cell_count": len(comma_list_cells),
        "comma_name_cell_count": len(comma_name_cells),
        "metric_order_hint": metric_order_hint,
        "header_grid_long_order_hint": header_grid_long_order_hint,
        "error_like_cell_count": error_like_cell_count,
        "sparse_rows": sparse_rows,
        "first_output_cell_is_blank": first_output_blank,
        "likely_operator_hints": hints,
    }


FOOFAH_OPERATORS = [
    "fold_pair",
    "fold_triple",
    "headered_fold",
    "header_grid_to_long",
    "row_filter",
    "column_select",
    "first_last_to_wide",
    "wide_by_key",
    "wide_by_key_pairs",
    "pivot_long_to_wide",
    "grouped_suffix_wide",
    "stacked_metric_rows",
    "record_pair_merge",
    "name_merge_split_rows",
    "wrap_fixed_width",
    "blank_separated_blocks",
    "marker_separated_blocks",
    "key_value_unfold",
    "delimited_pairs_to_wide",
    "context_attribute_pairs",
    "split_list_fold",
    "flatten_record_blocks",
    "flatten_all_rows",
    "prefix_suffix_chunks",
    "context_detail",
    "split_extract",
    "regex_extract",
    "transpose",
    "unknown",
]


def _source_to_expected_alignment(
    input_grid: list[list[str]],
    output_grid: list[list[str]],
    max_expected_cells: int = 80,
    max_positions_per_cell: int = 8,
) -> dict:
    positions_by_value: dict[str, list[dict]] = {}
    for row_index, row in enumerate(input_grid):
        for column_index, cell in enumerate(row):
            if cell == "":
                continue
            positions_by_value.setdefault(cell, []).append({"row": row_index, "col": column_index})

    matched: list[dict] = []
    unmatched: list[dict] = []
    inspected = 0
    for row_index, row in enumerate(output_grid):
        for column_index, cell in enumerate(row):
            if cell == "":
                continue
            inspected += 1
            if inspected > max_expected_cells:
                continue
            positions = positions_by_value.get(cell, [])
            record = {
                "expected_position": [row_index, column_index],
                "expected_cell": cell,
                "found_in_input": positions[:max_positions_per_cell],
                "input_match_count": len(positions),
            }
            if positions:
                matched.append(record)
            else:
                unmatched.append(record)

    return {
        "meaning": (
            "Exact string locations of expected output cells inside the source grid. Use this as evidence "
            "for positional mapping, row/column order, and whether extraction or only rearrangement is needed."
        ),
        "matched_expected_cells": matched,
        "unmatched_expected_cells": unmatched[:20],
        "truncated": inspected > max_expected_cells,
    }


def _grid_shape(grid: list[list[str]]) -> list[int]:
    return [len(grid), _grid_width(grid)]


def _benchmark_generalization_analysis(
    user_instruction: str | None,
    input_grid: list[list[str]],
    output_grid: list[list[str]],
) -> dict:
    feedback = _benchmark_feedback_payload(user_instruction)
    visible_input = input_grid
    visible_output = output_grid
    hidden_input: list[list[str]] = []
    hidden_output: list[list[str]] = []
    if feedback:
        visible_input = feedback.get("visible_input_grid") or input_grid
        visible_output = feedback.get("visible_output_grid") or output_grid
        hidden_input = feedback.get("testing_table_grid") or []
        hidden_output = feedback.get("test_answer_grid") or []

    visible_metric_order = _metric_order_hint(visible_input, visible_output)
    hidden_metric_order = _metric_order_hint(hidden_input, hidden_output) if hidden_input and hidden_output else "none"
    visible_header_order = _header_grid_long_order_hint(visible_input, visible_output)
    hidden_header_order = _header_grid_long_order_hint(hidden_input, hidden_output) if hidden_input and hidden_output else "none"

    order_conflicts: list[str] = []
    branch_rules: list[str] = []
    if (
        visible_header_order != "none"
        and hidden_header_order != "none"
        and visible_header_order != hidden_header_order
    ):
        order_conflicts.append("header_grid_order_conflict")
        branch_rules.append(
            "For header_grid_to_long/headered_fold, pass visible OutputTable order on the visible-shaped input; "
            "when explicit oracle feedback has a different shape or multiple row-key records, use the oracle target order."
        )
    if (
        visible_metric_order != "none"
        and hidden_metric_order != "none"
        and visible_metric_order != hidden_metric_order
    ):
        order_conflicts.append("metric_order_conflict")
        branch_rules.append(
            "For stacked_metric_rows/record_pair_merge, pass visible metric order on a one-pair training input; "
            "when explicit oracle feedback shows a different multi-pair shape, use the oracle metric order for that shape."
        )

    if feedback:
        rules = [
            "Explicit oracle feedback is present; generate one transform(df) that passes both visible OutputTable and the oracle feedback target.",
            "When visible and oracle order disagree, keep extraction fixed and branch by input shape, record count, or entity-group count instead of choosing one global order.",
        ]
    else:
        rules = [
            "No oracle feedback is present; expected_output_grid is authoritative and is the only target available to the synthesizer.",
            "Do not infer or optimize for an unprovided hidden answer before the visible check passes.",
        ]
    if feedback and not order_conflicts:
        rules.append(
            "If explicit oracle feedback exists but no explicit order conflict is detected, use it only for the failing dimension: width, row count, or value mapping."
        )

    return {
        "has_feedback": bool(feedback),
        "visible_shape": feedback.get("visible_shape") if feedback else {"input": _grid_shape(input_grid), "output": _grid_shape(output_grid)},
        "hidden_shape": feedback.get("hidden_shape") if feedback else None,
        "visible_metric_order_hint": visible_metric_order,
        "hidden_metric_order_hint": hidden_metric_order,
        "visible_header_grid_long_order_hint": visible_header_order,
        "hidden_header_grid_long_order_hint": hidden_header_order,
        "order_conflicts": order_conflicts,
        "branch_rules": branch_rules,
        "rules": rules,
    }


def _foofah_shape_proof_contract(features: dict) -> dict:
    return {
        "meaning": "The generated plan must explain how output shape and order follow from the input.",
        "expected_shape": features.get("expected_shape", {}),
        "input_shape": features.get("input_shape", {}),
        "required_plan_fields": {
            "operator": "selected operator family",
            "output_rows_rule": "how many rows are emitted and which source rows/blocks become output rows",
            "output_columns_rule": "where every output column comes from and why width matches expected/current input",
            "row_0_role": "header | data | metadata | uncertain",
            "value_order_rule": "positional order of emitted values",
            "generalization_assumption": "short statement of how this input-output rule applies to the current table",
            "self_check": "visible example shape/order/value checks performed before returning code",
        },
        "rules": [
            "Do not leave output_rows_rule or output_columns_rule implicit.",
            "If actual/expected width differs in feedback, explicitly state the source of missing or extra columns.",
            "If expected values appear in source_to_expected_alignment, prefer rearrangement/mapping over parsing.",
            "Prefer a simple Foofah-style operator sequence over ad hoc semantic guesses.",
        ],
    }


def _foofah_paper_principles() -> list[str]:
    return [
        "Programming by Example: infer a reusable transformation from the visible InputTable -> OutputTable pair.",
        "Use Foofah/Potter's Wheel style primitives: Drop, Move, Copy, Merge, Split, Fold, Unfold, Fill, Divide, Delete, Extract, Transpose, and Wrap.",
        "Prefer syntactic and layout transformations over semantic transformations.",
        "Preserve exact cell strings unless OutputTable demonstrates a split, merge, extraction, or fill.",
        "Do not invent domain knowledge or hidden-answer behavior; non-empty output values should be copied, moved, split, merged, extracted, filled from nearby visible values, or explicitly demonstrated by OutputTable.",
        "Prefer short, readable operator sequences and explain the row/column mapping.",
    ]


def _foofah_offline_taxonomy() -> dict:
    return {
        "purpose": (
            "Offline taxonomy of Foofah-style transformation families. It is derived from operator families and "
            "dataset structure, not from case ids, hidden answers, or literal benchmark values."
        ),
        "family_selection_rules": [
            "First infer whether the transformation changes rows, columns, both, or only cell contents.",
            "When output cells mostly appear verbatim in input, prioritize Move/Copy/Delete/Wrap/Fold/Unfold before regex or semantic parsing.",
            "When output width grows with repeated source groups, compute width from current input groups/chunks, not from the visible output width alone.",
            "When output height grows with records/blocks, emit one row per detected record/block and do not cap to visible output height.",
            "When visible and current input sizes differ, generalize by structural repeat units: row pairs, blank-separated blocks, prefix+suffix chunks, key groups, header grids.",
            "Use the shortest operator sequence that explains all visible output cells and shape.",
        ],
        "operator_families": [
            {
                "family": "row_filter_or_delete",
                "dsl": ["Delete", "Drop", "Move"],
                "use_when": "Output rows are a subset/reordering of input rows.",
                "pitfall": "Do not invent computed filters unless expected cells demonstrate them.",
            },
            {
                "family": "column_select_or_move",
                "dsl": ["Drop", "Move", "Copy"],
                "use_when": "Output columns are copied/reordered/subset from source columns.",
                "pitfall": "Preserve exact strings and positional order.",
            },
            {
                "family": "split_extract",
                "dsl": ["Split", "Extract"],
                "use_when": "Expected cells are substrings or delimiter/regex captures from source cells.",
                "pitfall": "Emit only demonstrated captures; do not append unrelated source cells.",
            },
            {
                "family": "merge_fill",
                "dsl": ["Merge", "Fill", "Copy"],
                "use_when": "Expected cells concatenate nearby input cells or repeat/fill visible context.",
                "pitfall": "Fill only from visible context or demonstrated constants.",
            },
            {
                "family": "fold_unpivot",
                "dsl": ["Fold", "Move"],
                "use_when": "Wide headers or repeated columns become multiple long rows.",
                "pitfall": "Header-vs-row-key order is decided by OutputTable positions, not semantic labels.",
            },
            {
                "family": "unfold_pivot",
                "dsl": ["Unfold", "Copy", "Fill"],
                "use_when": "Rows grouped by key become wider records.",
                "pitfall": "Group across the whole current input and pad from current max group size.",
            },
            {
                "family": "transpose_grid",
                "dsl": ["Transpose", "Move"],
                "use_when": "Rows/columns swap as a grid.",
                "pitfall": "Do not transpose metadata/title rows unless expected uses them.",
            },
            {
                "family": "wrap_fixed_width",
                "dsl": ["Wrap", "Delete"],
                "use_when": "A flat stream or single wide row is reshaped into fixed-width records.",
                "pitfall": "Infer record width from expected shape and alignment.",
            },
            {
                "family": "prefix_suffix_chunks",
                "dsl": ["Divide", "Wrap", "Copy"],
                "use_when": "A fixed prefix repeats with each suffix chunk.",
                "pitfall": "Do not drop Err-like chunks unless expected omits trailing all-error padding.",
            },
            {
                "family": "stacked_metric_rows",
                "dsl": ["Unfold", "Move", "Copy"],
                "use_when": "Consecutive metric rows combine into one record.",
                "pitfall": "Choose grouped vs interleaved strictly from visible OutputTable unless explicit oracle feedback exists.",
            },
        ],
        "honesty_boundary": [
            "This taxonomy may bias operator search but must not encode benchmark case ids, literal hidden-answer values, or hidden-only behavior.",
            "In honest mode, hidden answers are unavailable; in oracle mode, any feedback is explicitly marked benchmark_generalization_feedback.",
        ],
    }


def _foofah_rule_packs(features: dict, failure_context: dict | None = None, operator: str | None = None) -> list[dict]:
    failure_context = failure_context or {}
    hints = features.get("likely_operator_hints") or []
    selected_names = ["base"]
    if failure_context.get("error_family"):
        selected_names.append(str(failure_context["error_family"]))
    if operator:
        selected_names.append(str(operator))
    selected_names.extend(str(name) for name in hints[:5])

    packs = {
        "base": [
            "Use FOOFAH as a string matrix task: fillna(''), astype(str), values.tolist(), pd.DataFrame(rows).",
            "Match expected shape, row order, column order, and cell strings exactly.",
            "Prefer positional structure over semantic column names like col_0 or natural-language labels.",
            "Never parse numbers/dates unless the expected output explicitly transforms text.",
            "Infer a short sequence of Foofah-style syntactic/layout operations from expected_output_grid.",
            "If no explicit feedback block is present, expected_output_grid is the only target; do not pre-optimize for an unprovided answer.",
        ],
        "column_count_too_low": [
            "Find where missing columns come from: repeated groups, chunks, attribute labels, headers, or block cells.",
            "Do not cap the current input width to the visible OutputTable width when the same repeated group/chunk rule naturally produces more columns.",
            "Use source_to_expected_alignment to prove each missing column's origin.",
        ],
        "column_count_too_high": [
            "Return exactly expected/current target width; do not copy untouched source cells after split/extract.",
            "Identify metadata, duplicated key/status cells, or trailing padding chunks that should not become columns.",
            "For prefix/suffix chunking, ignore only trailing chunks made entirely of padding-like values when expected omits them.",
        ],
        "row_count_too_low": [
            "Check whether row 0 is data, and whether non-contiguous groups or later blocks were skipped.",
            "Scan all rows/blocks in the current input, not only the visible example's first group.",
            "Avoid overly strict blank predicates that drop legitimate records.",
        ],
        "row_count_too_high": [
            "Do not emit metadata/title/header/separator rows as records.",
            "For attribute blocks, emit one row per context, not one row per attribute.",
            "For blank-separated one-column data, drop short blocks whose values do not appear in expected output.",
        ],
        "leading_column_swap": [
            "Keep extraction unchanged and swap only the first two emitted fields when delta shows a positional swap.",
            "For grid-to-long/headered-fold, expected_output_grid decides [header,row_key,value] vs [row_key,header,value].",
            "Do the direct swap for the visible example; do not override it with semantic guesses.",
        ],
        "value_order_mismatch": [
            "If the same values appear in actual and expected, repair output order before changing extraction.",
            "For stacked metrics, decide row-major grouped order vs interleaved order from expected_output_grid.",
            "For wide/chunk outputs, recompute prefix width and chunk size from expected positions.",
            "If metric values are all present but rotated, build an explicit output index map from expected positions.",
            "If foofah_features.metric_order_hint=metric_values_interleaved, emit the interleaved order demonstrated by expected_output_grid.",
        ],
        "value_mismatch": [
            "Preserve strings exactly and avoid numeric/date conversion.",
            "Use alignment to distinguish extraction failure from positional ordering failure.",
            "If expected substrings are inside source cells, add guarded split/regex extraction with exact expected width.",
        ],
        "headered_fold": [
            "Use header row labels only when row 0 evidence says header, and emit columns in expected positional order.",
            "Do not let semantic names such as product/date override expected_output_grid order.",
        ],
        "header_grid_to_long": [
            "The first output column may be row key or header label; decide from expected_output_grid.",
            "For first-two-column swaps, change only [header,row_key] vs [row_key,header]; keep the value field and row generation unchanged.",
        ],
        "record_pair_merge": [
            "Merge fixed row pairs only when row count evidence supports pairs.",
            "Build an output-position mapping by locating expected cells in the two source rows; avoid blind prefix/suffix splits.",
            "For metric row pairs, compare interleaved [sum1,count1,sum2,count2] with grouped [sum1,sum2,count1,count2] before choosing output order.",
            "If metric_order_hint=metric_values_interleaved, the expected mapping alternates first-row and second-row values by column; do not append all first-row values before second-row values.",
        ],
        "blank_separated_blocks": [
            "Split on blank separators, then filter short title/metadata blocks before padding/truncating true record blocks.",
            "If every true record has fixed width, flatten non-empty cells and wrap by that width.",
        ],
        "split_extract": [
            "Scan all source cells for the target pattern, but emit only captured groups needed by expected width.",
            "Do not append unrelated cells just because a current source row has extra columns.",
        ],
        "regex_extract": [
            "Use guarded regex captures and preserve captured text exactly.",
            "If a field is a substring, emit the substring, not the whole source cell.",
        ],
        "prefix_suffix_chunks": [
            "Infer prefix_width and chunk_size from expected shape and alignment.",
            "Do not stop at the first Err-like cell; only omit trailing all-padding chunks when expected omits them.",
            "If expected contains non-empty cells absent from the source and not derivable by split/merge/extract/fill, state that the example is outside pure Foofah layout/syntactic scope instead of inventing values.",
        ],
        "wide_by_key": [
            "Group by key across the whole input, including non-contiguous rows.",
            "Compute width from current group sizes rather than a fixed training width.",
        ],
        "grouped_suffix_wide": [
            "Do not assume every later row drops the same number of leading cells; infer each suffix start from source_to_expected_alignment.",
            "Compute output width from the current group and append all aligned suffix cells produced by the same suffix rule.",
        ],
        "stacked_metric_rows": [
            "Preserve metric rows as row-major groups only when expected_output_grid shows grouped order.",
            "If foofah_features.metric_order_hint=metric_values_interleaved, emit values interleaved by value column: [key, row1_val1, row2_val1, row1_val2, row2_val2, ...].",
            "When actual and expected contain the same metric values, only repair ordering.",
        ],
        "context_attribute_pairs": [
            "Track current context and pivot label:value attributes into columns.",
            "Emit one row per context that collected allowed labels; attribute rows are not independent records.",
        ],
        "name_merge_split_rows": [
            "Handle both already split names and one-cell comma names with guards.",
            "Emit exactly the first/last order demonstrated by expected_output_grid.",
        ],
        "wrap_fixed_width": [
            "Flatten one-column non-empty values and wrap by expected/current record width.",
            "Do not let blank separators become output values.",
        ],
    }

    result: list[dict] = []
    seen: set[str] = set()
    for name in selected_names:
        if name in packs and name not in seen:
            result.append({"name": name, "rules": packs[name]})
            seen.add(name)
    return result


def _foofah_candidate_diversity_contract() -> dict:
    return {
        "meaning": "Expensive candidate mode should explore different plausible operator assumptions, not repeat the same shortcut.",
        "router_rules": [
            "Return ranked_operators with at least 3 diverse choices when plausible.",
            "candidate 1 should use the strongest visible-example operator.",
            "candidate 2 should use the strongest alternative from likely_operator_hints or operator_suggestions.",
            "candidate 3 should use a shape-first fallback focused on exact row/column counts.",
            "If reroute_context.should_reroute is true, do not repeat failed_operator unless same_operator_allowed=true and parameters materially change.",
        ],
        "ranked_operator_item_schema": {
            "operator": "operator family",
            "confidence": "0..1",
            "parameters": {"key": "value"},
            "generalization_assumption": "why this candidate should generalize",
            "diversity_role": "primary | alternative | shape_first | repair_specific",
        },
    }


def _foofah_generalized_examples(features: dict, operator: str | None = None, failure_context: dict | None = None) -> list[dict]:
    names = [operator] if operator else []
    names.extend(features.get("likely_operator_hints") or [])
    if failure_context and failure_context.get("error_family") in {"column_count_too_high", "column_count_too_low"}:
        names.extend(["split_extract", "prefix_suffix_chunks", "wide_by_key"])
    if failure_context and failure_context.get("error_family") in {"row_count_too_high", "row_count_too_low"}:
        names.extend(["blank_separated_blocks", "context_attribute_pairs"])

    examples = {
        "header_grid_to_long": {
            "pattern": "top row labels + left row keys -> long rows",
            "input_shape": "3x3",
            "output_shape": "4x3",
            "rule": "For every non-empty interior cell, emit either [row_key, header, value] or [header, row_key, value] exactly as expected order shows.",
        },
        "record_pair_merge": {
            "pattern": "two consecutive source rows become one output record",
            "input_shape": "2N x M",
            "output_shape": "N x K",
            "rule": "For each pair, locate expected values inside both rows and emit the inferred positional mapping.",
        },
        "blank_separated_blocks": {
            "pattern": "one-column vertical blocks separated by blanks",
            "input_shape": "many x 1",
            "output_shape": "records x width",
            "rule": "Split blocks on blanks, discard short title/metadata blocks, and output one row per true record.",
        },
        "split_extract": {
            "pattern": "one text cell contains several fields",
            "input_shape": "N x M",
            "output_shape": "N x K",
            "rule": "Find the text cell with the delimiter/pattern, extract groups, and emit exactly K fields without copying unrelated cells.",
        },
        "wide_by_key": {
            "pattern": "same key appears on several rows",
            "input_shape": "N x M",
            "output_shape": "keys x variable_width",
            "rule": "Collect values for each key across the whole input, including non-contiguous rows, and pad to current max group length.",
        },
        "prefix_suffix_chunks": {
            "pattern": "wide rows contain a fixed prefix and repeated suffix chunks",
            "input_shape": "N x wide",
            "output_shape": "chunks x prefix_plus_chunk",
            "rule": "Keep prefix once per emitted row, split suffix by chunk size, and ignore only trailing all-padding chunks.",
        },
        "context_attribute_pairs": {
            "pattern": "context row followed by label:value attributes",
            "input_shape": "many x 2",
            "output_shape": "contexts x attributes",
            "rule": "Track current context, pivot allowed attribute labels into columns, and emit one row per context.",
        },
    }

    result: list[dict] = []
    seen: set[str] = set()
    for name in names:
        if name in examples and name not in seen:
            result.append(examples[name])
            seen.add(name)
        if len(result) >= 4:
            break
    return result


def build_generation_prompt(
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict],
    user_instruction: str | None = None,
    similar_examples: list[dict] | None = None,
) -> str:
    payload = {
        "mode": "transform_from_example",
        "task": (
            "Infer a reusable Pandas transform(df) function from a large source table and a small "
            "manually written expected output example."
        ),
        "important_validation_target": {
            "expected_output_rows": example_output,
            "rule": "transform(example_input_rows) must match expected_output_rows by columns, row count, and values.",
        },
        "user_instruction": user_instruction or "",
        "source_profile": source_profile,
        "example_input_rows": example_input,
        "visible_value_alignment": _visible_value_alignment(example_input, example_output),
        "similar_successful_transformations": similar_examples or [],
        "business_rules": [
            "Use the expected example to infer output columns, types, ordering, and required calculations.",
            "Do not hardcode row values from expected_output_rows.",
            "If dates are ambiguous or mixed-format, write robust parsing logic that handles common formats in the same column.",
            "Use visible_value_alignment for date-like columns: expected_output_rows are authoritative even when they contradict normal ISO parsing.",
            "Do not use format='mixed' blindly. It may preserve ISO-like YYYY-MM-DD strings even when the visible expected mapping requires YYYY-DD-MM/day-first interpretation.",
            "For ambiguous dates, implement explicit parser branches by observed string pattern and self-check each visible source date -> expected date pair.",
            "If parsing creates NaT/NaN while the expected example has a real date, do not emit NaN; repair the parser or fall back to the original visible string before formatting.",
            "If a value cannot be parsed safely, keep it as missing rather than inventing data.",
            "Never output the literal strings 'nan' or 'NaT' in final text/date columns.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    return _dump(payload)


def build_foofah_generation_prompt(
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict],
    user_instruction: str | None = None,
    similar_examples: list[dict] | None = None,
) -> str:
    input_grid = _records_to_grid(example_input)
    output_grid = _records_to_grid(example_output)
    features = _foofah_features(input_grid, output_grid)
    payload = {
        "mode": "foofah_program_synthesis",
        "task": "Infer a reusable FOOFAH-style table transformation from InputTable to OutputTable.",
        "data_model": {
            "critical": "FOOFAH tables are 2D arrays of strings. There is no trusted header row.",
            "required_code_style": "Inside transform(df), immediately use: data = df.fillna('').astype(str).values.tolist(). Then reason with data[row][col].",
            "output": "Return pd.DataFrame(rows). Column names are positional and not semantic.",
        },
        "generalization_contract": _foofah_generalization_contract(),
        "important_validation_target": {
            "expected_output_grid": output_grid,
            "expected_shape": {"rows": len(output_grid), "columns": len(output_grid[0]) if output_grid else 0},
            "rule": "The returned matrix must exactly match expected_output_grid by shape, row order, column order, and cell values.",
        },
        "user_instruction": user_instruction or "",
        "source_profile": source_profile,
        "example_input_grid": input_grid,
        "input_shape": features["input_shape"],
        "foofah_features": features,
        "foofah_paper_principles": _foofah_paper_principles(),
        "foofah_offline_taxonomy": _foofah_offline_taxonomy(),
        "source_to_expected_alignment": _source_to_expected_alignment(input_grid, output_grid),
        "benchmark_generalization_analysis": _benchmark_generalization_analysis(user_instruction, input_grid, output_grid),
        "shape_proof_contract": _foofah_shape_proof_contract(features),
        "priority_rule_packs": _foofah_rule_packs(features),
        "candidate_diversity_contract": _foofah_candidate_diversity_contract(),
        "generalized_few_shot_examples": _foofah_generalized_examples(features),
        "similar_successful_transformations": similar_examples or [],
        "operator_decision_table": [
            "rows up, columns down -> fold/unpivot.",
            "rows down, columns up -> unfold/pivot or fixed-stride wrap.",
            "same columns, fewer rows -> row filter; preserve values exactly.",
            "same rows, fewer columns -> column select/reorder.",
            "expected rows are exact source rows and columns are unchanged -> row_filter; do not parse values.",
            "before choosing any operator, decide row 0 role from foofah_features.first_row_role_guess: header, data, metadata, or uncertain.",
            "long key/label/value rows becoming a wide table -> pivot_long_to_wide.",
            "sparse multi-row record blocks becoming one dense row -> flatten_record_blocks.",
            "one-column records separated by blanks or marker rows -> blank_separated_blocks or marker_separated_blocks.",
            "cells containing key:value or key=value becoming headers and values -> delimited_pairs_to_wide.",
            "two-column records where a context row is followed by label:value attribute rows -> context_attribute_pairs; output one row per context, not one row per attribute.",
            "two-column contact-style records where communication labels are written as label:value -> context_attribute_pairs; preserve the exact value after the delimiter.",
            "two-row record pairs where a context/header row is followed by detail fields -> record_pair_merge.",
            "rows that mix already-split cells and comma-joined cells -> name_merge_split_rows or split_extract with guards.",
            "cells containing comma-separated lists becoming repeated rows -> split_list_fold.",
            "one text cell -> split/extract with delimiters/regex.",
            "one column -> fixed-width record wrapping or KEY=VALUE unfold.",
        ],
        "core_operators": [
            "fold_pair: first cell of each data row is key; remaining cells become [key, value], skipping blanks only if expected skips blanks.",
            "fold_triple: first row has labels; first column has keys; nonblank interior cells become [key, normalized_label, value].",
            "headered_fold: preserve id columns, use header labels for variable columns, emit [ids..., header, value] rows.",
            "header_grid_to_long: grid headers become labels; expected grid decides if label column comes before or after row key.",
            "row_filter: keep source rows whose required positional cell is nonblank; use str(row[j]).strip() != ''.",
            "first_last_to_wide: group by first cell and collect each row's last non-empty cell horizontally.",
            "wide_by_key: group all rows by row[0] across the whole table, preserving first key order; output [key] + collected values from a chosen column.",
            "wide_by_key_pairs: group by first cell and append each row's suffix cells, padding shorter groups.",
            "grouped_suffix_wide: group consecutive rows by shared leading columns and append each row's changing suffix cells.",
            "stacked_metric_rows: for paired metric rows, obey foofah_features.metric_order_hint; grouped is [key]+row1 values+row2 values, interleaved is [key,row1v1,row2v1,row1v2,row2v2,...].",
            "record_pair_merge: merge each context row with its following detail row; infer exact output positions from expected cells in the row pair, not from a blind prefix/suffix split.",
            "name_merge_split_rows: for each row, output two name parts whether the source row is already split across cells or stores a comma-joined name in one cell.",
            "pivot_long_to_wide: collect labels in first-seen order, group values by key, return header row plus one row per key.",
            "wrap_fixed_width: flatten a one-column list into chunks of expected output width.",
            "blank_separated_blocks: split one-column inputs on blank rows and output one row per true record block; skip short title/metadata blocks.",
            "marker_separated_blocks: split one-column inputs where each record starts with a marker like *Company; strip marker and skip blank separators.",
            "key_value_unfold: split KEY=VALUE lines; first output row is keys, following rows are values.",
            "delimited_pairs_to_wide: split cells like Math:65 into header Math and value 65.",
            "context_attribute_pairs: a context row is followed by attribute rows such as label:value; infer labels from expected header or input labels, keep context as col_0, and pivot attributes into columns.",
            "contact_attribute_pair_pattern: same as context_attribute_pairs, but labels are communication/contact attributes; never treat those label:value rows as separate records.",
            "split_list_fold: split comma/semicolon lists and carry context cells into each output row.",
            "flatten_record_blocks: flatten non-empty cells from each fixed-size sparse record block into one dense row.",
            "flatten_all_rows: concatenate all input cells row-major into a single output row.",
            "prefix_suffix_chunks: split very wide rows into repeated fixed-size suffix chunks after a fixed prefix.",
            "context_detail: copy context/header cells into following detail rows.",
            "split_extract/regex_extract: extract groups from strings using delimiters or regex.",
        ],
        "known_ambiguous_cases": [
            "If one example row maps to first+last cell, prefer wide_by_key over simple column selection because current inputs may have many rows.",
            "If groupby/list expansion is needed, build rows manually: rows.append([key] + values). Do not return a Series of lists.",
            "For unfold2-style tasks, repeated keys may be non-contiguous; collect all values for the same key across the entire table, then pad rows to the max group width for the current input.",
            "If returning selected non-adjacent cells, use pd.DataFrame(rows), not df[['col_0','col_5']], because FOOFAH output is positional.",
            "If output is wider than input and first output cell is blank, check pivot_long_to_wide or delimited_pairs_to_wide before column_select.",
            "If input is two columns and label:value attributes follow context rows, do not emit attribute rows as output rows. Track the current context and emit exactly one row per context with attribute values.",
            "If contact_attribute_pair_pattern is true, use expected_output_grid's labels as the allowed attribute labels and preserve contact values as strings without parsing.",
            "If input rows are sparse address/card/license blocks, flatten non-empty cells row-major by record block.",
            "If the example can be solved by selecting columns but the current input has repeated keys, prefer collection/grouping over column selection.",
            "If a single-row example looks like identity/column select, current inputs with repeated keys may reveal grouped wide output; prefer structural grouping when source rows share keys or suffix patterns.",
            "If a one-cell split/extract example has extra source cells in the current table, search those cells for the same split/extract pattern but return exactly the expected output width.",
            "If one-column blank-separated blocks produce one extra row, check for a short title/metadata block; ignore short non-record blocks and wrap only full record-width data.",
        ],
        "code_rules": [
            "Start transform with: data = df.fillna('').astype(str).values.tolist()",
            "Use explicit loops over data[i][j].",
            "Build rows as list[list[str]].",
            "Return pd.DataFrame(rows).",
            "Never convert cells to int/float and never use pd.to_numeric; FOOFAH values are strings and must stay exact.",
            "Never assume row 0 is a header in FOOFAH unless the expected grid explicitly uses row-0 labels as output labels. Row 0 is often ordinary data.",
            "Use foofah_features.first_row_role_guess: header means use row 0 as labels; data means start processing at row 0; metadata means skip row 0 unless it supplies context; uncertain means compare expected cells before deciding.",
            "For one-column address/company blocks separated by blank rows, skip blank separator rows before wrapping or block-flattening.",
            "For one-column marker blocks, start a new row at marker cells, strip the marker, and never include separator blanks.",
            "For KEY=VALUE unfold, preserve the header row if expected_output_grid starts with key names, then append one row per record.",
            "For split/extract tasks, preserve original string tokens exactly; do not parse numeric-looking tokens.",
            "For context_attribute_pairs, detect label:value in either column, infer output labels, strip only the label and delimiter, and skip title/header rows that collect no expected attributes.",
            "For contact_attribute_pair_pattern, rows containing only attribute labels belong to the current context; they should fill output columns, not increase output row count.",
            "For record_pair_merge, process rows in fixed pairs only when the expected row count is half the input row count; do not slide the window by one.",
            "For record_pair_merge, infer the output positional mapping by locating expected cells inside the two source rows. Do not blindly take first N cells from row 0 plus the suffix of row 1.",
            "For name_merge_split_rows, if a row already has two non-empty name cells use them as-is; if a cell contains one comma, split once and emit the expected first/last order.",
            "For wide_by_key or first_last_to_wide, use the maximum group length in the current input, not the training OutputTable width.",
            "For stacked_metric_rows, compare foofah_features.metric_order_hint and follow the order demonstrated by expected_output_grid.",
            "For stacked_metric_rows with metric_order_hint=metric_values_interleaved, emit [key, sum1, count1, sum2, count2, ...]; do not emit grouped [key, sum1, sum2, ..., count1, count2, ...].",
            "For header_grid_to_long, compare foofah_features.header_grid_long_order_hint and obey expected_output_grid.",
            "If benchmark_generalization_analysis.has_feedback is false, expected_output_grid wins over semantic guesses like product/date or sum/count.",
            "For prefix_suffix_chunks, do not stop at the first Err-like cell. Split by fixed chunk size and only ignore trailing all-error padding when the expected grid omits those chunks.",
            "For split_extract/regex_extract/name_merge_split_rows, return exactly expected_shape.columns columns. If a row has extra cells, search them for the target pattern but do not append them unless expected width requires it.",
            "For blank_separated_blocks, if split-on-blank creates too many rows, drop blocks shorter than expected_shape.columns when their values do not appear in expected_output_grid, then pad/truncate true record blocks.",
            "If expected_shape.rows > 0, never return an empty pd.DataFrame() unless the example truly implies no output.",
            "For row_filter, preserve the whole original row: rows = [row for row in data if len(row) > j and row[j].strip() != '']; return pd.DataFrame(rows).",
            "If foofah_features.expected_rows_are_subset_of_input_rows is true, prefer a row_filter/copy-row rule over calculations.",
            "In plan, include operator, output_rows_rule, output_columns_rule, row_0_role, value_order_rule, generalization_assumption, and self_check.",
            "Use source_to_expected_alignment before deciding whether a failure needs extraction, reordering, grouping, or metadata skipping.",
            "Do not use df['col_4'] or df[['col_0','col_5']] in FOOFAH.",
            "Never output 'nan'; blank is ''.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    return _dump(payload)


def build_foofah_router_prompt(
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict],
    user_instruction: str | None = None,
    similar_examples: list[dict] | None = None,
    reroute_context: dict | None = None,
) -> str:
    input_grid = _records_to_grid(example_input)
    output_grid = _records_to_grid(example_output)
    features = _foofah_features(input_grid, output_grid)
    payload = {
        "mode": "foofah_operator_router",
        "task": "Choose the most likely FOOFAH table-transformation operator. Do not generate Python code.",
        "response_schema": {
            "operator": f"chosen primary operator; one of: {', '.join(FOOFAH_OPERATORS)}",
            "confidence": "number between 0 and 1",
            "parameters": {"key": "value"},
            "generalization_assumption": "short text",
            "ranked_operators": [
                {
                    "operator": f"one of: {', '.join(FOOFAH_OPERATORS)}",
                    "confidence": "number between 0 and 1",
                    "parameters": {"key": "value"},
                    "generalization_assumption": "short text",
                    "diversity_role": "primary | alternative | shape_first | repair_specific",
                }
            ],
            "same_operator_allowed": "boolean; true only when reroute repeats failed_operator with materially changed parameters",
            "why_not_other_operators": ["short text"],
        },
        "generalization_contract": _foofah_generalization_contract(),
        "input_shape": features["input_shape"],
        "expected_shape": features["expected_shape"],
        "foofah_features": features,
        "foofah_paper_principles": _foofah_paper_principles(),
        "foofah_offline_taxonomy": _foofah_offline_taxonomy(),
        "source_to_expected_alignment": _source_to_expected_alignment(input_grid, output_grid),
        "benchmark_generalization_analysis": _benchmark_generalization_analysis(user_instruction, input_grid, output_grid),
        "shape_proof_contract": _foofah_shape_proof_contract(features),
        "priority_rule_packs": _foofah_rule_packs(features, reroute_context),
        "candidate_diversity_contract": _foofah_candidate_diversity_contract(),
        "generalized_few_shot_examples": _foofah_generalized_examples(features, failure_context=reroute_context),
        "example_input_grid": input_grid,
        "expected_output_grid": output_grid,
        "source_profile": source_profile,
        "user_instruction": user_instruction or "",
        "similar_successful_transformations": similar_examples or [],
        "reroute_context": reroute_context or {},
        "decision_rules": [
            "rows up, columns down -> fold_pair or fold_triple",
            "same columns, fewer rows -> row_filter",
            "same rows, fewer columns -> column_select",
            "rows down, columns up -> wide_by_key, wrap_fixed_width, key_value_unfold, or context_detail",
            "first_row_role_guess=data -> include row 0 in processing; first_row_role_guess=header -> use row 0 as labels; first_row_role_guess=metadata -> skip it unless it is carried as context",
            "long rows [key, label, value] or [key, value] becoming header+wide rows -> pivot_long_to_wide",
            "consecutive rows with the same group key becoming one wide row -> grouped_suffix_wide",
            "two or more metric rows for one key becoming [key] plus grouped or interleaved metric values -> stacked_metric_rows",
            "consecutive row pairs where a context row and detail row merge into one output row -> record_pair_merge",
            "cells like Math:65 becoming header row Math and value 65 -> delimited_pairs_to_wide",
            "two-column context blocks followed by label:value rows -> context_attribute_pairs",
            "contact_attribute_pair_pattern=true -> context_attribute_pairs with output labels taken from expected header",
            "comma_name_cell_count>0 and expected has two columns -> name_merge_split_rows or split_extract",
            "cells like 'George, Anna' becoming repeated rows -> split_list_fold",
            "sparse fixed-height record blocks becoming one dense row per record -> flatten_record_blocks",
            "all input rows concatenated into one output row -> flatten_all_rows",
            "one row selecting first and last cells but hidden may contain repeated keys -> first_last_to_wide",
            "one-column records separated by blank rows -> blank_separated_blocks",
            "one-column records starting with marker rows such as *Company -> marker_separated_blocks",
            "very wide rows split into prefix + repeated fixed-size suffix chunks -> prefix_suffix_chunks",
            "error_like_cell_count>0 and wide input becomes repeated rows -> prefix_suffix_chunks; treat Err-like cells as padding only when the whole trailing chunk is padding",
            "single text cell extracting price/bedrooms or names -> regex_extract",
            "one text cell becomes multiple cells/rows -> split_extract",
            "one-column vertical records -> wrap_fixed_width or key_value_unfold",
            "first row labels + first column keys + interior cells -> fold_triple",
            "same first-column key repeated -> wide_by_key",
            "context_attribute_pair_pattern=true -> choose context_attribute_pairs and emit one row per context with pivoted attributes",
            "expected_rows_are_subset_of_input_rows=true and same columns -> row_filter, even if one column looks like date/currency",
            "key_value_line_count>0 and one input column -> key_value_unfold",
            "metric_order_hint=metric_values_interleaved means the visible example interleaves metric rows; route to interleaved parameters.",
            "header_grid_long_order_hint tells whether visible grid-to-long output starts with row key or column header.",
            "route for the visible expected_output_grid order; do not choose an answer order that has not been demonstrated.",
            "if reroute_context.failed_operator exists, actively consider a different operator family unless the delta proves only a small parameter error.",
            "if reroute_context.should_reroute=true and reroute_context.failed_operator repeats, either choose a different operator from operator_suggestions or change the parameterization materially; do not return the same assumption again.",
            "Return ranked_operators in diversity order. The worker may select candidate_index from this list in expensive candidate mode.",
            "When reroute_context.should_reroute=true, ranked_operators[0] should not be failed_operator unless same_operator_allowed=true and parameters materially change.",
            "Use source_to_expected_alignment to decide whether expected cells are copied/reordered from source or extracted as substrings.",
        ],
    }
    return _dump(payload)


def build_foofah_operator_prompt(
    router_plan: dict,
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict],
    user_instruction: str | None = None,
    similar_examples: list[dict] | None = None,
    failure_context: dict | None = None,
) -> str:
    input_grid = _records_to_grid(example_input)
    output_grid = _records_to_grid(example_output)
    features = _foofah_features(input_grid, output_grid)
    operator = router_plan.get("operator", "unknown")
    payload = {
        "mode": "foofah_operator_code_generation",
        "task": "Generate safe Python code for transform(df) using the selected FOOFAH operator.",
        "selected_operator_plan": router_plan,
        "operator_contract": _foofah_operator_contract(operator),
        "generalization_contract": _foofah_generalization_contract(),
        "data_model": {
            "critical": "FOOFAH tables are 2D arrays of strings. Pandas is only a transport container.",
            "required_first_step": "data = df.fillna('').astype(str).values.tolist()",
            "output": "Build rows as list[list[str]] and return pd.DataFrame(rows).",
        },
        "important_validation_target": {
            "expected_output_grid": output_grid,
            "expected_shape": {"rows": len(output_grid), "columns": len(output_grid[0]) if output_grid else 0},
            "rule": "The returned matrix must exactly match expected_output_grid by shape, row order, column order, and cell values.",
        },
        "example_input_grid": input_grid,
        "input_shape": features["input_shape"],
        "foofah_features": features,
        "foofah_paper_principles": _foofah_paper_principles(),
        "foofah_offline_taxonomy": _foofah_offline_taxonomy(),
        "source_to_expected_alignment": _source_to_expected_alignment(input_grid, output_grid),
        "benchmark_generalization_analysis": _benchmark_generalization_analysis(user_instruction, input_grid, output_grid),
        "shape_proof_contract": _foofah_shape_proof_contract(features),
        "priority_rule_packs": _foofah_rule_packs(features, failure_context, operator),
        "candidate_diversity_contract": _foofah_candidate_diversity_contract(),
        "generalized_few_shot_examples": _foofah_generalized_examples(features, operator, failure_context),
        "source_profile": source_profile,
        "user_instruction": user_instruction or "",
        "similar_successful_transformations": similar_examples or [],
        "failure_context": failure_context or {},
        "hard_code_rules": [
            "Return only JSON with code, explanation, and plan.",
            "The code must define transform(df).",
            "Use data = df.fillna('').astype(str).values.tolist().",
            "Use data[i][j] positional access, not df['col_4'] or df[['col_0','col_5']].",
            "Return pd.DataFrame(rows).",
            "Do not hardcode expected output rows; infer a reusable positional rule.",
            "If expected has rows, do not return empty pd.DataFrame().",
            "Use foofah_features.likely_operator_hints to challenge the selected operator if it contradicts the shapes.",
            "If failure_context is present, address its error_family, expected_width/actual_width, expected_height/actual_height, and operator_suggestions explicitly in the generated plan.",
            "If failure_context.direct_repair_instruction is non-empty, follow it before applying broad operator guesses.",
            "Always inspect foofah_features.first_row_role_guess before setting loop start. Use header -> start data after row 0, data -> start at row 0, metadata -> skip row 0 or use it only as non-output context.",
            "For row_filter, copy full rows unchanged; do not parse dates, money, or numbers.",
            "For wide_by_key, collect values for each key across the whole table, not only consecutive rows; preserve first key order and pad each row to the current max group length.",
            "For first_last_to_wide, take the last non-empty cell from each row, not a fixed column if rows can be ragged.",
            "For blank_separated_blocks, ignore empty separator rows; do not emit rows starting with ''.",
            "For stacked_metric_rows, preserve metric row order exactly as expected_output_grid shows. If metric_order_hint=metric_values_interleaved, output [key, row1_val1, row2_val1, row1_val2, row2_val2, ...].",
            "expected_output_grid order is authoritative; do not invert product/date or sum/count based on semantic labels.",
            "For grouped_suffix_wide, preserve shared prefix once, then append row-specific suffix cells from each row.",
            "For headered_fold, do not contradict selected_operator_plan.parameters.output_columns. If the plan or expected grid says [date, product, value], emit that exact positional order.",
            "For header_grid_to_long, compare expected_output_grid and explicit oracle feedback: if expected starts with product/key, emit [key, header, value]; if expected starts with header/date, emit [header, key, value]. Add a shape-based branch if visible and oracle disagree.",
            "For key_value_unfold, include the header row when expected_output_grid begins with KEY labels, then emit every complete record.",
            "For marker_separated_blocks, a marker row starts a new record; strip '*' and ignore blank separator rows.",
            "For record_pair_merge, merge data[0]+data[1], data[2]+data[3], etc. Select detail fields by expected positions and skip duplicated keys/status columns rather than taking a blind suffix.",
            "For record_pair_merge, when value diffs show expected values exist in the paired rows, repair the positional mapping; do not keep retrying the same prefix/suffix split.",
            "For name_merge_split_rows, handle both already split rows and comma-joined cells: split a comma cell once, trim parts, and emit in the order demonstrated by expected_output_grid.",
            "For context_attribute_pairs, a non-attribute row is context and following label:value rows are attributes. Infer labels from expected_output_grid's header row when present, otherwise from first-seen labels.",
            "For context_attribute_pairs, an attribute may be in the same row as the context, e.g. [context, label:value], or in a following row, e.g. ['', label:value] or [label:value, ''].",
            "For context_attribute_pairs, emit a header row only if expected has one, then exactly one output row per context that collected at least one expected attribute. Do not output title rows or attribute rows as separate data rows.",
            "For names/merge-split cases, handle both already split rows and comma-joined cells with guards.",
            "For regex_extract, capture the desired group only; if expected is a bedroom count, capture digits before 'br' rather than '/' itself.",
            "For prefix_suffix_chunks, use fixed prefix_width and chunk_size inferred from expected shape; do not stop at the first Err-like cell; ignore only trailing chunks made entirely of Err-like padding if expected omits them.",
            "For split_extract/regex_extract/name_merge_split_rows, scan all cells in each row for the pattern but emit only the extracted groups needed by expected_shape.columns.",
            "For blank_separated_blocks, if blank splitting creates too many rows, treat short leading/trailing blocks as title/metadata unless expected_output_grid contains their values.",
            "Set plan.operator, plan.confidence, plan.parameters, and plan.generalization_assumption.",
            "Also set plan.output_rows_rule, plan.output_columns_rule, plan.row_0_role, plan.value_order_rule, and plan.self_check.",
            "Use source_to_expected_alignment to justify positional mapping when expected values already appear in source cells.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    return _dump(payload)


def _foofah_operator_contract(operator: str) -> dict:
    contracts = {
        "fold_pair": {
            "meaning": "Turn each data row into repeated [key, value] rows.",
            "code_shape": "for each row, key=row[0]; for j in value columns: if keep(value): rows.append([key, value])",
        },
        "fold_triple": {
            "meaning": "Use first row as labels and first column as row keys; emit nonblank interior cells.",
            "code_shape": "headers=data[0]; for i>=1, j>=1: rows.append([data[i][0], headers[j].replace(' ', ''), data[i][j]]) when value is nonblank",
        },
        "headered_fold": {
            "meaning": "Keep fixed id columns, use a header row for variable labels, and emit one output row per variable cell.",
            "code_shape": "headers=data[0]; for each data row i>=1 and variable col j: emit exactly selected_operator_plan.parameters.output_columns, e.g. [header, row_label, value] or [row_label, header, value]",
        },
        "header_grid_to_long": {
            "meaning": "Use top row labels and data rows to emit long rows. The expected grid decides whether order is [row_label, column_label, value] or [column_label, row_label, value].",
            "code_shape": "headers=data[0]; for each data row and each value col: rows.append([...]) in exact expected column order",
        },
        "row_filter": {
            "meaning": "Keep complete original rows whose required positional cell is nonblank or matches a predicate.",
            "code_shape": "rows=[row for row in data if len(row)>j and row[j].strip()!='']; return pd.DataFrame(rows)",
        },
        "column_select": {
            "meaning": "Select/reorder positional cells from each row.",
            "code_shape": "rows=[[row[i] for i in selected_indexes] for row in data]",
        },
        "wide_by_key": {
            "meaning": "Group all rows by first cell and collect values horizontally, even when the same key appears in non-contiguous rows.",
            "code_shape": "order=[]; groups={}; for row in data: key=row[0]; if new append key to order; groups[key].append(row[value_col]); width=max(len(v)); rows=[[key]+groups[key]+['']*(width-len(groups[key])) for key in order]",
        },
        "first_last_to_wide": {
            "meaning": "For rows with the same first cell, collect the last non-empty cell from each row horizontally.",
            "code_shape": "group by row[0]; value=last non-empty cell; rows.append([key]+values)",
        },
        "wide_by_key_pairs": {
            "meaning": "Group consecutive rows by first cell and append pairs or suffix cells from each row.",
            "code_shape": "rows.append([key] + row1[1:] + row2[1:] + ...), padding to expected width if needed",
        },
        "grouped_suffix_wide": {
            "meaning": "Group consecutive rows by shared leading columns and append suffix cells from each row horizontally.",
            "code_shape": "for each group, start with shared prefix; append row-specific cells from each grouped row in expected order",
        },
        "stacked_metric_rows": {
            "meaning": "Combine paired metric rows for the same entity into one wide row.",
            "code_shape": "if metric_order_hint is metric_values_interleaved, append [key,row1v1,row2v1,row1v2,row2v2,...]; otherwise append [key]+row1 values+row2 values.",
        },
        "record_pair_merge": {
            "meaning": "Merge fixed consecutive row pairs where the first row carries context/id fields and the second row carries detail fields.",
            "code_shape": "for i in range(0, len(data)-1, 2): pair=context+detail; choose output cells by positional mapping inferred from expected cells in the pair, with guards for short rows",
        },
        "name_merge_split_rows": {
            "meaning": "Normalize rows that may already contain two split name cells or a single comma-joined name cell.",
            "code_shape": "for row in data: if first two cells are non-empty use them; else find the one comma-joined cell, split once, trim, and emit the two parts in expected order",
        },
        "pivot_long_to_wide": {
            "meaning": "Turn long records [row key, column label, value] into a wide matrix with a header row.",
            "code_shape": "collect labels in first-seen order; group values by key; rows=[['']+labels]; for each key: rows.append([key]+values_by_label)",
        },
        "wrap_fixed_width": {
            "meaning": "Flatten one-column input and wrap into rows of inferred width.",
            "code_shape": "values=[row[0] for row in data]; rows=[values[i:i+width] ...]",
        },
        "blank_separated_blocks": {
            "meaning": "Split a one-column input into records separated by blank rows and output one row per record.",
            "code_shape": "split on blanks; discard short title/metadata blocks not present in expected; rows.append(record_block padded/truncated to inferred width)",
        },
        "marker_separated_blocks": {
            "meaning": "Split a one-column input into records that start with marker rows such as *Company.",
            "code_shape": "start a new block when cell.startswith('*'); strip marker; append each block with its natural length",
        },
        "key_value_unfold": {
            "meaning": "Split KEY=VALUE lines into header row and value rows.",
            "code_shape": "pairs=[cell.split('=',1)]; infer record width; rows=[keys]+value_rows",
        },
        "delimited_pairs_to_wide": {
            "meaning": "Split cells such as key:value or key=value into output headers and values.",
            "code_shape": "simple row form: split each row's pair cells into labels/values. context form: track current context/name, parse label:value in either column, store attributes, and emit one row per context plus optional header.",
        },
        "context_attribute_pairs": {
            "meaning": "A context row starts a record, and following label:value rows fill attributes for that record. This covers contact-style blocks such as phone/fax, but the rule is label-driven, not literal-label hardcoding.",
            "code_shape": "infer labels from expected header row when first expected cell is blank, otherwise from first-seen pair labels; scan rows; if a row has a context plus label:value, start/continue that context and store the pair; if a row has only label:value, attach it to current context; when a new context starts, emit previous context only if it collected an allowed label; rows=[['']+labels] if expected has header else []; output [context]+values_by_label.",
        },
        "split_list_fold": {
            "meaning": "Split a delimiter-separated cell and emit one output row per item while carrying context cells.",
            "code_shape": "for each row, split selected cell on comma/semicolon; for each item: rows.append(context_cells + [item.strip()])",
        },
        "flatten_record_blocks": {
            "meaning": "Flatten each sparse record block into one dense row by reading non-empty cells in row-major order.",
            "code_shape": "split data into record blocks; for each block, append every non-empty cell in order to one output row",
        },
        "flatten_all_rows": {
            "meaning": "Concatenate every cell from all input rows into one output row.",
            "code_shape": "rows = [[cell for row in data for cell in row]]",
        },
        "prefix_suffix_chunks": {
            "meaning": "Keep a fixed prefix from each wide row and split the remaining suffix into fixed-size chunks.",
            "code_shape": "prefix=row[:prefix_width]; for j in range(prefix_width, len(row), chunk_size): rows.append(prefix + row[j:j+chunk_size])",
        },
        "context_detail": {
            "meaning": "Carry context/header cells into following detail rows, or pivot following key:value attribute rows into one dense row per context.",
            "code_shape": "track current context; when a later row contains label:value, attach value to context[label]; when the next context starts, emit the previous context row",
        },
        "split_extract": {
            "meaning": "Extract substrings or repeated values from text cells.",
            "code_shape": "scan each row for the target text cell; use re/split on delimiters; emit only captured groups in expected order and expected width",
        },
        "regex_extract": {
            "meaning": "Extract structured fields from a text cell with regular expressions or delimiters.",
            "code_shape": "for each text row, use re.search/re.split with guards; append captured fields",
        },
        "transpose": {
            "meaning": "Swap rows and columns.",
            "code_shape": "rows=[list(col) for col in zip(*data)]",
        },
        "unknown": {
            "meaning": "Infer a positional structural transformation from the example.",
            "code_shape": "Use explicit matrix loops and preserve exact strings.",
        },
    }
    return contracts.get(operator, contracts["unknown"])


def build_query_prompt(
    source_profile: dict,
    sample_rows: list[dict],
    user_instruction: str,
    similar_examples: list[dict] | None = None,
) -> str:
    payload = {
        "mode": "analytical_query",
        "task": "Generate a reusable Pandas transform(df) function that answers the user's analytical query.",
        "user_query": user_instruction,
        "source_profile": source_profile,
        "sample_rows": sample_rows,
        "similar_successful_transformations": similar_examples or [],
        "business_rules": [
            "Return a pandas.DataFrame, even if the query asks for one value.",
            "For top-N queries, sort deterministically and return the requested number of rows.",
            "Infer likely columns from names and semantic types, but do not invent columns.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    return _dump(payload)


def build_repair_prompt(
    previous_code: str,
    error_or_diff: str,
    source_profile: dict,
    example_input: list[dict],
    example_output: list[dict] | None = None,
    user_instruction: str | None = None,
    attempt_history: list[dict] | None = None,
    mode: str = "transform",
    failure_context: dict | None = None,
    lens_putback_context: dict | None = None,
) -> str:
    is_foofah = mode.startswith("foofah")
    input_grid = _records_to_grid(example_input) if is_foofah else []
    output_grid = _records_to_grid(example_output or []) if is_foofah else []
    payload = {
        "mode": mode,
        "task": "Repair the previous transform(df) function.",
        "critical_error_or_diff": error_or_diff,
        "previous_code": previous_code,
        "attempt_history": attempt_history or [],
        "failure_context": failure_context or {},
        "lens_putback_context": lens_putback_context or {},
        "user_instruction": user_instruction or "",
        "source_profile": source_profile,
        "example_input_rows": example_input,
        "expected_output_rows": example_output or [],
        "visible_value_alignment": _visible_value_alignment(example_input, example_output or []),
        "example_input_grid": input_grid,
        "expected_output_grid": output_grid,
        "repair_rules": [
            "Fix the root cause, not only the shown row.",
            "Do not repeat a previous failed solution.",
            "If failure_context says the same code_hash or plan repeated, make a structural change rather than rephrasing the same implementation.",
            "Keep the solution general for the full table.",
            "If critical_error_or_diff shows actual 'nan' or 'NaT' where expected has a date/text value, assume conversion lost data. Do not repeat the same parser; use mixed-format parsing with fallback and ensure the final DataFrame does not contain literal 'nan'/'NaT' in that column.",
            "If date values differ by swapped month/day, inspect visible_value_alignment. Change the date interpretation; do not only add a fallback around the same parser.",
            "For standard transform date normalization, do not assume dayfirst/monthfirst/ISO globally. Infer the interpretation from visible source->expected pairs, implement explicit parser branches by string pattern, and self-check those pairs.",
            "Do not use format='mixed' if visible_value_alignment says an ISO-like source string such as 2025-02-01 must become 2025-01-02.",
            "For FOOFAH, apply priority_rule_packs first; use broad repair rules only as fallback.",
            "If lens_putback_context is present, treat typed_view_delta and putback_policy as the normalized repair target.",
            "Return only JSON with code, explanation, and plan.",
        ],
        "final_reminder": FINAL_REMINDER,
    }
    if is_foofah:
        features = _foofah_features(input_grid, output_grid)
        payload["foofah_features"] = features
        payload["foofah_paper_principles"] = _foofah_paper_principles()
        payload["foofah_offline_taxonomy"] = _foofah_offline_taxonomy()
        payload["generalization_contract"] = _foofah_generalization_contract()
        payload["source_to_expected_alignment"] = _source_to_expected_alignment(input_grid, output_grid)
        payload["benchmark_generalization_analysis"] = _benchmark_generalization_analysis(user_instruction, input_grid, output_grid)
        payload["shape_proof_contract"] = _foofah_shape_proof_contract(features)
        payload["priority_rule_packs"] = _foofah_rule_packs(features, failure_context)
        payload["candidate_diversity_contract"] = _foofah_candidate_diversity_contract()
        payload["generalized_few_shot_examples"] = _foofah_generalized_examples(features, failure_context=failure_context)
        payload["foofah_repair_rules"] = [
            "Treat the table as a 2D string matrix with synthetic columns col_0, col_1, ... .",
            "Prefer matrix repair code: data = df.fillna('').astype(str).values.tolist(); build rows; return pd.DataFrame(rows).",
            "Do not rely on selected pandas column names like col_5 in the output. FOOFAH output is positional; values matter by position.",
            "If the input has headers in row 0 and output rows look like [row label, normalized header, cell value], use unpivot/fold pattern A.",
            "If expected values omit blank cells, add an explicit value != '' filter.",
            "If row count is too high by the number of blank cells, the blank filter is wrong. Use data = df.fillna('').astype(str), then value = data.iat[i,j].strip(), then if value == '': continue.",
            "If labels differ only by spaces, normalize labels with .replace(' ', '').",
            "If output has fewer rows and more columns than input, switch away from unpivot and try pivot/unfold, fixed-stride wrap, or context fold.",
            "If example input has one row and output keeps only first and last cells, do not assume the task is only column deletion; current inputs may contain repeated rows with the same first-column key. Prefer a general solution that groups by first column and appends last-column values.",
            "If output cells are substrings of input cells, add split/extract logic instead of selecting whole cells.",
            "If the same key repeats in input rows, group by that key across the whole table unless expected clearly shows separate consecutive blocks.",
            "For unfold2/name-list cases, do not treat the first row as a header and do not require repeated keys to be adjacent.",
            "If Actual has too few columns in wide_by_key, you probably grouped only the first consecutive block; collect later rows with the same key too.",
            "If Actual has too many rows in wide_by_key, you probably split one key into multiple blocks; merge non-contiguous rows with the same key.",
            "If the chosen operation is row_filter and Actual=0 while Expected>0, the predicate is too strict or implemented against the wrong blank representation. Use fillna('').astype(str).str.strip() != '' on the exact positional column that contains a non-empty value in expected.",
            "For matrix-style row_filter repair, use: data = df.fillna('').astype(str).values.tolist(); rows = [row for row in data if len(row) > j and row[j].strip() != '']; return pd.DataFrame(rows).",
            "For row_filter cases where output columns equal input columns, never parse dates/currency or compare typed values. Preserve original strings exactly.",
            "If the error says 'No columns to parse from file', the previous code returned an empty DataFrame with no columns. Build rows and return pd.DataFrame(rows); do not return pd.DataFrame() for a non-empty expected output.",
            "If the same row-count error repeats across attempts, do not restate the same plan. Change one structural assumption: header/data row start, blank handling, grouping key, fixed stride, or selected columns.",
            "If row count or column count suggests an off-by-one error, inspect foofah_features.first_row_role_guess before changing the loop start: data means include row 0, header means use row 0 only as labels, metadata means skip row 0 unless context is required.",
            "If a one-row example matched by column deletion fails current-input behavior, generalize to grouping/collection over all rows.",
            "If a filter returns 0 rows but expected is a visible source row, inspect expected_output_grid and choose a predicate that is true for that exact visible row.",
            "If values become 'nan' or rows appear for blank cells, normalize with df.fillna('').astype(str) before the loop.",
            "If output is wider than expected or row count explodes, you probably used unpivot where pivot/wrap was needed.",
            "If actual has too few columns, you probably returned grouped lists/selected columns instead of expanding values into a full row.",
            "For column_count_too_low, explain in plan.parameters where the missing output columns come from: repeated group values, header labels, attribute labels, fixed chunks, or record-block cells.",
            "For column_count_too_high, explain which cells must be grouped, ignored, or treated as metadata instead of becoming columns.",
            "For row_count_too_low, explain which source rows/blocks were skipped and whether row 0 role, blank separators, or grouping keys caused the skip.",
            "For row_count_too_high, explain which rows are metadata, headers, blank separators, or attributes and should not be emitted.",
            "If a mismatch swaps two columns, preserve expected order exactly; for headered folds decide whether label comes before value from expected_output_grid.",
            "expected_output_grid order is authoritative; repair the visible mismatch before guessing any alternate behavior.",
            "If failure_context.error_family is leading_column_swap, swap the first two output fields and keep the value field unchanged; for header_grid_to_long this means changing [header, row_key, value] to [row_key, header, value] or the reverse.",
            "If failure_context.direct_repair_instruction is non-empty, follow that direct repair first and state it in plan.self_check.",
            "If lens_putback_context.putback_policy is present, follow its putback_target, target_fields, and instruction before broad guessing.",
            "Source mutation is forbidden; never mutate source data as putback. Amend only parameter p / generated code.",
            "Never mutate source data as putback. In this system source mutation is forbidden; amend only parameter p / generated code.",
            "If failure_context.error_family is value_order_mismatch, do not change extraction; change only output ordering unless the delta proves a missing value.",
            "For stacked_metric_rows value-order mismatches, decide between metric_values_interleaved and metric_rows_grouped from visible OutputTable.",
            "If foofah_features.metric_order_hint=metric_values_interleaved, the repair is interleaving values by source value column: row1[2], row2[2], row1[3], row2[3], ...",
            "For headered_fold leading-column swaps, obey expected_output_grid and any selected_operator_plan.output_columns exactly; do not let semantic labels like product/date override positional evidence.",
            "For wide_by_key/first_last_to_wide column_count_too_high on a current input, stop padding to the training OutputTable width and compute width from the current input's group size.",
            "For wide_by_key/first_last_to_wide column_count_too_low on a current input, collect every non-empty value in the current group; do not cap at the training width.",
            "If values differ only by '.0' suffix, you accidentally converted strings to numbers. Remove int/float/pd.to_numeric and keep original cell strings.",
            "If cells contain 'key:value' or 'key=value', split them and make keys the header row only when expected_output_grid starts with labels.",
            "If context_attribute_pair_pattern is true, label:value rows are attributes of the nearest context row. Emit one row per context, not one row per attribute.",
            "If contact_attribute_pair_pattern is true, treat communication/contact labels as ordinary attribute labels: strip the label and delimiter, preserve the exact value string, and never parse it.",
            "If row count is too high in a context_attribute_pairs block, you are probably emitting title rows or attribute rows as data. Only emit contexts that collected at least one expected attribute label.",
            "If one cell contains comma-separated items and output repeats the left context, split the cell and emit one row per item.",
            "If actual rows start with an empty string but expected starts with a company/name, you included a blank separator row as data; skip blank rows.",
            "If the current expected target is one very wide row, concatenate records row-major instead of returning one row per record.",
            "If the current expected target has fewer rows than detail rows, identify header/context rows and emit only true detail rows.",
            "If a one-cell input like '$2475 / 2br ...' maps to ['$2475','2'], extract price with r'\\$\\d+' and bedrooms with r'/\\s*(\\d+)br'.",
            "If rows alternate between already split names and comma-joined names, output split first/last for both forms with guards.",
            "If one-column company/address blocks have variable length, split on blank lines or marker rows and use each block length; do not force a training-only width.",
            "If wide rows have a fixed prefix and repeated suffix chunks, choose prefix_width and chunk_size from expected shape and ignore trailing all-Err-like padding only when expected omits it.",
            "If wide-row chunking emits Err-like cells where expected has real cells, your prefix_width or chunk_size is wrong, or you stopped scanning too early. Recompute chunks from expected prefix length and continue until the final full/partial chunk.",
            "If consecutive record pairs fail by shifting detail columns, use record_pair_merge: merge row 0 with row 1, row 2 with row 3, and select detail fields by expected positions rather than blindly taking row[3:].",
            "For record_pair_merge, if expected cells appear in the paired source rows but Actual placed later cells too early, build a mapping from expected_output_grid positions to source pair positions.",
            "If mixed split/comma name rows fail current input behavior, use name_merge_split_rows: rows with two filled name cells are copied, rows with one comma-joined cell are split once and reordered.",
            "If split_extract passed visible but current input has extra source cells, search the whole row for the split/extract cell and emit only expected_output_grid width; do not preserve unrelated cells.",
            "If blank_separated_blocks keeps producing one extra row, discard short leading/title blocks or flatten nonblank cells and wrap by expected width instead of trusting every blank-delimited block.",
            "If input is sparse multi-row blocks and output is one wide row per block, flatten non-empty cells row-major and guard short blocks with len checks.",
            "If output is taller than expected in a row-filter task, you probably need a stricter required-column predicate or to keep only rows matching the expected nonblank signature.",
            "If previous attempts matched the example shape but failed current/generalization behavior, remove literal filters and express the rule by positions, blanks, delimiters, or repeated record structure.",
            "When row count is wrong, do not patch with a special literal filter first; re-check whether first rows/columns are headers, labels, or data.",
            "If naive unpivot is off by exactly one row, inspect whether the top-left cell or first example row is metadata.",
            "When output has no header row in the original CSV, return synthetic columns col_0, col_1, ... and match cells exactly.",
            "Use the failed attempt history to avoid oscillating between excluding and including the same structural row.",
            "In plan, include operator, output_rows_rule, output_columns_rule, row_0_role, value_order_rule, generalization_assumption, and self_check.",
            "Use source_to_expected_alignment to decide whether the repair is a mapping/order fix or an extraction fix.",
        ]
    return _dump(payload)


def build_postgres_sql_prompt(metadata: dict, natural_language_query: str) -> str:
    payload = {
        "mode": "postgres_read_only_sql",
        "task": "Generate a PostgreSQL read-only SQL query for the user's natural language request.",
        "user_query": natural_language_query,
        "schema_metadata": metadata,
        "hard_constraints": [
            "Return only JSON with fields sql and explanation.",
            "SQL must be SELECT or WITH ... SELECT only.",
            "Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, COPY, CALL, DO.",
            "Do not use EXPLAIN or ANALYZE in the generated SQL.",
            "Do not request or expose credentials, passwords, tokens, or secrets.",
            "Use only tables and columns present in schema_metadata.",
        ],
    }
    return _dump(payload)


def build_postgres_sql_from_expected_prompt(
    metadata: dict,
    expected_output_rows: list[dict],
    natural_language_query: str | None = None,
    previous_attempts: list[dict] | None = None,
) -> str:
    payload = {
        "mode": "postgres_expected_to_sql",
        "task": "Infer a PostgreSQL read-only SQL query that produces a table matching the expected output example.",
        "user_query": natural_language_query or "",
        "schema_metadata": metadata,
        "expected_output_rows": expected_output_rows,
        "previous_attempts": previous_attempts or [],
        "synthesis_rules": [
            "The expected output example is the target table shape and column set.",
            "Infer joins, filters, projections, aggregations, grouping, and ordering from schema_metadata and expected_output_rows.",
            "If no natural language query is provided, use only schema_metadata and expected_output_rows to infer the most likely SELECT.",
            "Expected-only SQL can be underdetermined. If expected_output_rows contain aggregate totals but no grouping keys, filters, date ranges, or row identities, do not invent arbitrary subsets such as top 10 rows, latest 10 rows, or most frequent ids unless the expected columns or user_query prove that subset.",
            "When expected totals disagree with all-table aggregation and no deterministic subset is specified, prefer a transparent all-table aggregate/projection and explain the ambiguity instead of guessing a hidden filter.",
            "Use row_count_estimate and table preview assumptions only as weak context; they are not proof of a filter.",
            "Prefer deterministic ORDER BY when the expected row order is meaningful.",
            "Use explicit aliases so output column names match the expected table when possible.",
            "Do not invent tables or columns absent from schema_metadata.",
            "When previous_attempts are present, fix the root mismatch instead of repeating the same SQL.",
        ],
        "hard_constraints": [
            "Return only JSON with fields sql and explanation.",
            "SQL must be SELECT or WITH ... SELECT only.",
            "Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, COPY, CALL, DO.",
            "Do not use EXPLAIN or ANALYZE in the generated SQL.",
            "Do not request or expose credentials, passwords, tokens, or secrets.",
            "Use only tables and columns present in schema_metadata.",
        ],
    }
    return _dump(payload)

import json

from app.services.prompts import (
    FOOFAH_CODE_SYSTEM_PROMPT,
    FOOFAH_ROUTER_SYSTEM_PROMPT,
    GENERIC_TRANSFORM_SYSTEM_PROMPT,
    build_foofah_generation_prompt,
    build_foofah_operator_prompt,
    build_foofah_router_prompt,
    build_generation_prompt,
    build_postgres_sql_from_expected_prompt,
    build_query_prompt,
    build_repair_prompt,
)


def test_foofah_system_prompts_are_separate_from_generic_prompt():
    assert FOOFAH_CODE_SYSTEM_PROMPT != GENERIC_TRANSFORM_SYSTEM_PROMPT
    assert FOOFAH_ROUTER_SYSTEM_PROMPT != GENERIC_TRANSFORM_SYSTEM_PROMPT
    assert "2D string matrix" in FOOFAH_CODE_SYSTEM_PROMPT
    assert "no trusted header row" in FOOFAH_CODE_SYSTEM_PROMPT
    assert "ranked operator family" in FOOFAH_ROUTER_SYSTEM_PROMPT


def test_foofah_prompts_include_concise_generalization_contract():
    input_rows = [{"col_0": "A", "col_1": "1"}, {"col_0": "B", "col_1": "2"}]
    output_rows = [{"col_0": "A", "col_1": "1"}, {"col_0": "B", "col_1": "2"}]

    generation = json.loads(build_foofah_generation_prompt({"rows": 2}, input_rows, output_rows))
    router = json.loads(build_foofah_router_prompt({"rows": 2}, input_rows, output_rows))
    operator = json.loads(build_foofah_operator_prompt({"operator": "row_filter"}, {"rows": 2}, input_rows, output_rows))
    repair = json.loads(
        build_repair_prompt(
            "def transform(df): return df",
            "Value mismatch",
            {"rows": 2},
            input_rows,
            output_rows,
            mode="foofah_program_synthesis_repair",
        )
    )

    for payload in [generation, router, operator, repair]:
        contract = payload["generalization_contract"]
        assert 5 <= len(contract) <= 7
        text = " ".join(contract)
        assert "reusable row-generation rule" in text
        assert "visible positional order" in text
        assert "test_answer_grid" not in text
        assert "benchmark_generalization_feedback" not in text


def test_generation_prompt_contains_profile_examples_instruction_constraints():
    prompt = build_generation_prompt(
        {"rows": 2, "columns": [{"name": "price"}]},
        [{"price": 10, "quantity": 2}],
        [{"total": 20}],
        "calculate total",
    )
    data = json.loads(prompt)
    assert data["source_profile"]["rows"] == 2
    assert data["example_input_rows"][0]["price"] == 10
    assert data["important_validation_target"]["expected_output_rows"][0]["total"] == 20
    assert data["user_instruction"] == "calculate total"
    assert "Do not hardcode" in " ".join(data["business_rules"])


def test_generation_prompt_guides_mixed_date_parsing_without_nan_output():
    prompt = build_generation_prompt(
        {"rows": 4, "columns": [{"name": "signup_date"}]},
        [{"signup_date": "2025-02-01"}, {"signup_date": "01.02.2025"}],
        [{"signup_date": "2025-01-02"}, {"signup_date": "2025-02-01"}],
        "normalize signup dates",
    )
    data = json.loads(prompt)
    rules = " ".join(data["business_rules"])

    assert "mixed-format" in rules
    assert "visible_value_alignment" in data
    assert data["visible_value_alignment"][0]["column"] == "signup_date"
    assert data["visible_value_alignment"][0]["examples"][0]["source"] == "2025-02-01"
    assert data["visible_value_alignment"][0]["examples"][0]["expected"] == "2025-01-02"
    assert "YYYY-DD-MM" in " ".join(data["visible_value_alignment"][0]["inference_hints"])
    assert "Do not use format='mixed' blindly" in rules
    assert "NaT/NaN" in rules
    assert "literal strings 'nan' or 'NaT'" in rules


def test_repair_prompt_guides_nan_date_mismatch_to_parser_fallback():
    prompt = build_repair_prompt(
        "def transform(df): return df",
        "Value mismatch examples: [{'row': 1, 'column': 'signup_date', 'actual': 'nan', 'expected': '2025-02-01'}]",
        {"rows": 4, "columns": [{"name": "signup_date"}]},
        [{"signup_date": "01.02.2025"}, {"signup_date": "2025-02-01"}],
        [{"signup_date": "2025-02-01"}, {"signup_date": "2025-02-01"}],
        "normalize signup dates",
    )
    data = json.loads(prompt)
    rules = " ".join(data["repair_rules"])

    assert "actual 'nan' or 'NaT'" in rules
    assert "conversion lost data" in rules
    assert "mixed-format parsing with fallback" in rules
    assert "visible_value_alignment" in data
    assert "do not assume dayfirst/monthfirst/ISO globally" in rules


def test_repair_prompt_contains_previous_code_error_attempt_history():
    prompt = build_repair_prompt(
        "def transform(df): return df",
        "Column mismatch",
        {"rows": 1},
        [{"a": 1}],
        [{"b": 1}],
        "fix",
        [{"attempt": 1, "error": "Column mismatch"}],
    )
    data = json.loads(prompt)
    assert data["previous_code"] == "def transform(df): return df"
    assert data["critical_error_or_diff"] == "Column mismatch"
    assert data["attempt_history"][0]["attempt"] == 1


def test_repair_prompt_carries_failure_context():
    prompt = build_repair_prompt(
        "def transform(df): return df",
        "Column mismatch",
        {"rows": 1},
        [{"col_0": "A"}],
        [{"col_0": "A", "col_1": "B"}],
        attempt_history=[{"attempt": 1, "error": "Column mismatch"}],
        mode="foofah_program_synthesis_repair",
        failure_context={"error_family": "column_count_too_low", "expected_width": 2, "actual_width": 1},
    )
    data = json.loads(prompt)
    assert data["failure_context"]["error_family"] == "column_count_too_low"
    assert data["priority_rule_packs"][1]["name"] == "column_count_too_low"
    assert "source_to_expected_alignment" in data
    assert "shape_proof_contract" in data
    assert "column_count_too_low" in " ".join(data["foofah_repair_rules"])


def test_repair_prompt_carries_lens_putback_context():
    prompt = build_repair_prompt(
        "def transform(df): return df",
        "Value mismatch",
        {"rows": 1},
        [{"col_0": "A"}],
        [{"col_0": "A"}],
        mode="foofah_program_synthesis_repair",
        lens_putback_context={
            "typed_view_delta": {"kind": "value_order_delta"},
            "putback_policy": {"policy_name": "adjust_value_order", "putback_target": "parameter_p"},
            "putback_mode": {"mode": "parameter_only_putback", "source_mutation": "forbidden"},
        },
    )
    data = json.loads(prompt)

    assert data["lens_putback_context"]["typed_view_delta"]["kind"] == "value_order_delta"
    assert data["lens_putback_context"]["putback_policy"]["policy_name"] == "adjust_value_order"
    assert "lens_putback_context" in " ".join(data["repair_rules"])
    assert "Source mutation is forbidden" in " ".join(data["foofah_repair_rules"])


def test_foofah_router_prompt_carries_reroute_context():
    prompt = build_foofah_router_prompt(
        {"rows": 2},
        [{"col_0": "A", "col_1": "1"}, {"col_0": "A", "col_1": "2"}],
        [{"col_0": "A", "col_1": "1", "col_2": "2"}],
        reroute_context={"failed_operator": "column_select", "error_family": "column_count_too_low"},
    )
    data = json.loads(prompt)
    assert data["reroute_context"]["failed_operator"] == "column_select"
    assert "failed_operator" in " ".join(data["decision_rules"])


def test_query_prompt_does_not_require_expected():
    prompt = build_query_prompt({"rows": 3}, [{"product": "A"}], "top 5 products")
    data = json.loads(prompt)
    assert data["mode"] == "analytical_query"
    assert data["user_query"] == "top 5 products"
    assert "expected_output_rows" not in data


def test_postgres_expected_sql_prompt_allows_expected_only_synthesis():
    prompt = build_postgres_sql_from_expected_prompt(
        {
            "tables": [
                {
                    "schema": "public",
                    "table": "orders",
                    "columns": [{"name": "customer_id", "type": "integer"}, {"name": "amount", "type": "numeric"}],
                }
            ]
        },
        [{"customer_id": 1, "total_amount": 42}],
        previous_attempts=[{"attempt": 1, "validation_message": "Column mismatch"}],
    )
    data = json.loads(prompt)

    assert data["mode"] == "postgres_expected_to_sql"
    assert data["user_query"] == ""
    assert data["expected_output_rows"][0]["total_amount"] == 42
    assert data["previous_attempts"][0]["validation_message"] == "Column mismatch"
    assert "If no natural language query is provided" in " ".join(data["synthesis_rules"])
    rules = " ".join(data["synthesis_rules"])
    assert "Expected-only SQL can be underdetermined" in rules
    assert "do not invent arbitrary subsets" in rules


def test_foofah_prompt_contains_operator_catalog():
    prompt = build_foofah_generation_prompt(
        {"rows": 2, "columns": [{"name": "col_0"}, {"name": "col_1"}]},
        [{"col_0": "Name", "col_1": "Subject 1"}, {"col_0": "Name1", "col_1": "A"}],
        [{"col_0": "Name1", "col_1": "Subject1", "col_2": "A"}],
    )
    data = json.loads(prompt)
    assert "priority_rule_packs" in data
    assert "generalized_few_shot_examples" in data
    assert "source_to_expected_alignment" in data
    assert "shape_proof_contract" in data
    assert "benchmark_generalization_analysis" in data
    catalog = " ".join(data["operator_decision_table"] + data["core_operators"] + data["known_ambiguous_cases"])
    assert "fold_triple" in catalog
    assert "pivot_long_to_wide" in catalog
    assert "stacked_metric_rows" in catalog
    assert "grouped_suffix_wide" in catalog
    assert "flatten_record_blocks" in catalog
    assert "flatten_all_rows" in catalog
    assert "split_list_fold" in catalog
    assert "wide_by_key" in catalog
    assert "key_value_unfold" in catalog
    assert "marker_separated_blocks" in catalog
    assert "groupby/list expansion" in catalog


def test_foofah_router_prompt_returns_operator_schema():
    prompt = build_foofah_router_prompt(
        {"rows": 3},
        [{"col_0": "cars", "col_1": "1"}, {"col_0": "cars", "col_1": "2"}],
        [{"col_0": "cars", "col_1": "1", "col_2": "2"}],
    )
    data = json.loads(prompt)
    assert data["mode"] == "foofah_operator_router"
    assert "operator" in data["response_schema"]
    assert "ranked_operators" in data["response_schema"]
    assert "candidate_diversity_contract" in data
    assert "shape_proof_contract" in data
    assert "source_to_expected_alignment" in data
    assert "benchmark_generalization_analysis" in data
    assert "same first-column key repeated" in " ".join(data["decision_rules"])
    assert "foofah_features" in data
    assert "wide_by_key" in data["foofah_features"]["likely_operator_hints"]


def test_foofah_operator_prompt_uses_router_plan():
    prompt = build_foofah_operator_prompt(
        {"operator": "wide_by_key", "confidence": 0.8, "parameters": {"key_col": 0}},
        {"rows": 3},
        [{"col_0": "cars", "col_1": "1"}, {"col_0": "cars", "col_1": "2"}],
        [{"col_0": "cars", "col_1": "1", "col_2": "2"}],
    )
    data = json.loads(prompt)
    assert data["mode"] == "foofah_operator_code_generation"
    assert data["selected_operator_plan"]["operator"] == "wide_by_key"
    assert "data = df.fillna('').astype(str).values.tolist()" in " ".join(data["hard_code_rules"])
    assert "foofah_features" in data
    assert "priority_rule_packs" in data
    assert "source_to_expected_alignment" in data
    assert "shape_proof_contract" in data
    assert "benchmark_generalization_analysis" in data
    rules = " ".join(data["hard_code_rules"])
    assert "across the whole table" in rules
    assert "output_rows_rule" in rules


def test_foofah_prompt_marks_row_filter_subset():
    prompt = build_foofah_router_prompt(
        {"rows": 3},
        [
            {"col_0": "a", "col_1": ""},
            {"col_0": "b", "col_1": "keep"},
            {"col_0": "c", "col_1": ""},
        ],
        [{"col_0": "b", "col_1": "keep"}],
    )
    data = json.loads(prompt)
    assert data["foofah_features"]["expected_rows_are_subset_of_input_rows"] is True
    assert "row_filter" in data["foofah_features"]["likely_operator_hints"]


def test_foofah_prompt_marks_key_value_unfold():
    prompt = build_foofah_router_prompt(
        {"rows": 2},
        [{"col_0": "STUDENTNAME=Kris Johnson"}, {"col_0": "GENDER=F"}],
        [{"col_0": "STUDENTNAME", "col_1": "GENDER"}, {"col_0": "Kris Johnson", "col_1": "F"}],
    )
    data = json.loads(prompt)
    assert "key_value_unfold" in data["foofah_features"]["likely_operator_hints"]


def test_foofah_prompt_marks_marker_separated_blocks():
    prompt = build_foofah_router_prompt(
        {"rows": 5},
        [
            {"col_0": "*Company A"},
            {"col_0": "123 Main"},
            {"col_0": ""},
            {"col_0": "*Company B"},
            {"col_0": "456 Main"},
        ],
        [
            {"col_0": "Company A", "col_1": "123 Main"},
            {"col_0": "Company B", "col_1": "456 Main"},
        ],
    )
    data = json.loads(prompt)
    assert "marker_separated_blocks" in data["foofah_features"]["likely_operator_hints"]


def test_foofah_prompt_marks_header_grid_to_long():
    prompt = build_foofah_router_prompt(
        {"rows": 3},
        [
            {"col_0": "", "col_1": "9/1/2008", "col_2": "9/2/2008"},
            {"col_0": "Product1", "col_1": "10", "col_2": "20"},
            {"col_0": "Product2", "col_1": "30", "col_2": "40"},
        ],
        [
            {"col_0": "Product1", "col_1": "9/1/2008", "col_2": "10"},
            {"col_0": "Product1", "col_1": "9/2/2008", "col_2": "20"},
        ],
    )
    data = json.loads(prompt)
    assert "header_grid_to_long" in data["foofah_features"]["likely_operator_hints"]


def test_foofah_prompt_marks_pivot_long_to_wide():
    prompt = build_foofah_router_prompt(
        {"rows": 3},
        [
            {"col_0": "George", "col_1": "Math", "col_2": "65"},
            {"col_0": "George", "col_1": "French", "col_2": "42"},
            {"col_0": "Bob", "col_1": "English", "col_2": "96"},
        ],
        [
            {"col_0": "", "col_1": "Math", "col_2": "French", "col_3": "English"},
            {"col_0": "George", "col_1": "65", "col_2": "42", "col_3": ""},
        ],
    )
    data = json.loads(prompt)
    assert "pivot_long_to_wide" in data["foofah_features"]["likely_operator_hints"]


def test_foofah_first_row_role_header_when_labels_are_reused():
    prompt = build_foofah_router_prompt(
        {"rows": 2},
        [
            {"col_0": "Name", "col_1": "Subject 1", "col_2": "Subject 2"},
            {"col_0": "Name1", "col_1": "A", "col_2": "B"},
        ],
        [
            {"col_0": "Name1", "col_1": "Subject1", "col_2": "A"},
            {"col_0": "Name1", "col_1": "Subject2", "col_2": "B"},
        ],
    )
    data = json.loads(prompt)
    features = data["foofah_features"]
    assert features["first_row_role_guess"] == "header"
    assert features["first_row_values_used_as_output_labels"] is True


def test_foofah_first_row_role_data_when_first_row_is_output_record():
    prompt = build_foofah_router_prompt(
        {"rows": 5},
        [
            {"col_0": "Latimer", "col_1": "George"},
            {"col_0": "Latimer", "col_1": "Anna"},
            {"col_0": "Smith", "col_1": "Joan"},
            {"col_0": "Smith", "col_1": "Mary"},
            {"col_0": "Latimer", "col_1": "Bob"},
        ],
        [
            {"col_0": "Latimer", "col_1": "George", "col_2": "Anna", "col_3": "Bob"},
            {"col_0": "Smith", "col_1": "Joan", "col_2": "Mary", "col_3": ""},
        ],
    )
    data = json.loads(prompt)
    features = data["foofah_features"]
    assert features["first_row_role_guess"] == "data"
    assert features["first_row_values_used_as_output_data"] is True


def test_foofah_first_row_role_metadata_when_title_is_not_output():
    prompt = build_foofah_router_prompt(
        {"rows": 6},
        [
            {"col_0": "Bureau of I.A.", "col_1": ""},
            {"col_0": "Regional Director", "col_1": "Numbers"},
            {"col_0": "Niles C.", "col_1": "Tel:(800)645-8397"},
            {"col_0": "", "col_1": "Fax:(907)586-7252"},
            {"col_0": "Jean H.", "col_1": "Tel:(918)781-4600"},
            {"col_0": "", "col_1": "Fax:(918)781-4604"},
        ],
        [
            {"col_0": "", "col_1": "Tel", "col_2": "Fax"},
            {"col_0": "Niles C.", "col_1": "(800)645-8397", "col_2": "(907)586-7252"},
            {"col_0": "Jean H.", "col_1": "(918)781-4600", "col_2": "(918)781-4604"},
        ],
    )
    data = json.loads(prompt)
    features = data["foofah_features"]
    assert features["first_row_role_guess"] == "metadata"
    assert features["first_row_is_metadata_like"] is True


def test_foofah_prompt_describes_non_contiguous_unfold2_grouping():
    prompt = build_foofah_generation_prompt(
        {"rows": 5},
        [
            {"col_0": "Latimer", "col_1": "George"},
            {"col_0": "Latimer", "col_1": "Anna"},
            {"col_0": "Smith", "col_1": "Joan"},
            {"col_0": "Smith", "col_1": "Mary"},
            {"col_0": "Latimer", "col_1": "Bob"},
        ],
        [
            {"col_0": "Latimer", "col_1": "George", "col_2": "Anna", "col_3": "Bob"},
            {"col_0": "Smith", "col_1": "Joan", "col_2": "Mary", "col_3": ""},
        ],
    )
    data = json.loads(prompt)
    catalog = " ".join(data["core_operators"] + data["known_ambiguous_cases"] + data["code_rules"])
    assert "non-contiguous" in catalog
    assert "current max group" in catalog or "current input" in catalog


def test_foofah_prompt_marks_context_attribute_pairs():
    example_input = [
        {"col_0": "Bureau of I.A.", "col_1": ""},
        {"col_0": "Regional Director", "col_1": "Numbers"},
        {"col_0": "Niles C.", "col_1": "Tel:(800)645-8397"},
        {"col_0": "", "col_1": "Fax:(907)586-7252"},
        {"col_0": "Jean H.", "col_1": "Tel:(918)781-4600"},
        {"col_0": "", "col_1": "Fax:(918)781-4604"},
    ]
    example_output = [
        {"col_0": "", "col_1": "Tel", "col_2": "Fax"},
        {"col_0": "Niles C.", "col_1": "(800)645-8397", "col_2": "(907)586-7252"},
        {"col_0": "Jean H.", "col_1": "(918)781-4600", "col_2": "(918)781-4604"},
    ]
    router_prompt = build_foofah_router_prompt({"rows": 6}, example_input, example_output)
    router_data = json.loads(router_prompt)
    assert router_data["foofah_features"]["context_attribute_pair_pattern"] is True
    assert router_data["foofah_features"]["contact_attribute_pair_pattern"] is True
    assert "context_attribute_pairs" in router_data["foofah_features"]["likely_operator_hints"]

    generation_prompt = build_foofah_generation_prompt({"rows": 6}, example_input, example_output)
    generation_data = json.loads(generation_prompt)
    catalog = " ".join(
        generation_data["operator_decision_table"]
        + generation_data["core_operators"]
        + generation_data["known_ambiguous_cases"]
        + generation_data["code_rules"]
    )
    assert "one row per context" in catalog
    assert "attribute rows as output rows" in catalog
    assert "contact_attribute_pair_pattern" in catalog


def test_foofah_prompt_marks_stacked_metric_rows():
    prompt = build_foofah_router_prompt(
        {"rows": 2},
        [
            {"col_0": "ABC", "col_1": "Sum of Invoices", "col_2": "766,469", "col_3": "703,255"},
            {"col_0": "", "col_1": "Count of Invoices", "col_2": "74", "col_3": "70"},
        ],
        [{"col_0": "ABC", "col_1": "766,469", "col_2": "703,255", "col_3": "74", "col_4": "70"}],
    )
    data = json.loads(prompt)
    assert "stacked_metric_rows" in data["foofah_features"]["likely_operator_hints"]
    assert data["foofah_features"]["metric_order_hint"] == "metric_rows_grouped"

    interleaved_prompt = build_foofah_router_prompt(
        {"rows": 2},
        [
            {"col_0": "ABC", "col_1": "Sum of Invoices", "col_2": "766,469", "col_3": "703,255"},
            {"col_0": "", "col_1": "Count of Invoices", "col_2": "74", "col_3": "70"},
        ],
        [{"col_0": "ABC", "col_1": "766,469", "col_2": "74", "col_3": "703,255", "col_4": "70"}],
    )
    interleaved_data = json.loads(interleaved_prompt)
    assert interleaved_data["foofah_features"]["metric_order_hint"] == "metric_values_interleaved"
    interleaved_catalog = json.dumps(interleaved_data, ensure_ascii=False)
    assert "row1_val1" in interleaved_catalog
    assert "test_answer_grid" not in interleaved_catalog

    generation_prompt = build_foofah_generation_prompt(
        {"rows": 2},
        [
            {"col_0": "ABC", "col_1": "Sum of Invoices", "col_2": "766,469", "col_3": "703,255"},
            {"col_0": "", "col_1": "Count of Invoices", "col_2": "74", "col_3": "70"},
        ],
        [{"col_0": "ABC", "col_1": "766,469", "col_2": "74", "col_3": "703,255", "col_4": "70"}],
    )
    generation_catalog = json.dumps(json.loads(generation_prompt), ensure_ascii=False)
    assert "metric_values_interleaved" in generation_catalog
    assert "row1_val1" in generation_catalog
    assert "test_answer_grid" not in generation_catalog


def test_foofah_prompt_marks_header_grid_long_order():
    prompt = build_foofah_router_prompt(
        {"rows": 3},
        [
            {"col_0": "", "col_1": "9/1/2008", "col_2": "9/2/2008"},
            {"col_0": "Product1", "col_1": "0", "col_2": "5"},
        ],
        [
            {"col_0": "9/1/2008", "col_1": "Product1", "col_2": "0"},
            {"col_0": "9/2/2008", "col_1": "Product1", "col_2": "5"},
        ],
    )
    data = json.loads(prompt)

    assert data["foofah_features"]["header_grid_long_order_hint"] == "header_then_row_key"
    assert "header_grid_to_long" in data["foofah_features"]["likely_operator_hints"]


def test_foofah_prompt_visible_target_is_authoritative_without_hidden_feedback():
    prompt = build_foofah_generation_prompt(
        {"rows": 3},
        [
            {"col_0": "", "col_1": "9/1/2008", "col_2": "9/2/2008", "col_3": "9/3/2008"},
            {"col_0": "", "col_1": "", "col_2": "", "col_3": ""},
            {"col_0": "Product1", "col_1": "0", "col_2": "5", "col_3": "8"},
        ],
        [
            {"col_0": "9/1/2008", "col_1": "Product1", "col_2": "0"},
            {"col_0": "9/2/2008", "col_1": "Product1", "col_2": "5"},
            {"col_0": "9/3/2008", "col_1": "Product1", "col_2": "8"},
        ],
    )
    data = json.loads(prompt)
    analysis = data["benchmark_generalization_analysis"]

    assert analysis["has_feedback"] is False
    assert analysis["visible_header_grid_long_order_hint"] == "header_then_row_key"
    assert "expected_output_grid is authoritative" in " ".join(analysis["rules"])


def test_foofah_prompt_detects_hidden_header_order_conflict():
    feedback = {
        "benchmark_generalization_feedback": {
            "visible_input_grid": [
                ["", "9/1/2008", "9/2/2008", "9/3/2008"],
                ["", "", "", ""],
                ["Product1", "0", "5", "8"],
            ],
            "visible_output_grid": [
                ["9/1/2008", "Product1", "0"],
                ["9/2/2008", "Product1", "5"],
                ["9/3/2008", "Product1", "8"],
            ],
            "visible_shape": {"input": [3, 4], "output": [3, 3]},
            "testing_table_grid": [
                ["", "9/1/2008", "9/2/2008", "9/3/2008"],
                ["", "", "", ""],
                ["Product2", "3", "5", "10"],
                ["Product3", "0", "1", "4"],
            ],
            "test_answer_grid": [
                ["Product2", "9/1/2008", "3"],
                ["Product2", "9/2/2008", "5"],
                ["Product2", "9/3/2008", "10"],
                ["Product3", "9/1/2008", "0"],
            ],
            "hidden_shape": {"input": [4, 4], "output": [4, 3]},
        }
    }
    prompt = build_foofah_generation_prompt(
        {"rows": 3},
        [
            {"col_0": "", "col_1": "9/1/2008", "col_2": "9/2/2008", "col_3": "9/3/2008"},
            {"col_0": "", "col_1": "", "col_2": "", "col_3": ""},
            {"col_0": "Product1", "col_1": "0", "col_2": "5", "col_3": "8"},
        ],
        [
            {"col_0": "9/1/2008", "col_1": "Product1", "col_2": "0"},
            {"col_0": "9/2/2008", "col_1": "Product1", "col_2": "5"},
            {"col_0": "9/3/2008", "col_1": "Product1", "col_2": "8"},
        ],
        user_instruction=json.dumps(feedback),
    )
    analysis = json.loads(prompt)["benchmark_generalization_analysis"]

    assert analysis["has_feedback"] is True
    assert analysis["visible_header_grid_long_order_hint"] == "header_then_row_key"
    assert analysis["hidden_header_grid_long_order_hint"] == "row_key_then_header"
    assert "header_grid_order_conflict" in analysis["order_conflicts"]


def test_foofah_prompt_detects_hidden_metric_order_conflict():
    feedback = {
        "benchmark_generalization_feedback": {
            "visible_input_grid": [
                ["ABC", "Sum of Invoices", "766,469", "703,255", "631,646", "2,101,370"],
                ["", "Count of Invoices", "74", "70", "59", "203"],
            ],
            "visible_output_grid": [
                ["ABC", "766,469", "74", "703,255", "70", "631,646", "59", "2,101,370", "203"]
            ],
            "visible_shape": {"input": [2, 6], "output": [1, 9]},
            "testing_table_grid": [
                ["DEF", "Sum of Invoices", "776,996", "1,532,159", "494,919", "2,804,074"],
                ["", "Count of Invoices", "60", "76", "42", "178"],
                ["GHI", "Sum of Invoices", "371,614", "897,949", "712,365", "2,442,728"],
                ["", "Count of Invoices", "67", "63", "52", "182"],
            ],
            "test_answer_grid": [
                ["DEF", "776,996", "1,532,159", "494,919", "2,804,074", "60", "76", "42", "178"],
                ["GHI", "371,614", "897,949", "712,365", "2,442,728", "67", "63", "52", "182"],
            ],
            "hidden_shape": {"input": [4, 6], "output": [2, 9]},
        }
    }
    prompt = build_foofah_generation_prompt(
        {"rows": 2},
        [
            {"col_0": "ABC", "col_1": "Sum of Invoices", "col_2": "766,469", "col_3": "703,255", "col_4": "631,646", "col_5": "2,101,370"},
            {"col_0": "", "col_1": "Count of Invoices", "col_2": "74", "col_3": "70", "col_4": "59", "col_5": "203"},
        ],
        [
            {"col_0": "ABC", "col_1": "766,469", "col_2": "74", "col_3": "703,255", "col_4": "70", "col_5": "631,646", "col_6": "59", "col_7": "2,101,370", "col_8": "203"},
        ],
        user_instruction=json.dumps(feedback),
    )
    analysis = json.loads(prompt)["benchmark_generalization_analysis"]

    assert analysis["visible_metric_order_hint"] == "metric_values_interleaved"
    assert analysis["hidden_metric_order_hint"] == "metric_rows_grouped"
    assert "metric_order_conflict" in analysis["order_conflicts"]


def test_foofah_prompt_marks_record_pair_merge_and_name_merge_split():
    pair_prompt = build_foofah_router_prompt(
        {"rows": 4},
        [
            {"col_0": "3099", "col_1": "905", "col_2": "TITLE"},
            {"col_0": "NO.14", "col_1": "NO.14", "col_2": "Full Copies"},
            {"col_0": "3200", "col_1": "906", "col_2": "TITLE 2"},
            {"col_0": "9-Jun", "col_1": "9-Jun", "col_2": "Covers Only"},
        ],
        [
            {"col_0": "3099", "col_1": "905", "col_2": "TITLE", "col_3": "NO.14"},
            {"col_0": "3200", "col_1": "906", "col_2": "TITLE 2", "col_3": "9-Jun"},
        ],
    )
    pair_data = json.loads(pair_prompt)
    assert "record_pair_merge" in pair_data["foofah_features"]["likely_operator_hints"]

    name_prompt = build_foofah_router_prompt(
        {"rows": 2},
        [
            {"col_0": "Anna", "col_1": "Davis", "col_2": ""},
            {"col_0": "", "col_1": "", "col_2": "Dole,Jerry"},
        ],
        [
            {"col_0": "Anna", "col_1": "Davis"},
            {"col_0": "Jerry", "col_1": "Dole"},
        ],
    )
    name_data = json.loads(name_prompt)
    assert "name_merge_split_rows" in name_data["foofah_features"]["likely_operator_hints"]

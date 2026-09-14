import ast
import os
import ast
import pandas as pd
import pytest


class TestIngestionDiscovery:
    """Script must auto-discover CSVs, not hard-code filenames."""

    def test_no_hardcoded_filenames(self, reload_module):
        """ingestion.py must not contain literal dataset filenames."""
        import ingestion

        source = open(ingestion.__file__).read()
        # Should not have literal "dataset1.csv" or "dataset2.csv" as string literals
        tree = ast.parse(source)
        string_literals = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.add(node.value)

        hardcoded = {"dataset1.csv", "dataset2.csv", "dataset3.csv", "dataset4.csv"}
        found = hardcoded & string_literals
        assert not found, f"Hard-coded filenames found in ingestion.py: {found}"

    def test_discovers_all_csv_files(self, workspace_with_input, reload_module):
        """Ingestion must find all CSV files in input_folder_path."""
        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        output_file = workspace_with_input["paths"]["output"] / "finaldata.csv"
        assert output_file.exists(), "finaldata.csv was not created"

        df = pd.read_csv(output_file)
        # dataset1 has 10 rows, dataset2 has 8 rows, no overlap
        assert len(df) == 18, f"Expected 18 rows, got {len(df)}"

    def test_works_with_different_file_count(self, workspace, reload_module):
        """Script must work with 1, 3, or N files — not just 2."""
        from tests.conftest import make_sample_df

        # Write 3 files instead of 2
        for i in range(3):
            df = make_sample_df(n=5, seed=100 + i)
            df.to_csv(workspace["paths"]["input"] / f"file_{i}.csv", index=False)

        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        output_file = workspace["paths"]["output"] / "finaldata.csv"
        assert output_file.exists()
        df = pd.read_csv(output_file)
        assert len(df) == 15, f"Expected 15 rows from 3 files, got {len(df)}"


class TestIngestionDeduplication:
    """Duplicate rows across input files must be removed."""

    def test_removes_duplicate_rows(self, workspace_with_duplicates, reload_module):
        """Overlapping rows between files should appear only once."""
        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        output_file = workspace_with_duplicates["paths"]["output"] / "finaldata.csv"
        df = pd.read_csv(output_file)

        # df1 has 10 rows, df2 has 5 dupes + 5 new = 15 total unique
        assert len(df) == 15, f"Expected 15 unique rows, got {len(df)}"
        assert not df.duplicated().any(), "Output still contains duplicates"


class TestIngestionOutput:
    """Output files must be written to the correct locations."""

    def test_finaldata_csv_created(self, workspace_with_input, reload_module):
        """finaldata.csv must exist in output_folder_path."""
        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        output_file = workspace_with_input["paths"]["output"] / "finaldata.csv"
        assert output_file.exists(), "finaldata.csv not found in output_folder_path"

    def test_finaldata_has_correct_columns(self, workspace_with_input, reload_module):
        """Output CSV must preserve the original schema."""
        from tests.conftest import SAMPLE_COLUMNS

        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        output_file = workspace_with_input["paths"]["output"] / "finaldata.csv"
        df = pd.read_csv(output_file)
        assert list(df.columns) == SAMPLE_COLUMNS

    def test_ingestedfiles_txt_created(self, workspace_with_input, reload_module):
        """ingestedfiles.txt must exist in output_folder_path."""
        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        record_file = workspace_with_input["paths"]["output"] / "ingestedfiles.txt"
        assert record_file.exists(), "ingestedfiles.txt not found in output_folder_path"

    def test_ingestedfiles_txt_contains_all_filenames(self, workspace_with_input, reload_module):
        """ingestedfiles.txt must list every CSV that was read."""
        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        record_file = workspace_with_input["paths"]["output"] / "ingestedfiles.txt"
        content = record_file.read_text()
        assert "dataset1.csv" in content
        assert "dataset2.csv" in content

    def test_ingestedfiles_is_parseable_list(self, workspace_with_input, reload_module):
        """ingestedfiles.txt should be a parseable Python list."""
        import ast

        mod = reload_module("ingestion")
        mod.merge_multiple_dataframe()

        record_file = workspace_with_input["paths"]["output"] / "ingestedfiles.txt"
        content = record_file.read_text()
        parsed = ast.literal_eval(content)
        assert isinstance(parsed, list), f"Expected list, got {type(parsed)}"
        assert len(parsed) == 2

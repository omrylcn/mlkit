import pytest
import pandas as pd
import polars as pl
from pathlib import Path
import tempfile

from mlkit.data.data_manager import PandasAdapter, PolarsAdapter


@pytest.fixture
def simple_data():
    """Create simple test data"""
    return pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["New York", "London", "Paris"]})


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


class TestDataFrameAdapters:
    """Test individual DataFrame adapters"""

    @classmethod
    def setup_class(cls):
        """Setup shared test data"""
        cls.test_data = {"name": ["John", "Jane", "Bob"], "age": [30, 25, 35], "city": ["New York", "London", "Paris"]}

        cls.test_df = pd.DataFrame(cls.test_data)

    @staticmethod
    def create_test_file(data: dict, path: Path, format: str) -> None:
        """Utility to create test files in different formats"""
        df = pd.DataFrame(data)
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "parquet":
            df.to_parquet(path, index=False)
        elif format == "json":
            df.to_json(path, orient="records")

    @staticmethod
    def validate_dataframe(df, expected_data: dict) -> bool:
        """Validate DataFrame contents"""
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        expected_df = pd.DataFrame(expected_data)
        try:
            pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_df.reset_index(drop=True), check_dtype=False)
            return True
        except AssertionError:
            return False

    @pytest.mark.parametrize(
        "adapter_class,format", [(PandasAdapter, "csv"), (PandasAdapter, "parquet"), (PandasAdapter, "json"), (PolarsAdapter, "csv"), (PolarsAdapter, "parquet"), (PolarsAdapter, "json")]
    )
    def test_adapter_read(self, adapter_class, format, temp_dir):
        """Test reading and writing with different adapters and formats"""
        adapter = adapter_class()
        file_path = temp_dir / f"test.{format}"

        self.create_test_file(self.test_data, file_path, format)
        df = adapter.read_file(file_path, format)
        assert self.validate_dataframe(df, self.test_data)

    @pytest.mark.parametrize("adapter_class,data_type", [(PandasAdapter, pd.DataFrame), (PolarsAdapter, pl.DataFrame)])
    def test_to_origin(self, adapter_class, data_type, temp_dir):
        """Test DataFrame conversion between formats"""
        adapter = adapter_class()
        converted_data = adapter.to_origin(self.test_df)
        assert isinstance(converted_data, data_type)

    @pytest.mark.parametrize("adapter_class,data_class,format", [(PandasAdapter, pd.DataFrame, "csv"), (PolarsAdapter, pl.DataFrame, "csv")])
    def test_to_df(self, adapter_class, data_class, format):
        data_frame = data_class(self.test_data)
        adapter = adapter_class()
        converted_data = adapter.to_df(data_frame)
        assert isinstance(converted_data, pd.DataFrame)

    @pytest.mark.parametrize(
        "adapter_class,format", [(PandasAdapter, "csv"), (PandasAdapter, "parquet"), (PandasAdapter, "json"), (PolarsAdapter, "csv"), (PolarsAdapter, "parquet"), (PolarsAdapter, "json")]
    )
    def test_adapter_write(self, adapter_class, format, temp_dir):
        """Test writing with different adapters and formats"""
        adapter = adapter_class()
        file_path = temp_dir / f"write_test.{format}"

        input_df = adapter.to_origin(self.test_df)

        adapter.write_file(input_df, file_path, format)
        assert file_path.exists()

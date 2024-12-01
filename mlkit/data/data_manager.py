from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, TypeVar, Generic, List, Optional
import polars as pl
import pandas as pd

from mlkit.log import logger
from mlkit.config.data import DataConfig

DF = TypeVar("DF")



class DataFrameAdapter(ABC, Generic[DF]):
    """Abstract base adapter interface."""

    @abstractmethod
    def read_file(self, path: Path, file_format: str, **kwargs) -> DF:
        """Read data from file."""
        pass

    @abstractmethod
    def write_file(self, df: DF, path: Path, file_format: str, columns: List[str], **kwargs) -> None:
        """Write data to file."""
        pass

    @abstractmethod
    def to_df(self, df: DF) -> pd.DataFrame:
        """Convert adapter's DataFrame type to pandas DataFrame."""
        pass

    @abstractmethod
    def to_origin(self, df: pd.DataFrame) -> DF:
        """Convert pandas DataFrame to adapter's DataFrame type."""
        pass
    
    @abstractmethod
    def get_columns(self, df: DF) -> List[str]:
        """Get column names from DataFrame."""
        pass

    @abstractmethod
    def get_shape(self, df: DF) -> tuple[int, int]:
        """Get DataFrame shape (rows, columns)."""
        pass



class PolarsAdapter(DataFrameAdapter[pl.DataFrame]):
    """Polars implementation of DataFrame adapter."""

    def read_file(self, path: Path, file_format: str, use_col: Optional[List[str]] = None, **kwargs) -> pl.DataFrame:
        readers = {"csv": pl.read_csv, "parquet": pl.read_parquet, "excel": pl.read_excel, "json": pl.read_json}
        reader = readers.get(file_format.lower())
        if not reader:
            raise ValueError(f"Unsupported file format for Polars: {file_format}")

        if use_col:
            kwargs["columns"] = use_col  # Polars uses 'columns' parameter
        return reader(path, **kwargs)

    def write_file(self, df: pl.DataFrame, path: Path, file_format: str, columns: List[str] = None, **kwargs) -> None:

        if columns:
            df = df.select(columns)

        writers = {"csv": df.write_csv, "parquet": df.write_parquet, "excel": df.write_excel, "json": df.write_json}
        writer = writers.get(file_format.lower())
        if not writer:
            raise ValueError(f"Unsupported file format for Polars: {file_format}")
        writer(path, **kwargs)

    def to_origin(self, df: pd.DataFrame) -> pl.DataFrame:
        return pl.from_pandas(df)

    def to_df(self, df: pl.DataFrame) -> pd.DataFrame:
        return df.to_pandas()
   
    def get_columns(self, df: pl.DataFrame) -> List[str]:
        """Get column names from Polars DataFrame."""
        return df.columns
    
    def get_shape(self, df: pl.DataFrame) -> tuple[int, int]:
        """Get shape from Polars DataFrame."""
        return df.shape

class PandasAdapter(DataFrameAdapter[pd.DataFrame]):
    """Pandas implementation of DataFrame adapter."""

    def read_file(self, path: Path, file_format: str, use_col: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        readers = {"csv": pd.read_csv, "parquet": pd.read_parquet, "excel": pd.read_excel, "json": pd.read_json}
        reader = readers.get(file_format.lower())
        if not reader:
            raise ValueError(f"Unsupported file format for Pandas: {file_format}")

        if use_col:
            if file_format.lower() in ["csv", "excel"]:
                kwargs["usecols"] = use_col  # Pandas CSV/Excel uses 'usecols'
            elif file_format.lower() == "parquet":
                kwargs["columns"] = use_col  # Pandas Parquet uses 'columns'

        return reader(path, **kwargs)

    def write_file(self, df: pd.DataFrame, path: Path, file_format: str, columns: List[str] = None, **kwargs) -> None:

        if columns:
            df = df[columns]

        writers = {"csv": df.to_csv, "parquet": df.to_parquet, "excel": df.to_excel, "json": df.to_json}
        writer = writers.get(file_format.lower())
        if not writer:
            raise ValueError(f"Unsupported file format for Pandas: {file_format}")
        writer(path, **kwargs)

    def to_origin(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
    def get_columns(self, df: pd.DataFrame) -> List[str]:
        """Get column names from Pandas DataFrame."""
        return df.columns.tolist()
    
    def get_shape(self, df: pd.DataFrame) -> tuple[int, int]:
        """Get shape from Pandas DataFrame."""
        return df.shape

class DataManager:
    """Data manager with YAML configuration support."""

    def __init__(self, config: DataConfig, loader_type: str = "polars"):
        """Initialize with either config path or DataConfig object."""
        self._adapters = {
            "polars": PolarsAdapter(),
            "pandas": PandasAdapter(),
            # "daft": DaftAdapter(),
        }

        self.config = config
        self.loader_type = loader_type
        self.adapter = self._adapters.get(loader_type)
        if not self.adapter:
            raise ValueError(f"Unsupported loader: {loader_type}")

        logger.info(f"Initialized DataLoader with {loader_type} adapter")

    def read(self, custom_path: Union[str, Path] = None, use_col: Optional[List[str]] = None, **kwargs) -> DF:
        """Load data using configuration"""
        path = Path(custom_path or self.config.path)
        file_format = path.suffix[1:]
        options = {**(self.config.options or {}), **kwargs}
        columns_to_use = use_col if use_col is not None else self.config.use_col

        logger.info(f"Loading data from {path} using {self.loader_type}")
        
        try:
            df = self.adapter.read_file(path, file_format, use_col=columns_to_use, **options)
            logger.info(f"Successfully loaded data from {path}")
            if columns_to_use:
                logger.info(f"Selected columns: {columns_to_use}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}")
            raise

    def save(self, df: DF, path: Union[str, Path] = None, columns: Optional[List[str]] = None, **kwargs) -> None:
        """Save data using configuration."""
        save_path = Path(path or self.config.path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        file_format = save_path.suffix[1:]
        options = {**(self.config.options or {}), **kwargs}

        logger.info(f"Saving data to {save_path}")
        try:
            self.adapter.write_file(df, save_path,file_format,columns,**options)
            logger.info(f"Successfully saved data to {save_path}")
        except Exception as e:
            logger.error(f"Error saving data to {save_path}: {str(e)}")
            raise

    def to_df(self, df: DF) -> pd.DataFrame:
        """Convert DataFrame to pandas DataFrame."""
        logger.info("Converting DataFrame")
        try:
            return self.adapter.to_df(df)
        except Exception as e:
            logger.error(f"Error converting DataFrame: {str(e)}")
            raise

    def to_origin(self, df: pd.DataFrame) -> DF:
        """Convert DataFrame to original format."""
        logger.info("Converting pd.DataFrame")
        try:
            return self.adapter.to_origin(df)
        except Exception as e:
            logger.error(f"Error converting DataFrame: {str(e)}")
            raise
    
    def get_columns(self, df: DF) -> List[str]:
        """Get column names from DataFrame."""
        logger.info("Getting DataFrame columns")
        try:
            return self.adapter.get_columns(df)
        except Exception as e:
            logger.error(f"Error getting DataFrame columns: {str(e)}")
            raise

    def get_shape(self, df: DF) -> tuple[int, int]:
        """Get DataFrame shape (rows, columns)."""
        logger.info("Getting DataFrame shape")
        try:
            return self.adapter.get_shape(df)
        except Exception as e:
            logger.error(f"Error getting DataFrame shape: {str(e)}")
            raise


        

# class DaftAdapter(DataFrameAdapter[daft.DataFrame]):
#     """Daft implementation of DataFrame adapter."""

#     def read_file(self, path: Path, file_format: str, **kwargs) -> daft.DataFrame:
#         readers = {"csv": daft.read_csv, "parquet": daft.read_parquet, "json": daft.read_json}
#         reader = readers.get(file_format.lower())
#         if not reader:
#             raise ValueError(f"Unsupported file format for Daft: {file_format}")
#         return reader(str(path), **kwargs)

#     def write_file(self, df: daft.DataFrame, path: Path, file_format: str, **kwargs) -> None:
#         writers = {"csv": df.write_csv, "parquet": df.write_parquet, "json": df.write_json}
#         writer = writers.get(file_format.lower())
#         if not writer:
#             raise ValueError(f"Unsupported file format for Daft: {file_format}")
#         writer(str(path), **kwargs)

#     def to_df(self, df: pd.DataFrame) -> daft.DataFrame:
#         return daft.from_pandas(df)

#     def to_origin(self, df: daft.DataFrame) -> pd.DataFrame:
#         return df.to_pandas()

from mlkit.data.data_process import PandasProcessor,ProcessorCard,CategoricalTransformPandas,ScalerTransformPandas,MissingValueProcessorPandas
import pandas as pd


class LastPurchaseDatePandas(PandasProcessor):
    """Pandas implementation of last purchase date calculation"""
    
    def __init__(self, processor_card: ProcessorCard):
        """
        Initialize the transformer with a feature card containing:
        - input_columns: List containing ['customer_id', 'purchase_date']
        - output_column: Name of the column to store last purchase date
        """
        super().__init__(processor_card)
        if len(self.processor_card.input_columns) != 2:
            raise ValueError("input_columns must contain [customer_id, purchase_date]")
            
    def _fit_impl(self, X: pd.DataFrame, y=None):
        # Validate required columns exist
        customer_id_col, purchase_date_col = self.processor_card.input_columns
        if not all(col in X.columns for col in [customer_id_col, purchase_date_col]):
            raise ValueError(f"DataFrame must contain columns {self.processor_card.input_columns}")
        
  
        return self

    def _transform_impl(self, X: pd.DataFrame)->pd.DataFrame:
        df = X.copy()
        customer_id_col, purchase_date_col = self.processor_card.input_columns
        
        df[purchase_date_col] = pd.to_datetime(df[purchase_date_col])
        df = df.sort_values([customer_id_col, purchase_date_col])

        df[self.processor_card.output_columns[0]] = df.groupby(customer_id_col)[purchase_date_col].shift(1).values
        return df
    
class DaysSinceLastPurchasePandas(PandasProcessor):
    """Pandas implementation of days since last purchase calculation"""
    
    def __init__(self, processor_card: ProcessorCard):
        """
        Initialize the transformer with a feature card containing:
        - input_columns: List containing ['purchase_date', 'last_purchase_date']
        - output_column: Name of the column to store days since last purchase
        """
        super().__init__(processor_card)
        if len(self.processor_card.input_columns) != 2:
            raise ValueError("input_columns must contain [purchase_date, last_purchase_date]")
        
    def _fit_impl(self, X: pd.DataFrame, y=None):
     
        current_date_col, last_purchase_col = self.processor_card.input_columns
        if not all(col in X.columns for col in [current_date_col, last_purchase_col]):
            raise ValueError(f"DataFrame must contain columns {self.processor_card.input_columns}")
        
        self._fitted_state['min_date'] = pd.to_datetime(X[current_date_col]).min()
        self._fitted_state['max_date'] = pd.to_datetime(X[current_date_col]).max()
        return self
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:

        column_name = self.processor_card.output_columns[0]
        
    
        df = X.copy()
        current_date_col, last_purchase_col = self.processor_card.input_columns
        
        df[current_date_col] = pd.DatetimeIndex(df[current_date_col]).tz_localize(None)
        df[last_purchase_col] = pd.DatetimeIndex(df[last_purchase_col])
        
        df[column_name] = ((df[current_date_col] - df[last_purchase_col]).dt.days)
        df[column_name] = df[column_name].fillna(-1)
        
        return df
    

class DateFeaturesPandas(PandasProcessor):
    """Pandas implementation of date feature extraction"""
    
    def __init__(self, processor_card: ProcessorCard):
        """
        Initialize the transformer with a feature card containing:
        - input_columns: List containing ['purchase_date']
        - output_column: Prefix for all generated date features
        - parameters: Dict with flags for which features to generate
        """
        super().__init__(processor_card)
        if len(self.processor_card.input_columns) != 1:
            raise ValueError("input_columns must contain [purchase_date]")
            
               
        # Season mapping
        self.season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
    
    def _fit_impl(self, X: pd.DataFrame, y=None):
        date_col = self.processor_card.input_columns[0]
        
        if date_col not in X.columns:
            raise ValueError(f"DataFrame must contain column {date_col}")
        
        # Store statistics in fitted state
        date_series = pd.to_datetime(X[date_col])
        self._fitted_state.update({
            'min_date': date_series.min(),
            'max_date': date_series.max(),
            'unique_years': sorted(date_series.dt.year.unique().tolist()),
            'unique_months': sorted(date_series.dt.month.unique().tolist()),
            'unique_days_of_week': sorted(date_series.dt.dayofweek.unique().tolist())
        })
        return self
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        date_col = self.processor_card.input_columns[0]
        prefix = self.processor_card.parameters["prefix"]
        
        # Convert to datetime if needed
        df[date_col] = pd.to_datetime(df[date_col])
        
        if "day" in self.processor_card.output_columns:
            df[f"{prefix}_day"] = df[date_col].dt.day


        if "year" in self.processor_card.output_columns:
            df[f"{prefix}_year"] = df[date_col].dt.year

        if "season" in self.processor_card.output_columns:
             df[f"{prefix}_season"] = df[date_col].dt.month.map(self.season_map)
        
        if "month" in self.processor_card.output_columns:
            df[f"{prefix}_month"] = df[date_col].dt.month
            
        if "day_of_week" in self.processor_card.output_columns:
            df[f"{prefix}_day_of_week"] = df[date_col].dt.dayofweek
        
        return df
    


processor_lib = {
    "last_purchase_date": LastPurchaseDatePandas,
    "days_since_last_purchase": DaysSinceLastPurchasePandas,
    "date_features": DateFeaturesPandas,
    "categorical_encoding": CategoricalTransformPandas,
    "scaler": ScalerTransformPandas,
    "missing_value_processor": MissingValueProcessorPandas,
    
}
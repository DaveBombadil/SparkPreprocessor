"""
SparkPreprocessor: A Comprehensive Data Preprocessing Library for PySpark
==========================================================================

Author: David Fonfria (david.fonfria@stellantis.com)
Created: November 23, 2024
License: MIT

This module provides a robust and efficient data preprocessing framework for PySpark DataFrames, 
offering a whole range of data cleaning, transformation, and analysis capabilities. It is designed 
to handle large-scale data processing tasks with built-in optimization and parallel processing. It 
makes use of multithreading to and hashing techniques to ensure optimal performance and maximum speed.

Key Features:
------------
- Automated data type detection and handling
- Comprehensive missing value imputation strategies
- Outlier detection and treatment
- Cardinality handling for categorical variables
- Feature scaling and encoding
- Correlation analysis with multiple methods
- Statistical analysis and profiling
- Multi-threaded operations for improved performance
- Visualization capabilities for data analysis

Main Components:
--------------
- SparkPreprocessor: Main preprocessing class
- PreprocessingConfig: Configuration management
- CachedStats: Statistics and metadata caching
- Multiple enums for configuration options:
  - ColumnType: Data type classification
  - ScalingMethod: Feature scaling options
  - NullStrategy: Missing value handling
  - OutlierStrategy: Outlier treatment
  - CorrelationMethod: Correlation analysis methods

Dependencies:
------------
- PySpark
- NumPy
- Pandas
- SciPy
- Plotly
- Threading and multiprocessing libraries
- Tabulate for formatted output

Usage:
-----
The preprocessor can be used both as a standalone tool for data analysis and as part
of a larger data processing pipeline. It supports both fit-transform and transform-only
operations, with extensive configuration options for each preprocessing step.

Example:
    preprocessor = SparkPreprocessor()
    df_processed = preprocessor.fit_transform(spark_dataframe, config={
                                                                    'scaling_method': 'standard',
                                                                    'null_strategy': 'mean',
                                                                    'outlier_strategy': 'nearest'}
    )

For detailed usage instructions and configuration options, refer to the class documentation.
"""
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

# PySpark ML Features
from pyspark.ml.feature import (StandardScaler, MinMaxScaler, VectorAssembler, StringIndexer, OneHotEncoder)
from pyspark.ml.stat import Correlation

# Scientific Computing
import numpy as np
import pandas as pd
from scipy import stats

# Visualization
import plotly.graph_objects as go

# Parallel Processing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Python Standard Library
from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
import sys

# Utilities
from tqdm import tqdm
from tabulate import tabulate

# Omit warnings
import warnings
import urllib3
warnings.filterwarnings("ignore")
pool_manager = urllib3.PoolManager(maxsize=100, retries=5)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

sys.tracebacklimit = 0

class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BINARY = "binary"
    ARRAY = "array"
    MAP = "map"
    STRUCT = "struct"

class ScalingMethod(Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    NONE = "none"

class NullStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    DROP = "drop"
    FLAG = "flag"
    NONE = "none"

class OutlierStrategy(Enum):
    DROP = "drop"      # Remove rows with outliers
    MEAN = "mean"      # Replace outliers with mean
    NEAREST = "nearest"  # Replace outliers with nearest bound
    NONE = "none"      # No outlier handling
    
class CorrelationMethod(Enum):
    PEARSON = "pearson"
    CHATTERJEE = "chatterjee"
    NONE = "none"

class ExitError(Exception):
    def __init__(self, message: str):
        # Initialize with the custom message
        self.message = message
    
    def __str__(self):
        # Return the custom message
        return self.message
    
# Colors for text output
class Colors:
    """ANSI color codes for terminal output"""
    RED = "\033[91m"
    BRICK = "\033[31m"
    ORANGE = "\033[38;5;208m"
    LIGHT_ORANGE = "\033[38;5;215m"
    YELLOW = "\033[93m"
    DARK_YELLOW = "\033[38;5;136m"
    GREEN = "\033[32m"
    DARK_GREEN = "\033[38;5;22m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    TEAL = "\033[38;5;44m"
    PURPLE = "\033[38;5;129m"
    MAGENTA = "\033[35m"
    PINK = "\033[38;5;213m"
    BROWN = "\033[38;5;130m"
    LIGHT_BROWN = "\033[38;5;137m"
    DEFAULT = "\033[0m"

    @classmethod
    def combine(cls, *colors) -> str:
        """Combine multiple colors"""
        return ''.join(colors)
        
# Colored text for logger
class ColorLogger:
    """Custom logger with colored output support"""
    def __init__(self, name: Optional[str] = None, level: int = logging.INFO):
        """Initialize ColorLogger with custom name and level"""
        
        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
        logging.basicConfig(level=logging.INFO)

        self.Colors = Colors()
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(level)

    def info(self, message: str, color: str = Colors.BLUE) -> None:
        """Log info message in blue by default"""
        text = f"{color}{message}{self.Colors.DEFAULT}"
        self.logger.info(text)

    def warning(self, message: str, color: str = Colors.LIGHT_ORANGE) -> None:
        """Log warning message in light orange by default"""
        text = f"{color}{message}{self.Colors.DEFAULT}"
        self.logger.warning(text)

    def error(self, message: str, color: str = Colors.RED) -> None:
        """Log error message in red by default"""
        text = f"{color}{message}{self.Colors.DEFAULT}"
        self.logger.error(text)

    def header(self, message: str, color: str = Colors.GREEN) -> None:
        """Log header message in green by default"""
        text = f"{color}{message}{self.Colors.DEFAULT}"
        self.logger.info(text)

    def text(self, message: str, color: str = Colors.DEFAULT) -> None:
        """Log plain text message"""
        text = f"{color}{message}{self.Colors.DEFAULT}"
        self.logger.info(text)

    def success(self, message: str, color: str = Colors.GREEN) -> None:
        """Log success message in green by default"""
        text = f"{color}{message}{self.Colors.DEFAULT}"
        self.logger.info(text)

    def debug(self, message: str, color: str = Colors.LIGHT_ORANGE) -> None:
        """Log debug message in light orange by default"""
        text = f"{color}{message}{self.Colors.DEFAULT}"
        self.logger.debug(text)

@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing parameters"""

    # Column filtering
    white_list: Optional[List[str]] = None
    black_list: Optional[List[str]] = None
    
    # Scaling method configuration
    scaling_method: Union[str, ScalingMethod] = ScalingMethod.NONE
    
    # Null handling configuration
    null_strategy: Union[str, NullStrategy] = NullStrategy.NONE
    null_fill_value: Optional[Union[str, float]] = None
    null_threshold: float = 0.5
    
    # Outlier configuration
    outlier_strategy: Union[str, OutlierStrategy] = OutlierStrategy.NONE
    # outlier_threshold: float = 3.0
    z_score: float = 3.0

    # Correlation configuration
    correlation_method: Union[str, CorrelationMethod] = CorrelationMethod.NONE
    correlation_threshold: float = 0.95
    correlation_action: str = "ignore"  # Valid options: ["ignore", "drop"]
    
    # Categorical handling configuration
    max_cardinality_ratio: float = None  # Max unique values as percentage of total rows. No cardinality filtering if None
    min_cardinality_ratio: float = None  # Min unique values as percentage of total rows. No cardinality filtering if None
    cardinality_action: str = "group"    # Valid options: [None, 'group', 'drop']
    other_label: str = "OTHER"
    
    # Feature engineering configuration
    enable_polynomial_features: bool = False
    polynomial_degree: int = 2
    enable_interaction_terms: bool = False
    
    # Text processing configuration
    standardize_text: bool = True
    text_columns: Optional[List[str]] = None
    text_avg_length_threshold: int = 50
    text_max_length_threshold: int = 200
    text_uniqueness_ratio: float = 0.8
    text_min_length: int = 20

    # Columns to exclude from analysis
    no_process: Optional[List[str]] = None

    # Encoding configuration
    encode: Optional[List[str]] = None
    encode_method: Optional[str] = None
    explode_columns: bool = False

    def __post_init__(self):
        """Validate configuration values and convert string values to Enums"""
        self.cl = ColorLogger()
        # Convert string values to Enums if necessary
        if isinstance(self.scaling_method, str):
            try:
                self.scaling_method = ScalingMethod(self.scaling_method.lower())
            except ValueError:
                self.cl.warning(f"Invalid scaling_method '{self.scaling_method}'. Using default 'standard'")

                self.scaling_method = ScalingMethod.STANDARD

        if isinstance(self.null_strategy, str):
            try:
                self.null_strategy = NullStrategy(self.null_strategy.lower())
            except ValueError:
                self.cl.warning(f"Invalid null_strategy '{self.null_strategy}'. Using default 'mean'")
                self.null_strategy = NullStrategy.MEAN

        if isinstance(self.outlier_strategy, str):
            try:
                self.outlier_strategy = OutlierStrategy(self.outlier_strategy.lower())
            except ValueError:
                self.cl.warning(f"Invalid outlier_strategy '{self.outlier_strategy}'. Using default 'cap'")
                self.outlier_strategy = OutlierStrategy.CAP

        if isinstance(self.correlation_method, str):
            try:
                self.correlation_method = CorrelationMethod(self.correlation_method.lower())
            except ValueError:
                self.cl.warning(f"Invalid correlation_method '{self.correlation_method}'. Using default 'pearson'")
                self.correlation_method = CorrelationMethod.PEARSON
            
            if self.correlation_action not in ["ignore", "drop"]:
                self.cl.warning(f"Invalid correlation_action '{self.correlation_action}'. Using default 'ignore'")
                self.correlation_action = "ignore"

        # Validate numerical thresholds
        if not 0 <= self.null_threshold <= 1:
            self.cl.warning(f"Invalid null_threshold {self.null_threshold}. Setting to default 0.5")
            self.null_threshold = 0.5
            
        if self.z_score <= 0:
            self.cl.warning(f"Invalid z_score {self.z_score}. Setting to default 3.0")
            self.z_score = 3.0
            
        if not 0 <= self.correlation_threshold <= 1:
            self.cl.warning(f"Invalid correlation_threshold {self.correlation_threshold}. Setting to default 0.95")
            self.correlation_threshold = 0.95

        # Validate cardinality ratios
        if self.max_cardinality_ratio is not None:
            if not 0 <= self.max_cardinality_ratio <= 1:
                self.cl.warning(
                    f"Invalid max_cardinality_ratio {self.max_cardinality_ratio}. "
                    "Setting to default 0.95"
                )
                self.max_cardinality_ratio = 0.95
                
        if self.min_cardinality_ratio is not None:
            if not 0 <= self.min_cardinality_ratio <= 1:
                self.cl.warning(
                    f"Invalid min_cardinality_ratio {self.min_cardinality_ratio}. "
                    "Setting to default 0.01"
                )
                self.min_cardinality_ratio = 0.01
            
        # Only check ratio relationship if both are specified
        if self.min_cardinality_ratio is not None and self.max_cardinality_ratio is not None:
            if self.min_cardinality_ratio >= self.max_cardinality_ratio:
                self.cl.warning(
                    f"min_cardinality_ratio ({self.min_cardinality_ratio}) must be less than "
                    f"max_cardinality_ratio ({self.max_cardinality_ratio}). "
                    "Adjusting to default values."
                )
                self.min_cardinality_ratio = 0.01
                self.max_cardinality_ratio = 0.95
            
        if self.cardinality_action and self.cardinality_action.lower() not in [None, 'group', 'drop']:
            self.cl.warning(f"Invalid cardinality_action '{self.cardinality_action}'. Using default 'group'")
            self.cardinality_action = 'group'
        
        # Validate encoding configuration
        if self.encode_method and self.encode_method.lower() not in ['ohe', 'binary']:
            self.cl.warning(f"Invalid encoding method '{self.encode_method}'. Using default 'ohe'")
            self.encode_method = 'ohe'
        elif self.encode_method:
            self.encode_method = self.encode_method.lower()

        # Validate column filtering lists
        if (self.white_list and not all(isinstance(col, str) for col in self.white_list)) or \
            (self.black_list and not all(isinstance(col, str) for col in self.black_list)):
                raise ValueError("All elements in white_list and black_list must be strings")
                
        # if self.white_list and self.black_list:
        #     self.cl.error("Both white_list and black_list are set. Prioritizing white_list; black_list will be ignored.")
        #     self.black_list = None


    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PreprocessingConfig':
        """Create configuration from dictionary"""
        # Create a working copy to avoid modifying input
        config = config_dict.copy()
        
        # Enum mappings for easier conversion
        enum_fields = {
            'scaling_method': ScalingMethod,
            'null_strategy': NullStrategy,
            'outlier_strategy': OutlierStrategy,
            'correlation_method': CorrelationMethod
        }
        
        # Create a single instance to reference default values
        default_instance = cls()
        
        # Convert and validate Enum fields
        for field, enum_type in enum_fields.items():
            if field in config:
                try:
                    if isinstance(config[field], str):
                        config[field] = enum_type(config[field].lower())
                    elif not isinstance(config[field], enum_type):
                        raise ValueError(f"Expected str or {enum_type.__name__}")
                except ValueError as e:
                    self.cl.warning(
                        f"Invalid {field} '{config[field]}' ({str(e)}). "
                        f"Using default: {getattr(default_instance, field)}"
                    )
                    config[field] = getattr(default_instance, field)
        
        # Threshold validations with default fallbacks
        validations = {
            'null_threshold': (
                lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
                "must be a number between 0 and 1"
            ),
            'z_score': (
                lambda x: isinstance(x, (int, float)) and x > 0,
                "must be a positive number"
            ),
            'correlation_threshold': (
                lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
                "must be a number between 0 and 1"
            )
        }
        
        for field, (validator, error_msg) in validations.items():
            if field in config and not validator(config[field]):
                self.cl.warning(
                    f"Invalid {field} '{config[field]}' ({error_msg}). "
                    f"Using default: {getattr(default_instance, field)}"
                )
                config[field] = getattr(default_instance, field)
        
        return cls(**config)

    @classmethod
    def from_json(cls, json_path: str) -> 'PreprocessingConfig':
        """Create configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == 'cl':
                continue
            if isinstance(value, Enum):
                config_dict[key] = value.value
            elif isinstance(value, list):
                config_dict[key] = list(value)
            else:
                config_dict[key] = value
        return config_dict

    def save_config(self, path: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class CachedStats:
    """Consolidated cache for DataFrame statistics and metadata"""
    
    # Basic DataFrame metrics
    total_rows: int = 0
    num_columns: int = 0
    
    # Column type information - single source of truth
    column_types: Dict[str, ColumnType] = field(default_factory=dict)
    
    # Statistics
    null_counts: Dict[str, int] = field(default_factory=dict)
    null_percentages: Dict[str, float] = field(default_factory=dict)
    numeric_stats: Dict[str, Dict] = field(default_factory=dict)
    text_stats: Dict[str, Dict] = field(default_factory=dict)
    categorical_stats: Dict[str, Dict] = field(default_factory=dict)
    datetime_stats: Dict[str, Dict] = field(default_factory=dict)
    value_counts: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Correlation information
    correlation_matrix: Optional[np.ndarray] = None
    correlation_method: Optional[str] = None
    
    # Display tables
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def set_column_types(self, df: DataFrame, overrides: Dict[ColumnType, List[str]], config: PreprocessingConfig):
        """Single source of truth for column type detection and setting"""
        schema_dict = {field.name: field.dataType for field in df.schema.fields}
        df_cached = df.persist()
        
        try:
            # Get all stats in one pass for efficiency
            stats = df_cached.select([
                *[F.avg(F.length(c)).alias(f"{c}_avg_len") 
                  for c in df.columns if isinstance(schema_dict[c], StringType)],
                *[F.countDistinct(c).alias(f"{c}_distinct") 
                  for c in df.columns]
            ]).collect()[0]
            
            for column in df.columns:
                # 1. Check overrides first - absolute precedence
                col_type = next((type_ for type_, cols in overrides.items() 
                             if column in cols), None)
                             
                if col_type:
                    self.add_column(column, col_type)
                    continue
                
                # 2. Detect type from schema and content
                dtype = schema_dict[column]
                if isinstance(dtype, (IntegerType, FloatType, LongType, DoubleType, DecimalType)):
                    distinct_count = stats[f"{column}_distinct"]
                    col_type = ColumnType.BINARY if distinct_count == 2 else ColumnType.NUMERIC
                    
                elif isinstance(dtype, StringType):
                    avg_length = float(stats[f"{column}_avg_len"] or 0)
                    unique_count = stats[f"{column}_distinct"]
                    uniqueness_ratio = unique_count / self.total_rows if self.total_rows > 0 else 0
                    
                    col_type = ColumnType.TEXT if (
                        avg_length > config.text_avg_length_threshold or
                        (uniqueness_ratio > config.text_uniqueness_ratio and 
                         avg_length > config.text_min_length)
                    ) else ColumnType.CATEGORICAL
                    
                elif isinstance(dtype, (TimestampType, DateType)):
                    col_type = ColumnType.DATETIME
                elif isinstance(dtype, MapType):
                    col_type = ColumnType.MAP
                elif isinstance(dtype, ArrayType):
                    col_type = ColumnType.ARRAY
                elif isinstance(dtype, StructType):
                    col_type = ColumnType.STRUCT
                else:
                    col_type = ColumnType.CATEGORICAL
                
                self.add_column(column, col_type)
                
        finally:
            df_cached.unpersist()
    
    def add_column(self, column: str, column_type: ColumnType):
        """Add or update a column's type"""
        old_type = self.column_types.get(column)
        if old_type and old_type != column_type:
            print(f"Changing type for {column}: {old_type.value} -> {column_type.value}")
        self.column_types[column] = column_type

    def generate_all_tables(self):
        """Generate all display tables from cached statistics"""
        self.generate_overview_table()
        self.generate_missing_values_table()
        self.generate_numeric_stats_table()
        self.generate_text_stats_table()
        self.generate_categorical_stats_table()

    def generate_overview_table(self):
        """Generate overview table from column information"""
        columns_by_type = {
            'Numeric Columns': self.get_columns_by_type(ColumnType.NUMERIC),
            'Categorical Columns': self.get_columns_by_type(ColumnType.CATEGORICAL),
            'Text Columns': self.get_columns_by_type(ColumnType.TEXT),
            'Datetime Columns': self.get_columns_by_type(ColumnType.DATETIME),
            'Binary Columns': self.get_columns_by_type(ColumnType.BINARY),
            'Array Columns': self.get_columns_by_type(ColumnType.ARRAY),
            'Map Columns': self.get_columns_by_type(ColumnType.MAP),
            'Struct Columns': self.get_columns_by_type(ColumnType.STRUCT)
        }
        
        # Create the expanded data structure
        expanded_data = {}
        max_rows = max(len(cols) for cols in columns_by_type.values() if cols)
        
        # For each column type, include all columns
        for col_type, columns in columns_by_type.items():
            if columns:
                extended_values = columns + [''] * (max_rows - len(columns))
                expanded_data[col_type] = extended_values
        
        overview_df = pd.DataFrame(expanded_data)
        overview_df = (overview_df
                    .loc[:, (overview_df != '').any()]
                    .replace({pd.NA: '', 'nan': '', 'None': '', 'NaN': ''})
                    .dropna(how='all'))
        
        self.tables['overview'] = overview_df

    def generate_missing_values_table(self):
        """Generate missing values table"""
        null_data = []
        for col, null_pct in self.null_percentages.items():
            if null_pct > 0:
                null_data.append({
                    'Column': col,
                    'Missing %': f"{null_pct:.2f}%",
                    'Type': self.column_types[col].value
                })
        
        if null_data:
            self.tables['missing_values'] = (
                pd.DataFrame(null_data)
                .sort_values(by='Missing %', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                .reset_index(drop=True)
            )

    def generate_numeric_stats_table(self):
        """Generate numeric statistics table"""
        if not self.numeric_stats:
            return
            
        numeric_data = []
        for col, stats in self.numeric_stats.items():
            numeric_data.append({
                'Column': col,
                'Mean': f"{stats['mean']:.2f}" if stats['mean'] is not None else "N/A",
                'Std Dev': f"{stats['stddev']:.2f}" if stats['stddev'] is not None else "N/A",
                'Min': f"{stats['min']:.2f}" if stats['min'] is not None else "N/A",
                '25%': f"{stats['quartiles'][0]:.2f}" if stats['quartiles'] else "N/A",
                '50%': f"{stats['quartiles'][1]:.2f}" if stats['quartiles'] else "N/A",
                '75%': f"{stats['quartiles'][2]:.2f}" if stats['quartiles'] else "N/A",
                'Max': f"{stats['max']:.2f}" if stats['max'] is not None else "N/A"
            })
        
        if numeric_data:
            self.tables['numeric_stats'] = pd.DataFrame(numeric_data)

    def generate_text_stats_table(self):
        """Generate text statistics table"""
        if not self.text_stats:
            return
            
        text_data = []
        for col, stats in self.text_stats.items():
            avg_length = f"{stats['avg_length']:.2f}" if stats.get('avg_length') is not None else "N/A"
            
            if 'unique_count' in stats and self.total_rows > 0:
                uniqueness_pct = f"{(stats['unique_count'] / self.total_rows * 100):.1f}%"
            else:
                uniqueness_pct = "N/A"
                
            text_data.append({
                'Column': col,
                'Avg Length': avg_length,
                'Min Length': stats.get('min_length', "N/A"),
                'Max Length': stats.get('max_length', "N/A"),
                'Uniqueness': uniqueness_pct,
                'Sample Values': stats.get('sample_values', "N/A")
            })
        
        if text_data:
            self.tables['text_stats'] = (
                pd.DataFrame(text_data)
                .sort_values(by='Uniqueness', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                .reset_index(drop=True)
            )

    def generate_categorical_stats_table(self):
        """Generate categorical statistics table"""
        if not self.categorical_stats:
            return
                
        cat_data = []
        for col, stats in self.categorical_stats.items():
            top_values = stats['top_values']
            if not top_values:
                continue
                    
            cat_data.append({
                'Column': col,
                'Unique Values': stats['unique_count'],
                'Unique %': f"{(stats['unique_count'] / self.total_rows) * 100:.4f}%",
                'Null %': f"{self.null_percentages.get(col, 0):.4f}%",
                'Top 1': f"{top_values[0]['value']} ({top_values[0]['percentage']:.1f}%)" if len(top_values) > 0 else "N/A",
                'Top 2': f"{top_values[1]['value']} ({top_values[1]['percentage']:.1f}%)" if len(top_values) > 1 else "N/A",
                'Top 3': f"{top_values[2]['value']} ({top_values[2]['percentage']:.1f}%)" if len(top_values) > 2 else "N/A"
            })
        
        if cat_data:
            self.tables['categorical_stats'] = (
                pd.DataFrame(cat_data)
                .sort_values(by='Unique %', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
                .reset_index(drop=True)
            )

    def clear(self):
        """Clear all cached statistics"""
        self.__init__()
    
    def get_columns_by_type(self, column_type: ColumnType) -> List[str]:
        """Get list of columns for a specific type"""
        return [col for col, type_ in self.column_types.items() 
                if type_ == column_type]
    
    def is_initialized(self) -> bool:
        """Check if all necessary components are initialized"""
        # Check basic stats
        basic_stats_ready = (
            self.total_rows > 0 
            and self.num_columns > 0
            and bool(self.column_types)
        )
        
        # If basic stats aren't ready, return False
        if not basic_stats_ready:
            return False
        
        return True
    
    def add_column(self, column: str, column_type: ColumnType):
        """Add a column with its type"""
        self.column_types[column] = column_type

class SparkPreprocessor:
    """Main preprocessing class for Spark DataFrame transformation"""

    def __init__(self, config: Optional[Dict] = None, column_type_overrides: Optional[Dict[str, List[str]]] = None):
        """Initialize SparkPreprocessor class"""
        self.cl = ColorLogger("SparkPreprocessor")
        self.stats = CachedStats()
        self.config = PreprocessingConfig() if config is None else PreprocessingConfig.from_dict(config)
        self.fitted_params = {}
        self.original_schema = None
        self._is_fitted = False  
        self._raw_column_type_overrides = column_type_overrides  # Store raw overrides
        self.column_type_overrides = {}  # Will be populated in fit()

    def is_fitted(self) -> bool:
        """Check if the object has been fitted to data"""
        return self._is_fitted

    def _initialize_column_type_overrides(self, column_type_overrides: Optional[Dict[str, List[str]]] = None):
        """Initialize column type overrides without requiring DataFrame schema"""
        self._raw_column_type_overrides = column_type_overrides  # Simply store the raw overrides to be processed in fit()
        self.column_type_overrides = {}

    def fit(self, df: DataFrame, column_overrides: Optional[Dict[str, List[str]]] = None):
        """Fit preprocessor to the data"""
        self.cl.info(f"=== .fit() ===", color=Colors.GREEN)
        
        # Store original schema 
        self.original_schema = df.schema
        
        # Now validate overrides with actual schema
        overrides_to_validate = column_overrides or self._raw_column_type_overrides
        self.column_type_overrides = self._validate_column_type_overrides(overrides_to_validate)
        
        # Perform initial analysis
        self._initial_analysis(df, display_summary=False)
        self._is_fitted = True
        return self
        
        self.cl.warning(f"{self.column_type_overrides}")
        # Perform initial analysis
        self._initial_analysis(df, display_summary=False)
        self._is_fitted = True
        return self

    def _validate_column_type_overrides(self, overrides: Dict[str, List[str]] = None) -> Dict[ColumnType, List[str]]:
        """Validate and convert string type overrides to ColumnType enum"""
        if not overrides:
            return {}
        
        valid_overrides = {}

        cols_df = self.original_schema.names
        # Extract and handle 'ignore' list and filter lists if present
        ignore_cols = overrides.pop('ignore', []) if overrides else []

        if ignore_cols:
            # Set the no_process list in config
            if self.config.no_process is None:
                self.config.no_process = ignore_cols
            else:
                self.config.no_process.extend(ignore_cols)
        
        # Process remaining column type overrides
        for type_str, columns in overrides.items():
            override_columns = [c for c in columns if c in cols_df]
            if override_columns:            
                try:
                    # Convert string to enum directly using the name                
                    col_type = ColumnType[type_str.upper()]
                    valid_overrides[col_type] = override_columns
                    self.cl.info(f"Applying {col_type.value} override for selected columns")
                except KeyError:
                    self.cl.warning(f"Invalid column type override '{type_str}'. Valid types are: {[t.name.lower() for t in ColumnType]}")
                    continue
        
        return valid_overrides

    def transform(self, df: DataFrame, config: Optional[Dict] = None) -> DataFrame:
        """Transform the DataFrame with optional config update"""
        self.cl.info(f"=== .transform() ===", color=Colors.GREEN)
        # Update config if provided
        if config is not None:
            self._update_config(config)
        
        # Check if we need to refit due to config changes
        if not self.is_fitted():
            raise SystemExit("Data is not fitted. Use .fit() before .transform()")

        df_processed = df
        
        transformations = [
            ('👯 duplicates', self._handle_duplicates),
            ('🔍 missing_values', self._handle_missing_values),
            ('🧩 complex_types', self._handle_complex_types),
            ('🗂️ data_formatting', self._handle_datatypes),
            ('🦄 outliers', self._handle_outliers),
            ('📊 cardinality', self._handle_cardinality),
            ('🗝️ encoding', self.encode_columns),
            ('🔗 multicollinearity', self._handle_multicollinearity),
            ('⚖️ scaling', self._scale_features)
        ]

        timing_stats = []
        for name, transformation in transformations:
            try:
                formatted_name = self.color_text(name, Colors.GREEN, printout=False)
                self.cl.info(f"🛠️ Processing: {formatted_name}...")
                
                start_time = time.time()
                df_processed = transformation(df_processed)
                end_time = time.time()
                timing_stats.append((name, end_time - start_time))
                
                if df_processed is None:
                    raise ValueError(f"❌ Transformation {name} returned None")
            except Exception as e:
                self.cl.error(f"❌ Error in {name} transformation: {str(e)}")
                raise
        
        self._print_time_statistics(timing_stats)
        return df_processed
    
    def fit_transform(self, df: DataFrame, column_overrides: Optional[Dict[str, List[str]]] = None, config: Optional[Dict] = None) -> DataFrame:
        """Fit to data, then transform it"""
        # Update config first if provided
            
        self.fit(df=df, column_overrides=column_overrides)
        return self.transform(df=df, config=config)
    
    def _cache_basic_metrics(self, df: DataFrame):
        """Cache basic DataFrame metrics"""
        try:
            self.columns = df.columns
            df_cached = df.persist()
            
            # Get basic counts and null stats
            null_stats = df_cached.select([
                F.count('*').alias('total_rows'),
                *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) 
                for c in df_cached.columns]
            ]).collect()[0]
            
            self.stats.total_rows = null_stats['total_rows']
            self.stats.num_columns = len(self.columns)

            self.cl.info(f"Fitting data with {self.stats.total_rows:,} rows and {self.stats.num_columns:,} columns", Colors.CYAN)
            
            # Cache null statistics
            for column in df_cached.columns:
                null_count = null_stats[column]
                self.stats.null_counts[column] = null_count
                self.stats.null_percentages[column] = (null_count / self.stats.total_rows) * 100 if self.stats.total_rows > 0 else 0
            
            # Set column types in a single place
            self.stats.set_column_types(df_cached, self.column_type_overrides, self.config)
            
            df_cached.unpersist()
            
        except Exception as e:
            self.cl.error(f"Error in _cache_basic_metrics: {str(e)}")
            # Set default values on error
            self.stats.total_rows = 0
            self.stats.num_columns = len(df.columns)
            self.stats.null_counts = {col: 0 for col in df.columns}
            self.stats.null_percentages = {col: 0 for col in df.columns}

    def _print_time_statistics(self, timing_stats: List[Tuple[str, float]]) -> None:
        """Print timing statistics for each transformation"""
        header = "=== Timing Statistics ==="
        line = len(header)
        self.color_text(f"\n{header}", Colors.BLUE)
        total_time = sum(t[1] for t in timing_stats)
        for name, exec_time in timing_stats:
            percentage = (exec_time / total_time) * 100
            if 0 <= percentage <= 33:
                color = Colors.GREEN
            if 33 < percentage <= 66:
                color = Colors.LIGHT_ORANGE
            if 66 < percentage <= 100:
                color = Colors.BRICK
            time = f"{exec_time:.2f}s ({percentage:.2f}%)"
            colored_time = self.color_text(f"{time}", color, printout=False)
            self.color_text(f"  {name}: {colored_time}")
            line = max(line, len(name+time)+6)
        self.color_text(f"{'-'*line}\n Total transformation time: {total_time:.2f}s", Colors.YELLOW)

    def _validate_config_entry(self, key: str, value: Any) -> Any:
        """Validate a single config entry and return converted value or raise ValueError."""
        
        # Optional List[str] validations
        if key in ['white_list', 'black_list', 'no_process', 'encode', 'text_columns']:
            if value is None:
                return None
            if not isinstance(value, (list, tuple)):
                value = [value]
            if not all(isinstance(x, str) for x in value):
                msg = f"CONFIG ERROR: All elements in {key} must be strings"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return list(value)
        
        # Enum validations
        enum_mappings = {
            'scaling_method': ScalingMethod,
            'null_strategy': NullStrategy,
            'outlier_strategy': OutlierStrategy,
            'correlation_method': CorrelationMethod
        }
        if key in enum_mappings:
            if isinstance(value, str):
                try:
                    return enum_mappings[key](value.lower())
                except ValueError:
                    msg = f"CONFIG ERROR: Invalid '{key}': '{value}'"
                    raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            elif not isinstance(value, enum_mappings[key]):
                msg = f"CONFIG ERROR: '{key}' must be string or '{enum_mappings[key].__name__}'"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return value
        
        # Float ratio validations (0 to 1)
        if key in ['null_threshold', 'correlation_threshold', 'text_uniqueness_ratio']:
            if not isinstance(value, (int, float)):
                msg = f"CONFIG ERROR: '{key}' must be numeric"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            if not 0 <= value <= 1:
                msg = f"CONFIG ERROR: '{key}' must be between 0 and 1, got {value}"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return float(value)
        
        # Optional float ratio validations
        if key in ['max_cardinality_ratio', 'min_cardinality_ratio']:
            if value is None:
                return None
            if not isinstance(value, (int, float)):
                msg = f"CONFIG ERROR: '{key}' must be numeric or None"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            if not 0 <= value <= 1:
                msg = f"CONFIG ERROR: '{key}' must be between 0 and 1, got {value}"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return float(value)
        
        # Positive float validations
        if key in ['z_score']:
            if not isinstance(value, (int, float)):
                msg = f"CONFIG ERROR: '{key}' must be numeric"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))

            if value <= 0:
                msg = f"CONFIG ERROR: '{key}' must be positive, got {value}"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return float(value)
        
        # Boolean validations
        if key in ['enable_polynomial_features', 'enable_interaction_terms', 
                'standardize_text', 'explode_columns']:
            if not isinstance(value, bool):
                msg = f"CONFIG ERROR: '{key}' must be boolean"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return value
        
        # Positive integer validations
        if key in ['polynomial_degree', 'text_avg_length_threshold', 
                'text_max_length_threshold', 'text_min_length']:
            if not isinstance(value, int):
                msg = f"CONFIG ERROR: '{key}' must be integer"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            if value <= 0:
                msg = f"CONFIG ERROR: '{key}' must be positive"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return value
        
        # String validations
        if key in ['cardinality_action', 'other_label', 'correlation_action']:
            if not isinstance(value, str):
                msg = f"{key} must be string"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            if key == 'cardinality_action' and value not in ['group', 'drop']:
                msg = f"CONFIG ERROR: 'cardinality_action' must be 'group' or 'drop'"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            if key == 'correlation_action' and value not in ['ignore', 'drop']:
                msg = f"CONFIG ERROR: 'cardinality_action' must be 'ignore' or 'drop'"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return value
        
        # Optional string validations
        if key == 'encode_method':
            if value is None:
                return None
            if not isinstance(value, str):
                msg = f"CONFIG ERROR: 'encode_method' must be string or None"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            if value not in ['ohe', 'binary']:
                msg=f"CONFIG ERROR: 'encode_method' must be 'ohe' or 'binary'"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return value
        
        # Optional mixed type validations
        if key == 'null_fill_value':
            if value is None:
                return None
            if not isinstance(value, (str, int, float)):
                msg=f"CONFIG ERROR: 'null_fill_value' must be string, numeric, or None"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            return value
        
    
        msg=f"CONFIG ERROR: Unknown config parameter: {key}"
        raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
        
    def _validate_config(self, new_config: Dict) -> Dict:
        """Validate all config entries and return validated dictionary."""
        validated_config = {}
        for key, value in new_config.items():
            if not hasattr(self.config, key):
                msg=f"CONFIG ERROR: Unknown config parameter: {key}: {value}"
                raise SystemExit(self.color_text(msg=msg, color=Colors.RED, printout=True, return_colored=False))
            validated_config[key] = self._validate_config_entry(key, value)
        return validated_config
    
    def _update_config(self, new_config: Optional[Dict] = None) -> None:
        """Update config with validated values and reset fitted state if config changes"""
        if new_config is None:
            return None

        try:
            validated_config = self._validate_config(new_config)
            
            # Check if any config values are actually changing
            config_changed = False
            for key, validated_value in validated_config.items():
                current_value = getattr(self.config, key)
                if current_value != validated_value:
                    config_changed = True
                    setattr(self.config, key, validated_value)
                    
            # If config changed and we were fitted, we need to reset (ooor?)
            # if config_changed and self._is_fitted:
            #     self._is_fitted = False
            #     self.stats.clear()
            #     self.fitted_params = {}

        except ValueError as e:
            error = str(e).splitlines()[0]
            self.color_text(f"Error: {error}", color=Colors.RED)
       
    def _initial_analysis(self, df: DataFrame, display_summary: bool = True):
        """Perform initial analysis"""
        if self.original_schema is None:
            self.original_schema = df.schema
        
        self.columns = df.columns

        df_filtered = self._filter_columns(df)
        self._cache_basic_metrics(df_filtered)
        self._analyze_data(df_filtered)
        # self.calculate_correlation_matrix(df_filtered)
        self.stats.generate_all_tables()
        if display_summary:
            self.display_summary(df_filtered)
    
    def display_summary(self, df: DataFrame = None, table_format='pretty'):
        """Display comprehensive summary statistics using cached data"""
        if not self.is_fitted():
            if df is None:
                raise ValueError("No cached analysis available and no DataFrame provided")
            self.cl.info("No cached analysis found. Analyzing data...")
            # self._cache_basic_metrics(df)
            self._analyze_data(df)
            self.stats.generate_all_tables()
        else:
            self.cl.info("Using cached analysis")
        
        self._display_all_tables(table_format)
    
    def _display_all_tables(self, table_format='pretty'):
        """Display all cached tables with formatting"""
        header_color = Colors.GREEN
        
        table_definitions = [
            ('overview', 'Dataset Overview'),
            ('numeric_stats', 'Numeric Column Statistics'),
            ('missing_values', 'Missing Values Analysis'),
            ('text_stats', 'Text Column Statistics'),
            ('categorical_stats', 'Categorical Column Statistics')
        ]
        
        for table_name, content in table_definitions:
            if table_name in self.stats.tables:
                self.color_text(f"\n=== {content} ===", header_color)

                # Check if table is 'overview' and modify cell values with an asterisk
                if table_name == 'overview':
                    has_asterisk = False
                    print(f"Num. rows: {self.stats.total_rows:,}\nNum. columns: {self.stats.num_columns:,}")
                    
                    if self.config.no_process:
                        # Copy the overview table to avoid modifying the original cached data
                        display_table = self.stats.tables[table_name].copy()
                        
                        # Process each column for asterisk marking
                        for col in display_table.columns:
                            modified_column = []
                            for cell in display_table[col]:
                                if cell in self.config.no_process:
                                    modified_column.append(f"* {cell}")
                                    has_asterisk = True
                                else:
                                    modified_column.append(cell)
                            display_table[col] = modified_column
                        
                        if has_asterisk:
                            self.color_text(
                                "Columns marked with an asterisk (*) will not be processed for transformations",
                                Colors.YELLOW
                            )
                    else:
                        display_table = self.stats.tables[table_name]
                else:
                    display_table = self.stats.tables[table_name]
                
                # Display the table with formatting
                try:
                    print(tabulate(display_table, headers='keys', tablefmt=table_format))
                except Exception as e:
                    self.cl.warning(f"Error displaying {table_name} table: {str(e)}")

    def _calculate_thread_count(self, df: DataFrame, max_threads: int = 64, multiplier: float = 1) -> int:
        """Calculate optimal number of threads based on CPU count and workload."""
        # Get base CPU count
        cpu_count = multiprocessing.cpu_count()
        
        # Number of columns to process
        num_columns = len(df.columns)
        
        # Calculate base thread count (2 threads per core is common practice)
        base_threads = cpu_count * multiplier
        
        # Add limits
        MIN_THREADS = 2  # At least 2 threads
        MAX_THREADS = int(max_threads)  # Avoid excessive threading

        # Buffer as 20% of columns
        thread_buffer = int(num_columns * 0.2)
        
        # Calculate optimal thread count
        optimal_threads = min(
            max(MIN_THREADS, base_threads),  # At least MIN_THREADS
            MAX_THREADS,                     # At most MAX_THREADS
            max(num_columns + thread_buffer, MIN_THREADS)    # No more than columns to process but no less than MIN_THREADS
        )
        
        return int(((optimal_threads + 7) // 8) * 8)
        
    def _analyze_data(self, df: DataFrame):
        """Calculate detailed statistics for each column based on its type"""
        self.cl.info("Starting multithread optimized analysis...")
        
        analysis_functions = {
            ColumnType.NUMERIC: self._cache_numeric_stats,
            ColumnType.CATEGORICAL: self._cache_categorical_stats,
            ColumnType.BINARY: self._cache_categorical_stats,
            ColumnType.TEXT: self._cache_text_stats,
            ColumnType.DATETIME: self._cache_datetime_stats,
            ColumnType.MAP: lambda df, col: None,
            ColumnType.ARRAY: lambda df, col: None,
            ColumnType.STRUCT: lambda df, col: None
        }
        
        def process_column(column):
            """Process detailed statistics for a single column"""
            try:
                col_type = self.stats.column_types.get(column)
                if not col_type:
                    return False
                    
                if self._should_process_column(column) and col_type in analysis_functions:
                    analysis_functions[col_type](df, column)
                return True
                
            except Exception as e:
                self.cl.error(f"Error processing column {column}: {str(e)}")
                return False
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = self._calculate_thread_count(df)
        self.cl.text(f"[{max_workers} threads allocated.]")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_column, col) for col in df.columns]
            colored_desc = f"{Colors.BLUE}Processing columns{Colors.DEFAULT}"
            for _ in tqdm(concurrent.futures.as_completed(futures), 
                        total=len(futures), 
                        desc=colored_desc, 
                        position=0, 
                        leave=True,
                        ncols=100):
                pass
        
        # Calculate correlation matrix if needed
        numeric_cols = self.stats.get_columns_by_type(ColumnType.NUMERIC)
        processable_numeric_cols = [col for col in numeric_cols 
                                if self._should_process_column(col)]
        
        if len(processable_numeric_cols) > 1:
            self._cache_correlation_matrix(df.select(processable_numeric_cols))

    def _validate_columns_for_encoding(self, df: DataFrame, columns: List[str]) -> List[str]:
        """Helper method to validate columns for encoding"""
        return [
            col for col in columns
            if (col in df.columns and 
                self._should_process_column(col) and
                self.get_column_type(col) in [ColumnType.CATEGORICAL, ColumnType.BINARY])
        ]

    def _get_distinct_values(self, df: DataFrame, column: str):
        """Helper function to get distinct values for a column efficiently"""
        return (df.select(column)
                .filter(F.col(column).isNotNull())
                .distinct()
                .orderBy(column)
                .rdd
                .map(lambda row: row[column])
                .collect())

    def encode_columns(self, df: DataFrame, encode: Union[str, List[str]] = None, encoding_type: str = None, explode_columns: bool = False) -> DataFrame:
        """Encode the specified columns using the specified encoding type."""
        if isinstance(encode, str):
            encode = [encode]
        
        stages = []
        index_cols = []
        handle_invalid = "keep"
        drop_original = True
        max_workers = self._calculate_thread_count(df)

        encoding_type = encoding_type or self.config.encode_method
        explode_columns = explode_columns or self.config.explode_columns
        columns = encode or self.config.encode

        if encoding_type is None:
            self.cl.text("No encoding type specified. Skipping step.")
            return df
        else:
            self.cl.text(f"Encoding method: {encoding_type.upper()}")

        columns = self._validate_columns_for_encoding(df, columns)
        
        for col in columns:
            output_col = f"{col}_index"
            indexer = StringIndexer(
                inputCol=col,
                outputCol=output_col,
                handleInvalid=handle_invalid,
                stringOrderType="frequencyDesc"
            )
            stages.append(indexer)
            index_cols.append(output_col)
        
        pipeline = Pipeline(stages=stages)
        df_encoded = pipeline.fit(df).transform(df)
        
        if encoding_type == "ohe":
            encoder_stages = []
            vec_cols = []
            
            for col in index_cols:
                base_col = col.replace("_index", "")
                output_col = f"{base_col}_vec"
                encoder = OneHotEncoder(
                    inputCol=col,
                    outputCol=output_col,
                    dropLast=False
                )
                encoder_stages.append(encoder)
                vec_cols.append(output_col)
            
            encoder_pipeline = Pipeline(stages=encoder_stages)
            df_encoded = encoder_pipeline.fit(df_encoded).transform(df_encoded)
            
            if explode_columns:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_col = {
                        executor.submit(self._get_distinct_values, df, col): (col, col, vec_col)
                        for col, vec_col in zip(columns, vec_cols)
                    }
                    
                    all_dummy_columns = []
                    for future in future_to_col:
                        col, base_col, vec_col = future_to_col[future]
                        distinct_values = future.result()
                        
                        dummy_columns = [
                            (F.element_at(vector_to_array(F.col(vec_col)), idx + 1).cast("integer")
                            .alias(f"{base_col}_ohe_{clean_category}"))
                            for idx, category in enumerate(distinct_values)
                            for clean_category in ["".join(c if c.isalnum() else "_" for c in str(category)).rstrip("_")]
                        ]
                        
                        if handle_invalid == "keep":
                            dummy_columns.append(F.when(F.col(col).isNull(), 1).otherwise(0).cast("integer").alias(f"{base_col}_ohe_NULL"))

                        
                        all_dummy_columns.extend(dummy_columns)
                    
                    df_encoded = df_encoded.select("*", *all_dummy_columns)
            else:
                # Keep vectors as is, just rename them appropriately
                for col, vec_col in zip(columns, vec_cols):
                    df_encoded = df_encoded.withColumnRenamed(vec_col, f"{col}_ohe")
                    
            cols_to_drop = index_cols
            if explode_columns:
                cols_to_drop.extend(vec_cols)

        elif encoding_type == "binary":
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def get_max_index(df, col):
                    return int(df.agg({col: "max"}).first()[0])
                
                future_to_col = {
                    executor.submit(get_max_index, df_encoded, col): (col, col.replace("_index", ""))
                    for col in index_cols
                }
                
                binary_columns = []
                for future in future_to_col:
                    col, base_col = future_to_col[future]
                    max_index = future.result()
                    num_bits = max_index.bit_length()
                    
                    def binary_list(value):
                        if value is None:
                            return [None] * num_bits
                        binary_str = bin(int(value))[2:].zfill(num_bits)
                        return [int(bit) for bit in binary_str]
                    
                    binary_udf = F.udf(binary_list, ArrayType(IntegerType()))
                    
                    binary_columns.extend([
                        (F.element_at(binary_udf(F.col(col)), i + 1).cast("integer")
                        .alias(f"{base_col}_bin_{i}"))
                        for i in range(num_bits)
                    ])
                
                df_encoded = df_encoded.select("*", *binary_columns)
                cols_to_drop = index_cols
        
        if drop_original:
            cols_to_drop.extend(columns)
        
        return df_encoded.drop(*cols_to_drop)

    def _scale_features(self, df):
        """
        Scales numeric features in the DataFrame using the configured scaling method.
        """
        if not isinstance(df, DataFrame):
            raise TypeError("Input df is not a PySpark DataFrame.")

        # Step 1: Check scaling method
        if self.config.scaling_method == ScalingMethod.NONE:
            return df
        
        self.cl.info(f"Scaling numeric features using {self.config.scaling_method.value.upper()} scaler.")
        
        # Step 2: Check if the scaler is already fitted
        if not self.is_fitted():
            self._analyze_data(df)

        # Step 3: Get numeric columns to scale
        numeric_cols = [col for col in self.stats.get_columns_by_type(ColumnType.NUMERIC) if self._should_process_column(col)]
        if not numeric_cols:
            return df

        try:
            # Step 4: Create feature vector for scaling
            df = VectorAssembler(
                inputCols=numeric_cols,
                outputCol="features_vec"
            ).transform(df)

            # Step 5: Fit and apply scaler
            scaler = self._get_scaler(self.config.scaling_method.value)
            scaler_model = scaler.fit(df)
            df = scaler_model.transform(df)
            self.fitted_params['scaler'] = scaler_model

            # Step 6: Convert scaled features to array
            if 'scaled_features' not in df.columns:
                raise ValueError("Column 'scaled_features' not found after scaling transformation.")
            
            df = df.withColumn("scaled_features_array", vector_to_array(F.col("scaled_features")))

            scaled_columns = list()
            # Step 7: Create individual scaled columns
            for i, col_name in enumerate(numeric_cols):
                df = df.withColumn(f"{col_name}_scaled", F.col("scaled_features_array")[i])
                scaled_columns.append(f"{col_name}_scaled")

            self.cl.text(f"➕ Added scaled columns: {scaled_columns}", Colors.LIGHT_ORANGE)

            # Step 8: Drop temporary columns and return final DataFrame
            df = df.drop("features_vec", "scaled_features", "scaled_features_array")
            self.columns = df.columns
            return df

        except Exception as e:
            raise RuntimeError(f"Error during feature scaling: {str(e)}")
    
    def _get_numeric_columns(self, df):
        """Get numeric and encoded columns to scale"""
        numeric_cols = [col for col in self.stats.get_columns_by_type(ColumnType.NUMERIC)
                        if self._should_process_column(col)]
                        
        if hasattr(self, 'encoded_columns'):
            numeric_cols.extend([col for col in self.encoded_columns 
                                if col in df.columns])
        return numeric_cols

    def _get_scaler(self, scaling_type):
        """Get appropriate scaler based on scaling type"""
        if scaling_type == "minmax":
            return MinMaxScaler(inputCol="features_vec", 
                                outputCol="scaled_features")
        elif scaling_type == "standard":
            return StandardScaler(inputCol="features_vec",
                                outputCol="scaled_features",
                                withMean=True, 
                                withStd=True)
        else:
            raise ValueError(f"Invalid scaling_type: {scaling_type}")
          
    def _cache_numeric_stats(self, df: DataFrame, column: str):
        """Cache comprehensive statistics for numeric columns using a single query"""
        # Optimize by computing all statistics in a single pass
        stats = df.select(
            F.mean(column).alias('mean'),
            F.stddev(column).alias('stddev'),
            F.min(column).alias('min'),
            F.max(column).alias('max'),
            F.expr(f'percentile({column}, array(0.25, 0.5, 0.75))').alias('quartiles'),
            F.skewness(column).alias('skewness'),
            F.kurtosis(column).alias('kurtosis'),
            F.variance(column).alias('variance')
        ).collect()[0]
        
        self.stats.numeric_stats[column] = {
            'mean': float(stats['mean']) if stats['mean'] is not None else None,
            'stddev': float(stats['stddev']) if stats['stddev'] is not None else None,
            'min': float(stats['min']) if stats['min'] is not None else None,
            'max': float(stats['max']) if stats['max'] is not None else None,
            'quartiles': [float(x) for x in stats['quartiles']] if stats['quartiles'] is not None else None,
            'skewness': float(stats['skewness']) if stats['skewness'] is not None else None,
            'kurtosis': float(stats['kurtosis']) if stats['kurtosis'] is not None else None,
            'variance': float(stats['variance']) if stats['variance'] is not None else None
        }

    def _cache_categorical_stats(self, df: DataFrame, column: str):
        """Cache comprehensive statistics for categorical columns"""
        # Skip processing for MAP type columns
        field = [field for field in df.schema.fields if field.name == column][0]
        if isinstance(field.dataType, MapType):
            self.stats.categorical_stats[column] = {
                'unique_count': None,
                'top_values': []
            }
            return

        try:
            value_counts = (df.groupBy(column)
                        .count()
                        .orderBy('count', ascending=False)
                        .collect())
            
            self.stats.value_counts[column] = value_counts
            unique_count = len(value_counts)
            
            L = 33
            self.stats.categorical_stats[column] = {
                'unique_count': unique_count,
                'top_values': [
                    {
                        'value': (str(row[column])[:L] + "[...]" if row[column] is not None and len(str(row[column])) > L 
                                else str(row[column]) if row[column] is not None 
                                else 'NULL'),
                        'count': row['count'],
                        'percentage': (row['count'] / self.stats.total_rows) * 100
                    }
                    for row in value_counts[:10]
                ]
            }
        except Exception as e:
            self.cl.error(f"Error processing column {column}: {str(e)}")
            # Set default values for error cases
            self.stats.categorical_stats[column] = {
                'unique_count': None,
                'top_values': []
            }

    def _cache_text_stats(self, df: DataFrame, column: str):
        """Cache comprehensive statistics for text columns using a single query"""
        # Optimize by computing all statistics in a single pass
        stats = df.select(
            F.avg(F.length(column)).alias('avg_length'),
            F.min(F.length(column)).alias('min_length'),
            F.max(F.length(column)).alias('max_length'),
            F.countDistinct(column).alias('unique_count')
        ).collect()[0]
        
        # Get sample values 
        sample_values = (df.select(column)
                        .distinct()
                        .limit(3)
                        .collect())
        
        # Combine sample values, truncate each to L characters and join with ", "
        L = 137
        sample_values_combined = "; ".join(
            [str(row[column])[:L] for row in sample_values if row[column] is not None]
        )
        
        # Check if the total length exceeds L characters
        if len(sample_values_combined) > L:
            # Trim to L characters and append "[...]" if necessary
            sample_values_combined = sample_values_combined[:L] + "[...]"

        # Store the stats in the dictionary
        self.stats.text_stats[column] = {
            'avg_length': float(stats['avg_length']) if stats['avg_length'] is not None else None,
            'min_length': int(stats['min_length']) if stats['min_length'] is not None else None,
            'max_length': int(stats['max_length']) if stats['max_length'] is not None else None,
            'unique_count': int(stats['unique_count']),
            'sample_values': sample_values_combined
        }

    def _cache_datetime_stats(self, df: DataFrame, column: str):
        """Cache comprehensive statistics for datetime columns using a single query"""
        # Optimize by computing all statistics in a single pass
        stats = df.select(
            F.min(column).alias('min_date'),
            F.max(column).alias('max_date'),
            F.countDistinct(column).alias('unique_count'),
            F.datediff(F.max(column), F.min(column)).alias('date_range')
        ).collect()[0]
        
        self.stats.datetime_stats[column] = {
            'min_date': str(stats['min_date']) if stats['min_date'] is not None else None,
            'max_date': str(stats['max_date']) if stats['max_date'] is not None else None,
            'unique_count': int(stats['unique_count']),
            'date_range_days': int(stats['date_range']) if stats['date_range'] is not None else None
        }

    def _cache_correlation_matrix(self, df: DataFrame):
        """Cache correlation matrix for numeric columns"""
        if self.config.correlation_method == CorrelationMethod.NONE:
            return df

        numeric_cols = self.stats.get_columns_by_type(ColumnType.NUMERIC)  # Corregido
        if not numeric_cols or len(numeric_cols) < 2:
            return df
        
        if self.config.correlation_method == CorrelationMethod.PEARSON:
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol="features",
                handleInvalid="skip"
            )
            df_vector = assembler.transform(df)
            correlation_matrix = Correlation.corr(df_vector, "features").collect()[0][0].toArray()
        elif self.config.correlation_method == CorrelationMethod.CHATTERJEE:  # Chatterjee correlation
            correlation_matrix = self._calculate_chatterjee_correlation_matrix(df, numeric_cols)
        else:
            self.cl.error(f"Invalid correlation method: {self.config.correlation_method}")
        
        self.stats.correlation_matrix = correlation_matrix
        self.stats.correlation_method = self.config.correlation_method.value

    def _handle_complex_types(self, df):
        """Handle complex data types like arrays, maps, and structs"""
        result_df = df
        
        for column in df.columns:
            if not self._should_process_column(column):
                continue
                
            col_type = self.get_column_type(column)
            if col_type == ColumnType.ARRAY:
                # Get the element type of the array
                array_type = df.schema[column].dataType
                
                if isinstance(array_type.elementType, StringType):
                    # For arrays of strings, use concat_ws
                    result_df = result_df.withColumn(
                        f"{column}_str", 
                        F.concat_ws(",", F.col(column))
                    )
                elif isinstance(array_type.elementType, StructType):
                    # For arrays of structs, convert each struct to a string representation
                    result_df = result_df.withColumn(
                        f"{column}_str",
                        F.expr(f"transform({column}, x -> to_json(x))")
                    )
                else:
                    # For other array types, convert elements to string
                    result_df = result_df.withColumn(
                        f"{column}_str",
                        F.expr(f"transform({column}, x -> cast(x as string))")
                    )
                    
            elif col_type == ColumnType.MAP:
                result_df = result_df.withColumn(
                    f"{column}_keys", 
                    F.map_keys(F.col(column))
                )
                result_df = result_df.withColumn(
                    f"{column}_values", 
                    F.map_values(F.col(column))
                )
                
            elif col_type == ColumnType.STRUCT:
                # Convert struct to JSON string
                result_df = result_df.withColumn(
                    f"{column}_str",
                    F.to_json(F.col(column))
                )
        
        return result_df
    
    def _handle_missing_values(self, df: DataFrame) -> DataFrame:
        """Handle missing values using cached statistics"""
        if self.config.null_strategy == NullStrategy.NONE:
            self.cl.text("'null_strategy' is set to 'none'. Skipping step.")
            return df

        # self.cl.text("⚙️ Handling missing values")
        cols_to_process = {}
        cols_to_drop = []

        for col in df.columns:
            null_pct = self.stats.null_percentages.get(col, 0)
            if null_pct > self.config.null_threshold * 100:
                self.cl.warning(f"🗑️ Dropping column '{col}' due to high null percentage: {null_pct:.2f}% (threshold: {self.config.null_threshold * 100}%)")
                cols_to_drop.append(col)
            else:
                col_type = self.stats.column_types.get(col)
                if col_type:
                    cols_to_process[col] = col_type

        # Handle column drops if any
        if cols_to_drop:
            df = df.drop(*cols_to_drop)
            self.columns = df.columns
            
            # Update column info
            for col in cols_to_drop:
                if col in self.stats.column_types:
                    del self.stats.column_types[col]
        else:
            self.cl.text(f"Null counts under threshold for all columns (null_threshold: {self.config.null_threshold * 100}%)")

        # Prepare fill values for batch processing
        fill_values = {}
        flag_columns = []

        for col, col_type in cols_to_process.items():
            if self.config.null_strategy == NullStrategy.FLAG:
                flag_columns.append(col)
                continue

            fill_value = None
            if col_type == ColumnType.NUMERIC and col in self.stats.numeric_stats:
                stats = self.stats.numeric_stats[col]
                if self.config.null_strategy == NullStrategy.MEAN:
                    fill_value = stats['mean']
                elif self.config.null_strategy == NullStrategy.MEDIAN:
                    fill_value = stats['quartiles'][1]

            elif col_type in [ColumnType.CATEGORICAL, ColumnType.BINARY]:
                if col in self.stats.categorical_stats:
                    stats = self.stats.categorical_stats[col]
                    if stats['top_values']:
                        fill_value = stats['top_values'][0]['value']

            if fill_value is not None:
                fill_values[col] = fill_value

        # Batch process fill values if any
        if fill_values:
            df = df.fillna(fill_values)

        # Batch process flag columns if any
        if flag_columns and self.config.null_strategy == NullStrategy.FLAG:
            for col in flag_columns:
                df = df.withColumn(f"{col}_is_null", F.col(col).isNull().cast("integer"))

        return df
       
    def _handle_datatypes(self, df):
        """Handle data type conversions"""
        if not self.is_fitted():
            self._analyze_data(df)
        
        for column, detected_type in self.stats.column_types.items():
            # if not self._should_process_column(column):
            #    continue
                
            try:
                if detected_type == ColumnType.NUMERIC:
                    df = df.withColumn(column, F.col(column).cast("double"))
                elif detected_type == ColumnType.DATETIME:
                    df = df.withColumn(column, F.to_timestamp(F.col(column)))
                elif detected_type == ColumnType.BINARY:
                    distinct_vals = [row[0] for row in df.select(column).distinct().collect() 
                                if row[0] is not None]
                    if len(distinct_vals) == 2:
                        df = df.withColumn(column, 
                                        F.when(F.col(column) == distinct_vals[0], True)
                                        .otherwise(False))
                elif detected_type in [ColumnType.CATEGORICAL, ColumnType.TEXT]:
                    df = df.withColumn(column, F.col(column).cast("string"))
                    if detected_type == ColumnType.TEXT and self.config.standardize_text:
                        df = df.withColumn(column, 
                                        F.regexp_replace(F.lower(F.col(column)), 
                                                        "[^a-z0-9\\s]", " "))
            except Exception as e:
                self.cl.warning(f"Error converting column {column} to type {detected_type}: {str(e)}")
                continue
        
        return df

    def _handle_duplicates(self, df):
        """Remove duplicate rows while handling MAP type columns"""
        # self.cl.text("⚙️ Handling duplicates")
        
        # Get all non-MAP type columns
        map_columns = []
        non_map_columns = []
        
        for field in df.schema.fields:
            if isinstance(field.dataType, MapType):
                map_columns.append(field.name)
            else:
                non_map_columns.append(field.name)
        
        if map_columns:
            if non_map_columns:
                self.cl.text(f"🧹 Found MAP type columns. Dropping duplicated rows only on non-MAP columns.")
                # Drop duplicates using only non-MAP columns
                df = df.dropDuplicates(non_map_columns)
        else:
            # If no MAP columns, proceed with normal duplicate dropping
            self.cl.text(f"🧹 Dropping duplicated rows.")
            df = df.dropDuplicates()
        
        return df

    def _handle_cardinality(self, df: DataFrame) -> DataFrame:
        """Handle cardinality by either grouping or dropping columns based on ratio thresholds."""
        if self.config.cardinality_action is None:
            self.cl.text("No cardinality action specified. Skipping step.")
            return df
        else:
            self.cl.text(f"Action selected for cardinality handling: '{self.config.cardinality_action}'")
        
        def process_column(col: str) -> Tuple[str, int, int, bool]:
            """Process a single column and return modulo parameters and drop flag."""
            if col not in self.stats.categorical_stats:
                return col, 0, 0, False
                
            stats = self.stats.categorical_stats[col]
            unique_count = stats['unique_count']
            total_rows = self.stats.total_rows
            
            if total_rows == 0:
                return col, 0, 0, False
                
            cardinality_ratio = unique_count / total_rows
            precision_factor = 3  # Balance between precision and performance
            
            if ((self.config.max_cardinality_ratio is not None and 
                cardinality_ratio > self.config.max_cardinality_ratio) or
                (self.config.min_cardinality_ratio is not None and 
                cardinality_ratio < self.config.min_cardinality_ratio)):
                
                if self.config.cardinality_action == 'drop':
                    return col, 0, 0, True
                
                # Calculate target ratio (using max if available, otherwise min)
                target_ratio = (self.config.max_cardinality_ratio 
                            if self.config.max_cardinality_ratio is not None 
                            else self.config.min_cardinality_ratio)
                
                # Calculate modulo parameters
                modulo_base = int(2 * precision_factor / target_ratio)
                keep_remainder = int(2 * precision_factor * target_ratio)
                
                self.cl.text(f"🧹 Column '{col}': Using modulo hash grouping to approximate {target_ratio:.4f} ratio")
                return col, modulo_base, keep_remainder, False
                
            return col, 0, 0, False

        categorical_cols = [
            col for col in self.stats.get_columns_by_type(ColumnType.CATEGORICAL)
            if self._should_process_column(col)
        ]
        
        if not categorical_cols:
            return df
            
        max_workers = self._calculate_thread_count(df)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_column, col) for col in categorical_cols]
            results = [future.result() for future in futures]
        
        # Separate drops from transformations to minimize DataFrame operations
        cols_to_drop = [col for col, _, _, should_drop in results if should_drop]
        if cols_to_drop:
            self.cl.text(f"Dropping columns due to cardinality ratio: {cols_to_drop}")
            df = df.drop(*cols_to_drop)
            self.columns = df.columns
        
        if self.config.cardinality_action == 'group':
            for col, modulo_base, keep_remainder, _ in results:
                if modulo_base > 0:  # means we need to transform this column
                    df = df.withColumn(
                        col,
                        F.when(
                            F.col(col).isNull(),
                            F.lit(self.config.other_label)
                        ).when(
                            (F.abs(F.hash(F.coalesce(F.col(col), F.lit("NULL_VALUE")))) % modulo_base) < keep_remainder,
                            F.col(col)
                        ).otherwise(F.lit(self.config.other_label))
                    )
        
        return df

    def _handle_outliers(self, df: DataFrame) -> DataFrame:
        """Handle outliers using efficient imputation or removal strategies"""
        if self.config.outlier_strategy == OutlierStrategy.NONE:
            self.cl.text("'outlier_strategy' is set to 'none'. Skipping step.")
            return df

        self.cl.text(f"Applying outlier strategy: '{self.config.outlier_strategy.name}'")
        numeric_cols = [col for col in self.stats.get_columns_by_type(ColumnType.NUMERIC)
                    if self._should_process_column(col)]
        if not numeric_cols:
            return df
        
        for col in numeric_cols:
            if col not in self.stats.numeric_stats:
                continue
                
            stats = self.stats.numeric_stats[col]
            if not stats['mean'] or not stats['stddev']:
                continue

            # Calculate bounds based on z-score
            lower_bound = stats['mean'] - self.config.z_score * stats['stddev']
            upper_bound = stats['mean'] + self.config.z_score * stats['stddev']
            
            if self.config.outlier_strategy == OutlierStrategy.DROP:
                # Remove rows where values are beyond z-score threshold
                df = df.filter(
                    (F.col(col) >= lower_bound) & 
                    (F.col(col) <= upper_bound)
                )
                
            elif self.config.outlier_strategy == OutlierStrategy.MEAN:
                # Replace outliers with the mean value
                df = df.withColumn(
                    col,
                    F.when(
                        (F.col(col) < lower_bound) | (F.col(col) > upper_bound),
                        stats['mean']
                    ).otherwise(F.col(col))
                )
                
            elif self.config.outlier_strategy == OutlierStrategy.NEAREST:
                # Replace outliers with the nearest bound value
                # This is more robust than mean imputation as it preserves the direction of the outlier
                df = df.withColumn(
                    col,
                    F.when(F.col(col) < lower_bound, lower_bound)
                    .when(F.col(col) > upper_bound, upper_bound)
                    .otherwise(F.col(col))
                )
        
        return df

    def _handle_multicollinearity(self, df: DataFrame) -> DataFrame:
        """Handle highly correlated features based on configuration."""
        if (self.config.correlation_method == CorrelationMethod.NONE):
            self.cl.text("Skipping multicollinearity handling (correlation_method: {self.config.correlation_method.value})")
            return df
        rho_name = f"ρ_{self.config.correlation_method.value.lower()}"
        numeric_cols = [
            col for col in self.stats.get_columns_by_type(ColumnType.NUMERIC)
            if self._should_process_column(col)
        ]
        if not numeric_cols or len(numeric_cols) < 2:
            return df

        # Get correlation matrix
        correlation_result = self.stats.correlation_matrix
        if correlation_result is None:
            self.calculate_correlation_matrix(df)
            correlation_matrix = self.stats.correlation_matrix
        else:
            correlation_matrix = correlation_result

        if correlation_matrix is None or len(correlation_matrix) == 0:
            return df
            
        if self.config.correlation_action == "drop":
            # Create a dictionary of columns and their correlations
            correlations = {}
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = abs(correlation_matrix[i][j])
                    if corr > self.config.correlation_threshold:
                        pair = (numeric_cols[i], numeric_cols[j])
                        correlations[pair] = corr

            # Sort pairs by correlation strength (highest first)
            sorted_pairs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Keep track of columns to drop
            cols_to_drop = set()
            
            # Process pairs in order of correlation strength
            for (col1, col2), corr in sorted_pairs:
                # Skip if we've already decided to drop one of these
                if col1 in cols_to_drop or col2 in cols_to_drop:
                    continue
                    
                # Get null percentages for both columns
                null_pct1 = self.stats.null_percentages.get(col1, 0)
                null_pct2 = self.stats.null_percentages.get(col2, 0)
                
                # Decision logic:
                # 1. Keep column with fewer nulls
                # 2. If nulls are equal, keep the first column alphabetically
                if null_pct1 > null_pct2:
                    cols_to_drop.add(col1)
                    self.cl.text(f"[↑] {col2} ~ [↓] {col1} ({rho_name}={corr:.3f}. Selection based on null pct)")
                elif null_pct2 > null_pct1:
                    cols_to_drop.add(col2)
                    self.cl.text(f"[↑] {col1} ~ [↓] {col2} ({rho_name}={corr:.3f}. Selection based on null pct)")
                else:
                    # If null percentages are equal, keep the first alphabetically
                    to_drop, to_keep = (col2, col1) if col1 < col2 else (col1, col2)
                    cols_to_drop.add(to_drop)
                    self.cl.text(f"[↑] {to_keep} ~ [↓] {to_drop} ({rho_name}={corr:.3f}. Alphabetical selection)")
                    
            
            if cols_to_drop:
                self.cl.warning(f"🗑️ Dropped {len(cols_to_drop)} correlated columns: {', '.join(sorted(cols_to_drop))}")
                df_corr = df.drop(*cols_to_drop)
                self.columns = df_corr.columns
                return df_corr
        
        return df

    def _filter_columns(self, df: DataFrame) -> DataFrame:
        """Filter columns based on configuration. White list takes preference over black list."""
        
        # Initialize lists if not present
        white_list = getattr(self.config, 'white_list', None)
        black_list = getattr(self.config, 'black_list', None)
        
        # If neither list is specified, return original dataframe
        if not white_list and not black_list:
            return df
        
        all_columns = set(df.columns)
        filtered_columns = all_columns
        msg_parts = [""]
        
        if white_list:
            # White list: only keep specified columns that exist
            filtered_columns = set(white_list) & all_columns
            msg_parts.append(f"🗑️ Applying White-List filtering.\nDropping all columns except: {set(white_list)}")
            
            if black_list:
                self.cl.text("Both 'white_list' and 'black_list' specified. Only 'white_list' will be used.")

        elif black_list:
            # Black list: remove specified columns
            filtered_columns = all_columns - set(black_list)
            dropped_columns = all_columns & set(black_list)
            msg_parts.append(f"🗑️ Applying Black-List filtering.\nDropping columns: {dropped_columns}")

        else:
            # No filter: keep all columns
            self.cl.text("No filters have been applied")
            return df
        
        # Show warning with filtering details
        self.cl.warning(" ".join(msg_parts))
        
        # Return filtered DataFrame
        df_filtered = df.select(*[col for col in filtered_columns if col in df.columns])
        self.columns = df_filtered.columns
        return df_filtered

    def plot_distribution(self, df: DataFrame, column: str, plot_type: str = 'both', num_bins: int = 30):
        """Generate distribution plots"""
        if column not in self.stats.column_types:
            self.cl.warning(f"🔍 Column '{column}' not found")
            return
        
        col_type = self.stats.column_types[column]

        if col_type == ColumnType.NUMERIC and column in self.stats.numeric_stats:
            stats = self.stats.numeric_stats[column]
            
            if plot_type in ['box', 'both']:
                fig1 = go.Figure()
                fig1.add_trace(go.Box(
                    quartilemethod="linear",
                    q1=[float(stats['quartiles'][0])],
                    median=[float(stats['quartiles'][1])],
                    q3=[float(stats['quartiles'][2])],
                    lowerfence=[float(stats['min'])],
                    upperfence=[float(stats['max'])],
                    name=column
                ))
                fig1.update_layout(title=f'Box Plot of {column}', showlegend=False)
                fig1.show()
            
            if plot_type in ['hist', 'both']:
                # Calculate histogram bins based on min and max values
                min_val = float(stats['min'])
                max_val = float(stats['max'])
                bin_width = (max_val - min_val) / num_bins
                
                # Create histogram using Spark SQL window functions
                histogram_df = (
                    df.select(column)
                    .filter(F.col(column).isNotNull())
                    .withColumn(
                        'bin',
                        F.floor((F.col(column).cast('double') - min_val) / bin_width)
                    )
                    .groupBy('bin')
                    .count()
                    .orderBy('bin')
                )
                
                # Collect results (this is small now - just bin counts)
                hist_data = histogram_df.collect()
                
                # Prepare data for plotting
                bins = [min_val + (bin_width * i) for i in range(num_bins + 1)]
                counts = [0] * num_bins
                
                for row in hist_data:
                    bin_idx = int(row['bin'])
                    if 0 <= bin_idx < num_bins:  # Ensure we're within bounds
                        counts[bin_idx] = row['count']
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=bins[:-1],  # Use left edges of bins
                    y=counts,
                    width=bin_width * 0.9,  # Slight gap between bars
                    name=column
                ))
                
                fig2.update_layout(
                    title=f'Distribution of {column}',
                    xaxis_title=column,
                    yaxis_title='Count',
                    bargap=0  # Remove gap between bars for histogram look
                )
                fig2.show()
                
        elif col_type in [ColumnType.CATEGORICAL, ColumnType.BINARY]:
            if column in self.stats.categorical_stats:
                stats = self.stats.categorical_stats[column]
                top_values = stats['top_values'][:20]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[v['value'] for v in top_values],
                    y=[v['count'] for v in top_values]
                ))
                fig.update_layout(
                    title=f'Value Counts for {column}',
                    xaxis_title=column,
                    yaxis_title='Count',
                    xaxis={'tickangle': 45}
                )
                fig.show()

    def plot_correlation_matrix(self, correlation_matrix: np.ndarray = None):
        """Generate correlation matrix heatmap"""
        if self.stats.correlation_matrix is None and correlation_matrix is None:
            self.cl.warning("No correlation matrix available in cache")
            return
        
        numeric_cols = self.stats.get_columns_by_type(ColumnType.NUMERIC)
        if len(numeric_cols) < 2:
            self.cl.warning("Need at least 2 numeric columns for correlation matrix")
            return
        
        if correlation_matrix is not None:
            cm = correlation_matrix
        else:
            cm = self.stats.correlation_matrix
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=numeric_cols,
            y=numeric_cols,
            zmin=-1,
            zmax=1,
            colorscale='RdBu'
        ))
        
        method_name = self.stats.correlation_method.capitalize()
        fig.update_layout(
            title=f'{method_name} Correlation Matrix',
            xaxis_title='Features',
            yaxis_title='Features',
            width=max(600, len(numeric_cols) * 50),
            height=max(600, len(numeric_cols) * 50)
        )
        fig.show()
   
    def plot_missing_values(self):
        """Generate missing values visualization using cached statistics"""
        null_data = [
            (col, pct) for col, pct in self.stats.null_percentages.items() 
            if pct > 0
        ]
        
        if not null_data:
            self.cl.info("No missing values found in the dataset")
            return
        
        null_data.sort(key=lambda x: x[1], reverse=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[x[0] for x in null_data],
            y=[x[1] for x in null_data],
            text=[f"{x[1]:.1f}%" for x in null_data],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Missing Values Percentage by Column',
            xaxis_title='Columns',
            yaxis_title='Null Percentage (%)',
            xaxis={'tickangle': 45},
            width=max(600, len(null_data) * 50),
            showlegend=False
        )
        fig.show()

    def save_config(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {k: str(v) if isinstance(v, Enum) else v 
                      for k, v in self.config.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, path: str):
        """Load configuration from JSON file"""
        self.config = PreprocessingConfig.from_json(path)

    def generate_summary(self) -> Dict:
        """Generate summary using cached statistics"""
        numeric_cols = self.stats.get_columns_by_type(ColumnType.NUMERIC)
        categorical_cols = self.stats.get_columns_by_type(ColumnType.CATEGORICAL)
        text_cols = self.stats.get_columns_by_type(ColumnType.TEXT)
        datetime_cols = self.stats.get_columns_by_type(ColumnType.DATETIME)
        binary_cols = self.stats.get_columns_by_type(ColumnType.BINARY)
        
        summary = {
            'basic_info': {
                'num_rows': self.stats.total_rows,
                'num_columns': self.stats.num_columns,
                'column_types': {col: type_.value for col, type_ 
                               in self.stats.column_types.items()}
            },
            'type_distribution': {
                'numeric': len(numeric_cols),
                'categorical': len(categorical_cols),
                'text': len(text_cols),
                'datetime': len(datetime_cols),
                'binary': len(binary_cols)
            },
            'null_analysis': self.stats.null_percentages
        }

        if self.stats.numeric_stats:
            summary['numeric_stats'] = self.stats.numeric_stats
        if self.stats.categorical_stats:
            summary['categorical_stats'] = self.stats.categorical_stats
        if self.stats.text_stats:
            summary['text_stats'] = self.stats.text_stats
        if self.stats.datetime_stats:
            summary['datetime_stats'] = self.stats.datetime_stats

        if self.stats.correlation_matrix is not None:
            correlations = {}
            for i, col1 in enumerate(numeric_cols):
                significant_correlations = {
                    col2: float(self.stats.correlation_matrix[i][j])
                    for j, col2 in enumerate(numeric_cols)
                    if i != j and abs(self.stats.correlation_matrix[i][j]) > 0.5
                }
                if significant_correlations:
                    correlations[col1] = significant_correlations
            
            if correlations:
                summary['correlations'] = correlations

        return summary

    def to_json(self, summary_dict: Dict, filepath: str):
        """Save summary to a formatted JSON file"""
        with open(filepath, 'w') as f:
            json.dump(summary_dict, f, indent=2)

    def generate_profile_report(self, df: DataFrame, output_path: str = None) -> str:
        """Generate profile report using cached statistics"""
        html_content = []
        
        # Add HTML header with styles
        html_content.append("""
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
                .table { border-collapse: collapse; width: 100%; }
                .table th, .table td { 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left; 
                }
                .table th { background-color: #f5f5f5; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
        """)

        # Dataset Overview using cached metrics
        html_content.append(f"""
        <div class="section">
            <h2>Dataset Overview</h2>
            <p>Number of Rows: {self.stats.total_rows:,}</p>
            <p>Number of Columns: {self.stats.num_columns}</p>
        </div>
        """)

        # Column Type Distribution using cached column info
        type_counts = {
            col_type.value: len(self.stats.get_columns_by_type(col_type))
            for col_type in [
                ColumnType.NUMERIC, 
                ColumnType.CATEGORICAL,
                ColumnType.TEXT,
                ColumnType.DATETIME,
                ColumnType.BINARY
            ]
        }

        html_content.append("""
        <div class="section">
            <h2>Column Type Distribution</h2>
            <table class="table">
                <tr><th>Type</th><th>Count</th></tr>
        """)
        
        for col_type, count in type_counts.items():
            html_content.append(f"<tr><td>{col_type}</td><td>{count}</td></tr>")
        
        html_content.append("</table></div>")

        # Missing Values using cached null percentages
        if self.stats.null_percentages:
            html_content.append("""
            <div class="section">
                <h2>Missing Values</h2>
                <table class="table">
                    <tr><th>Column</th><th>Missing Percentage</th></tr>
            """)
            
            for col, null_pct in sorted(
                self.stats.null_percentages.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                if null_pct > 0:
                    html_content.append(
                        f"<tr><td>{col}</td><td>{null_pct:.2f}%</td></tr>"
                    )
            
            html_content.append("</table></div>")

        # Numeric Statistics using cached numeric stats
        if self.stats.numeric_stats:
            html_content.append("""
            <div class="section">
                <h2>Numeric Column Statistics</h2>
                <table class="table">
                    <tr>
                        <th>Column</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>25%</th>
                        <th>50%</th>
                        <th>75%</th>
                        <th>Max</th>
                    </tr>
            """)
            
            for col, stats in self.stats.numeric_stats.items():
                quartiles = stats['quartiles'] or [None, None, None]
                html_content.append(f"""
                    <tr>
                        <td>{col}</td>
                        <td>{stats['mean']:.2f if stats['mean'] else 'N/A'}</td>
                        <td>{stats['stddev']:.2f if stats['stddev'] else 'N/A'}</td>
                        <td>{stats['min']:.2f if stats['min'] else 'N/A'}</td>
                        <td>{quartiles[0]:.2f if quartiles[0] else 'N/A'}</td>
                        <td>{quartiles[1]:.2f if quartiles[1] else 'N/A'}</td>
                        <td>{quartiles[2]:.2f if quartiles[2] else 'N/A'}</td>
                        <td>{stats['max']:.2f if stats['max'] else 'N/A'}</td>
                    </tr>
                """)
            
            html_content.append("</table></div>")

        # Text Statistics using cached text stats
        if self.stats.text_stats:
            html_content.append("""
            <div class="section">
                <h2>Text Column Statistics</h2>
                <table class="table">
                    <tr>
                        <th>Column</th>
                        <th>Avg Length</th>
                        <th>Min Length</th>
                        <th>Max Length</th>
                        <th>Unique Count</th>
                        <th>Sample Values</th>
                    </tr>
            """)
            
            for col, stats in self.stats.text_stats.items():
                sample_str = "; ".join(stats['sample_values'][:3])
                html_content.append(f"""
                    <tr>
                        <td>{col}</td>
                        <td>{stats['avg_length']:.1f if stats['avg_length'] else 'N/A'}</td>
                        <td>{stats['min_length'] if stats['min_length'] else 'N/A'}</td>
                        <td>{stats['max_length'] if stats['max_length'] else 'N/A'}</td>
                        <td>{stats['unique_count']}</td>
                        <td>{sample_str}</td>
                    </tr>
                """)
            
            html_content.append("</table></div>")

        # Close HTML
        html_content.append("</body></html>")
        
        # Save or return HTML
        html_report = "\n".join(html_content)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_report)
            self.cl.info(f"Profile report saved to {output_path}")
        
        return html_report

    def get_column_names_by_type(self, column_type: ColumnType) -> List[str]:
        """Get list of column names for a specific type"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        return self.stats.get_columns_by_type(column_type)

    def get_column_type(self, column: str) -> Optional[ColumnType]:
        """Get the type of a specific column from cache"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        return self.stats.column_types.get(column)
    
    def get_overview(self) -> Optional[pd.DataFrame]:
        """Get the overview table"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        return self.stats.tables.get('overview')

    def get_missing_values_stats(self) -> Optional[pd.DataFrame]:
        """Get the missing values analysis table"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        
        return self.stats.tables.get('missing_values')

    def get_numeric_stats(self) -> Optional[pd.DataFrame]:
        """Get the numeric statistics table"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        return self.stats.tables.get('numeric_stats')
    
    def get_text_stats(self) -> Optional[pd.DataFrame]:
        """Get the text statistics table"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        return self.stats.tables.get('text_stats')

    def get_categorical_stats(self) -> Optional[pd.DataFrame]:
        """Get the categorical statistics table"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        return self.stats.tables.get('categorical_stats')

    def get_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Get all available tables as a dictionary"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        return self.stats.tables
    
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get a specific table by name"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        if table_name not in self.stats.tables:
            raise ValueError(f"Table '{table_name}' not found. Available tables: {', '.join(self.stats.tables.keys())}")
        return self.stats.tables.get(table_name)

    def get_basic_stats(self) -> Optional[pd.DataFrame]:
        """Get the basic statistics table"""
        if not self.is_fitted():
            raise ValueError("Data not fitted. Call .fit() first.")
        
        # Create a dictionary with column types
        column_types = {col: preprocessor.get_column_type(col).value for col in preprocessor.stats.column_types}

        
        # Construct and return the dictionary with requested metrics
        return {
            'num_rows': self.stats.total_rows,
            'num_columns': self.stats.num_columns,
            'column_types': column_types
        }

    def clear_cache(self):
        """Clear all cached information"""
        self.stats.clear()
        self._is_fitted = False      
    
    def _should_process_column(self, column: str) -> bool:
        """Check if a column should be processed based on the no_process configuration"""
        if column not in self.columns:
            return False
        if self.config.no_process is None:
            return True
        return column not in self.config.no_process

    def get_correlation_matrix(self) -> Optional[Tuple[np.ndarray, List[str], str]]:
        """Retrieve the cached correlation matrix, column names, and correlation method"""
        if self.stats.correlation_matrix is None:
            self.cl.warning("No correlation matrix available in cache")
            return None
            
        numeric_cols = self.stats.get_columns_by_type(ColumnType.NUMERIC)  # Corregido
        if len(numeric_cols) < 2:
            self.cl.warning("Need at least 2 numeric columns for correlation matrix")
            return None
            
        result = (
            self.stats.correlation_matrix,
            numeric_cols,
            self.stats.correlation_method
        )

        print(f"Debug: Returning tuple with lengths: matrix={len(result[0]) if isinstance(result[0], (list, np.ndarray)) else 'scalar'}, cols={len(result[1])}, method={result[2]}")
        return result
   
    def calculate_correlation_matrix(self, df: DataFrame, correlation_method: Optional[str] = 'default', numeric_cols: Optional[List[str]] = None, ) -> np.ndarray:
        """Calculate correlation matrix based on configured method"""
        
        if correlation_method == 'default':
            correlation_method = self.config.correlation_method
        elif correlation_method == 'none':
            self.config.correlation_method = CorrelationMethod.NONE
        elif correlation_method == 'pearson':
            self.config.correlation_method = CorrelationMethod.PEARSON
        elif correlation_method == 'chatterjee':
            self.config.correlation_method = CorrelationMethod.CHATTERJEE
        else:
            raise ValueError(f"Invalid correlation method: {correlation_method}. Valid options are 'none', 'pearson', and 'chatterjee'.")
        
        correlation_method = self.config.correlation_method
        if correlation_method == CorrelationMethod.NONE:
            # self.cl.text("'correlation_method' is set to 'none'. Skipping correlation matrix calculations")
            return np.array([])
    
        if numeric_cols is None:
            numeric_cols = self.stats.get_columns_by_type(ColumnType.NUMERIC)
        
        if not numeric_cols or len(numeric_cols) < 2:
            return np.array([])
        
        if correlation_method == CorrelationMethod.PEARSON:
            # Calculate using Spark's built-in correlation
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol="features",
                handleInvalid="skip"
            )
            df_vector = assembler.transform(df)
            correlation_matrix = Correlation.corr(df_vector, "features").collect()[0][0].toArray()
            
        elif correlation_method == CorrelationMethod.CHATTERJEE:
            correlation_matrix = self._calculate_chatterjee_correlation_matrix(df, numeric_cols)
            
        else:
            raise ValueError(f"Invalid correlation method: {self.config.correlation_method}")
        
        # Cache the result
        self.stats.correlation_matrix = correlation_matrix
        self.stats.correlation_method = self.config.correlation_method.value
        return correlation_matrix

    def _calculate_chatterjee_correlation_matrix(self, df: DataFrame, numeric_cols: List[str]) -> np.ndarray:
        """Calculate correlation matrix using Chatterjee's correlation for numeric columns"""
        n = len(numeric_cols)
        correlation_matrix = np.zeros((n, n))
        
        # Optimize by caching the DataFrame with only numeric columns
        df_numeric = df.select(numeric_cols).cache()
        
        try:
            for i in range(n):
                correlation_matrix[i, i] = 1.0  # Diagonal is always 1
                
                # Only calculate upper triangle to avoid redundant calculations
                for j in range(i + 1, n):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    
                    # Convert to pandas for efficient calculation
                    df_pair = df_numeric.select(col1, col2).dropna().toPandas()
                    
                    if len(df_pair) > 1:  # Need at least 2 points for correlation
                        correlation = self._calculate_chatterjee_correlation(
                            df_pair[col1].values,
                            df_pair[col2].values
                        )
                        
                        # Fill both upper and lower triangle
                        correlation_matrix[i, j] = correlation
                        correlation_matrix[j, i] = correlation
        
        finally:
            # Ensure DataFrame is uncached
            df_numeric.unpersist()
        
        return correlation_matrix
                                   
    def get_high_correlations(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        if self.stats.correlation_matrix is None:
            self.cl.warning("No correlation matrix available in cache")
            return []
            
        numeric_cols = self.stats.get_columns_by_type(ColumnType.NUMERIC)  # <-- Usar self.stats
        if len(numeric_cols) < 2:
            self.cl.warning("Need at least 2 numeric columns for correlations")
            return []
            
        # Use provided threshold or fall back to config
        threshold = threshold or self.config.correlation_threshold
        
        high_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Only look at upper triangle
                    corr = self.stats.correlation_matrix[i, j]
                    if abs(corr) > threshold:
                        high_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': float(corr)
                        })
        
        return sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True)

    @staticmethod        
    def _calculate_chatterjee_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Chatterjee's correlation coefficient"""
        n = len(x)
        if n < 2:
            return 0.0
        
        # Sort data by x values
        idx = np.argsort(x)
        y_sorted = y[idx]
        
        # Calculate rank of y values
        rank_y = stats.rankdata(y_sorted)
        
        # Calculate differences in adjacent ranks
        rank_diffs = np.abs(np.diff(rank_y))
        
        # Calculate Chatterjee's correlation
        xi = np.sum(rank_diffs) / (n * (n - 1) / 2)  # Normalize by number of pairs
        correlation = 1 - xi
        
        return correlation

    def color_text(self, msg: str, color: Colors = Colors.DEFAULT, printout: bool = True, return_colored: bool = True) -> str:
        colored_text = f"{color}{msg}{Colors.DEFAULT}"
        if printout:
            print(colored_text)

        if return_colored:
            return colored_text
        else:
            return msg

    # def color_text(self, text: str, color: Colors = Colors.DEFAULT, printout: bool = True) -> str:
    #     text = f"{color}{text}{Colors.DEFAULT}"
    #     if printout:
    #         print(text)
    #     return text
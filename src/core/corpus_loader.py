import pandas as pd
from pathlib import Path
from typing import List, Optional, Union


class CorpusLoader:
    """
    Summary of the data source.
    - path: path to the file or resource.
    - require_columns: columns that must exist in the DataFrame (ex: ['lyrics', 'genre']).
    """

    def __init__(self, path: Union[str, Path], required_columns: Optional[List[str]] = None) -> None:
        self.path = Path(path)
        self.required_columns = required_columns or ["lyrics", "genre"]

    def load_data(self) -> pd.DataFrame:

        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        suffix = self.path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(self.path)
        elif suffix in (".xls", ".xlsx"):
            df = pd.read_excel(self.path)
        else:
            df = pd.DataFrame(columns=self.required_columns)

        self.validate_schema(df)

        initial_len = len(df)
        df.dropna(subset=self.required_columns, inplace=True)
        
        print(f"Data loaded. Lines: {initial_len}. Removed: {initial_len - len(df)}")
        
        return df
    
    def validate_schema(self, df: Optional[pd.DataFrame] = None) -> bool:

        if df is None:
            df = self.load_data()

        missing = [col for col in self.required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
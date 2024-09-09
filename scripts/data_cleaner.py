from typing import List, Any
import pandas as pd

class DataCleaner:
    """
    A class used to clean data by finding and filling missing values (NaN) in a pandas DataFrame.

    Attributes
    ----------
    data : pd.DataFrame
        The DataFrame that contains the data to be cleaned.

    Methods
    -------
    find_na()
        Returns the proportion of missing values in each column of the DataFrame.

    fill_na(columns: List[str], method: str = 'mode', values: List[Any] = [])
        Fills missing values in specified columns using the chosen method ('mode' or 'mean'),
        or with specified constant values.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Constructs all the necessary attributes for the DataCleaner object.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be cleaned.
        """
        self.data = data

    def find_na(self):
        """
        Identifies and returns the proportion of missing values in each column of the DataFrame.

        Returns
        -------
        pd.Series
            A Series where the index represents the column names and the values represent 
            the proportion of missing values in each respective column.
        """
        return self.data.isna().mean()

    def fill_na(self, columns: List[str], method: str = 'mode', values: List[Any] = []):
        """
        Fills missing values in the specified columns based on the selected method or constant values.

        Parameters
        ----------
        columns : List[str]
            List of column names where missing values should be filled.
        
        method : str, optional
            The method used to fill missing values. Options are 'mode' (default) or 'mean'. 
            If a constant value should be used instead, this argument should be ignored.
        
        values : List[Any], optional
            A list of constant values to fill missing data in the respective columns. If this is used, 
            the 'method' argument is ignored.

        Returns
        -------
        pd.DataFrame
            A DataFrame with missing values filled according to the method or provided constant values.
        
        Raises
        ------
        Exception
            If the filling operation with 'mode' or 'mean' fails or if there is a mismatch between the 
            number of columns and provided constant values.
        """

        if method == 'mode':
            try:
                # Fill NaN values using the mode for each specified column
                result = self.data[columns].fillna(self.data[columns].mode().iloc[0])
                return result
            except Exception as e:
                print(f"Couldn't fill missing values with mode due to: {e}")
        elif method == 'mean':
            try:
                # Fill NaN values using the mean for each specified column
                result = self.data[columns].fillna(self.data[columns].mean())
                return result
            except Exception as e:
                print(f"Couldn't fill missing values with mean due to: {e}")
        else:
            # If method isn't selected, ensure the user provided specific values for each column
            if len(values):
                try:
                    # Create a dictionary to map columns to constant values for NaN filling
                    value_dictionary = {}
                    for idx, column in enumerate(columns):
                        value_dictionary[column] = values[idx]

                    # Fill NaN values with the provided constants
                    result = self.data.fillna(value=value_dictionary)
                    return result
                except Exception as e:
                    print(f"Couldn't fill missing values with constants due to: {e}")
            else:
                print("Please select a method ('mode' or 'mean') or provide constant values to fill missing data!")

    def drop_na(self, drop_column: bool = False):
        """
        Drops rows or columns with missing values (NaN) from the DataFrame.
    
        Parameters
        ----------
        drop_column : bool, optional
            If True, columns with any missing values will be dropped (default is False, which drops rows).
    
        Returns
        -------
        pd.DataFrame
            A DataFrame with rows or columns containing NaN values dropped, depending on the value of drop_column.
    
        Raises
        ------
        Exception
            If the drop operation fails, an exception is caught and an error message is printed.
        """
        try:
            if drop_column:
                # Drop columns that contain any missing values
                result = self.data.dropna(axis=1)
            else:
                # Drop rows that contain any missing values
                result = self.data.dropna()
            return result
        except Exception as e:
            print(f"Couldn't drop columns because of: {e}")


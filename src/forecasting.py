"""
Forecasting Module for UIDAI Data Hackathon 2026
Time series forecasting and predictive modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ForecastingEngine:
    """Forecasting models for Aadhaar data"""
    
    def __init__(self):
        """Initialize forecasting engine"""
        pass
    
    def prepare_time_series(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str
    ) -> pd.Series:
        """
        Prepare time series data
        
        Args:
            df: Input DataFrame
            date_col: Date column
            value_col: Value column
            
        Returns:
            Time series as pandas Series
        """
        df_agg = df.groupby(date_col)[value_col].sum().reset_index()
        df_agg = df_agg.sort_values(date_col)
        df_agg = df_agg.set_index(date_col)
        
        return df_agg[value_col]
    
    def exponential_smoothing_forecast(
        self,
        series: pd.Series,
        periods: int = 30,
        seasonal_periods: int = 7,
        trend: str = 'add',
        seasonal: str = 'add'
    ) -> Dict:
        """
        Exponential smoothing forecast
        
        Args:
            series: Time series data
            periods: Number of periods to forecast
            seasonal_periods: Length of seasonal cycle
            trend: 'add', 'mul', or None
            seasonal: 'add', 'mul', or None
            
        Returns:
            Dictionary with forecast and model info
        """
        try:
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
            # In-sample predictions for evaluation
            fitted_values = fitted_model.fittedvalues
            
            return {
                'forecast': forecast,
                'fitted_values': fitted_values,
                'model': fitted_model,
                'method': 'Exponential Smoothing'
            }
        except Exception as e:
            print(f"Exponential smoothing failed: {e}")
            return None
    
    def arima_forecast(
        self,
        series: pd.Series,
        periods: int = 30,
        order: Tuple[int, int, int] = (1, 1, 1)
    ) -> Dict:
        """
        ARIMA forecast
        
        Args:
            series: Time series data
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q)
            
        Returns:
            Dictionary with forecast and model info
        """
        try:
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            forecast = fitted_model.forecast(steps=periods)
            fitted_values = fitted_model.fittedvalues
            
            return {
                'forecast': forecast,
                'fitted_values': fitted_values,
                'model': fitted_model,
                'method': 'ARIMA',
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        except Exception as e:
            print(f"ARIMA forecast failed: {e}")
            return None
    
    def simple_moving_average_forecast(
        self,
        series: pd.Series,
        periods: int = 30,
        window: int = 7
    ) -> Dict:
        """
        Simple moving average forecast
        
        Args:
            series: Time series data
            periods: Number of periods to forecast
            window: Moving average window
            
        Returns:
            Dictionary with forecast
        """
        # Calculate moving average
        ma = series.rolling(window=window).mean()
        
        # Forecast using last MA value
        last_ma = ma.iloc[-1]
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        forecast = pd.Series([last_ma] * periods, index=forecast_index)
        
        return {
            'forecast': forecast,
            'fitted_values': ma,
            'method': 'Moving Average'
        }
    
    def naive_forecast(
        self,
        series: pd.Series,
        periods: int = 30
    ) -> Dict:
        """
        Naive forecast (last value propagation)
        
        Args:
            series: Time series data
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast
        """
        last_value = series.iloc[-1]
        
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        forecast = pd.Series([last_value] * periods, index=forecast_index)
        
        return {
            'forecast': forecast,
            'method': 'Naive'
        }
    
    def seasonal_naive_forecast(
        self,
        series: pd.Series,
        periods: int = 30,
        seasonal_period: int = 7
    ) -> Dict:
        """
        Seasonal naive forecast
        
        Args:
            series: Time series data
            periods: Number of periods to forecast
            seasonal_period: Length of seasonal cycle
            
        Returns:
            Dictionary with forecast
        """
        # Get last seasonal cycle
        last_season = series.iloc[-seasonal_period:]
        
        # Repeat seasonal pattern
        n_full_cycles = periods // seasonal_period
        remainder = periods % seasonal_period
        
        forecast_values = list(last_season) * n_full_cycles + list(last_season[:remainder])
        
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        forecast = pd.Series(forecast_values, index=forecast_index)
        
        return {
            'forecast': forecast,
            'method': 'Seasonal Naive'
        }
    
    def evaluate_forecast(
        self,
        actual: pd.Series,
        predicted: pd.Series
    ) -> Dict:
        """
        Evaluate forecast accuracy
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Align series
        common_index = actual.index.intersection(predicted.index)
        actual_aligned = actual.loc[common_index]
        predicted_aligned = predicted.loc[common_index]
        
        mae = mean_absolute_error(actual_aligned, predicted_aligned)
        mse = mean_squared_error(actual_aligned, predicted_aligned)
        rmse = np.sqrt(mse)
        
        # MAPE
        mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
        
        # R-squared
        r2 = r2_score(actual_aligned, predicted_aligned)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def train_test_split_temporal(
        self,
        series: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Split time series into train and test
        
        Args:
            series: Time series data
            test_size: Proportion for test set
            
        Returns:
            Tuple of (train, test) series
        """
        n = len(series)
        split_point = int(n * (1 - test_size))
        
        train = series.iloc[:split_point]
        test = series.iloc[split_point:]
        
        return train, test
    
    def compare_models(
        self,
        series: pd.Series,
        test_size: float = 0.2
    ) -> pd.DataFrame:
        """
        Compare multiple forecasting models
        
        Args:
            series: Time series data
            test_size: Proportion for test set
            
        Returns:
            DataFrame with model comparisons
        """
        train, test = self.train_test_split_temporal(series, test_size)
        
        results = []
        
        # Naive
        naive_result = self.naive_forecast(train, len(test))
        if naive_result:
            metrics = self.evaluate_forecast(test, naive_result['forecast'])
            results.append({
                'model': 'Naive',
                **metrics
            })
        
        # Moving Average
        ma_result = self.simple_moving_average_forecast(train, len(test), window=7)
        if ma_result:
            metrics = self.evaluate_forecast(test, ma_result['forecast'])
            results.append({
                'model': 'Moving Average (7)',
                **metrics
            })
        
        # Seasonal Naive
        seasonal_result = self.seasonal_naive_forecast(train, len(test), seasonal_period=7)
        if seasonal_result:
            metrics = self.evaluate_forecast(test, seasonal_result['forecast'])
            results.append({
                'model': 'Seasonal Naive',
                **metrics
            })
        
        # Exponential Smoothing
        try:
            es_result = self.exponential_smoothing_forecast(
                train, 
                len(test), 
                seasonal_periods=7
            )
            if es_result:
                metrics = self.evaluate_forecast(test, es_result['forecast'])
                results.append({
                    'model': 'Exponential Smoothing',
                    **metrics
                })
        except:
            pass
        
        # ARIMA
        try:
            arima_result = self.arima_forecast(train, len(test), order=(1, 1, 1))
            if arima_result:
                metrics = self.evaluate_forecast(test, arima_result['forecast'])
                results.append({
                    'model': 'ARIMA(1,1,1)',
                    **metrics
                })
        except:
            pass
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('mae')
        
        return comparison_df
    
    def generate_forecast_confidence_interval(
        self,
        forecast: pd.Series,
        std: float,
        confidence: float = 0.95
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate confidence intervals for forecast
        
        Args:
            forecast: Forecast series
            std: Standard deviation of residuals
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound) series
        """
        from scipy.stats import norm
        
        z_score = norm.ppf((1 + confidence) / 2)
        margin = z_score * std
        
        lower_bound = forecast - margin
        upper_bound = forecast + margin
        
        return lower_bound, upper_bound


if __name__ == "__main__":
    print("Forecasting module loaded successfully")

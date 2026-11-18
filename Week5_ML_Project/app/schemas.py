"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastRequest(BaseModel):
    """Features required to predict next-hour Zone 1 consumption.

    Lag and rolling features must be calculated from observations available
    before the hour being forecast.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    hour: int = Field(..., ge=0, le=23, examples=[14], description="Hour of day (0-23)")
    dayofweek: int = Field(..., ge=0, le=6, examples=[2], description="Monday=0 ... Sunday=6")
    month: int = Field(..., ge=1, le=12, examples=[7], description="Month (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, examples=[0], description="1 if weekend else 0")
    temperature: float = Field(..., examples=[25.5], description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, examples=[55.0], description="Humidity (%)")
    windspeed: float = Field(..., ge=0, examples=[1.2], description="Wind speed")
    general_diffuse_flows: float = Field(..., ge=0, examples=[120.0])
    diffuse_flows: float = Field(..., ge=0, examples=[15.0])
    lag_1: float = Field(..., gt=0, examples=[32000.0], description="Consumption 1 hour ago")
    lag_2: float = Field(..., gt=0, examples=[31500.0], description="Consumption 2 hours ago")
    lag_3: float = Field(..., gt=0, examples=[31000.0], description="Consumption 3 hours ago")
    lag_24: float = Field(..., gt=0, examples=[30500.0], description="Consumption 24 hours ago")
    rolling_mean_3: float = Field(..., gt=0, examples=[31800.0])
    rolling_mean_6: float = Field(..., gt=0, examples=[31200.0])
    rolling_mean_24: float = Field(..., gt=0, examples=[30000.0])
    rolling_std_24: float = Field(..., ge=0, examples=[2500.0])

    @model_validator(mode="after")
    def validate_calendar_consistency(self):
        expected_weekend = int(self.dayofweek >= 5)
        if self.is_weekend != expected_weekend:
            raise ValueError(
                "is_weekend must be 1 for Saturday/Sunday (dayofweek 5 or 6), "
                "and 0 for Monday-Friday."
            )
        return self


class ForecastResponse(BaseModel):
    """Predicted next-hour electricity consumption for Zone 1."""

    predicted_consumption: float = Field(
        ..., description="Predicted mean power (kW) for next hour"
    )
    zone: str = Field(default="Zone1")
    unit: str = Field(default="kW (hourly mean power)")
    model: str = Field(default="XGBoost")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

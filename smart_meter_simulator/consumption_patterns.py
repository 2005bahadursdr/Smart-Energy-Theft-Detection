"""
Consumption Patterns Engine.
Simulates realistic diurnals, weekly schedules, annual seasonality,
solar rooftop offset, and stochastic noise for smart grid meters.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
import numpy as np


class ConsumerType(str, Enum):
    """Consumer archetypes mapped across distribution feeder topologies."""
    RESIDENTIAL = "RESIDENTIAL"
    COMMERCIAL = "COMMERCIAL"
    INDUSTRIAL = "INDUSTRIAL"
    RURAL = "RURAL"


# 24-hour diurnal relative load shapes (normalized around 1.0)
DIURNAL_BASELINES = {
    # Residential: Low nighttime, morning ramp (07:00-09:00), midday lull, high evening peak (18:00-22:00)
    ConsumerType.RESIDENTIAL: np.array([
        0.35, 0.28, 0.25, 0.24, 0.26, 0.38, 0.65, 1.10, 1.25, 0.95,
        0.80, 0.75, 0.72, 0.70, 0.72, 0.78, 0.92, 1.30, 1.65, 1.75,
        1.60, 1.35, 0.90, 0.55
    ], dtype=np.float64),

    # Commercial: Standby at night, sharp morning ramp (07:00-08:00), high sustained daytime plateau, evening shutdown
    ConsumerType.COMMERCIAL: np.array([
        0.20, 0.18, 0.18, 0.18, 0.22, 0.35, 0.60, 1.10, 1.55, 1.70,
        1.75, 1.80, 1.75, 1.75, 1.70, 1.65, 1.50, 1.20, 0.75, 0.45,
        0.35, 0.30, 0.25, 0.22
    ], dtype=np.float64),

    # Industrial: High continuous baseload, 3-shift or 2-shift operations, relatively flat with slight shift changes
    ConsumerType.INDUSTRIAL: np.array([
        0.85, 0.85, 0.82, 0.80, 0.85, 1.05, 1.30, 1.45, 1.50, 1.48,
        1.45, 1.40, 1.35, 1.45, 1.48, 1.45, 1.40, 1.35, 1.20, 1.15,
        1.05, 1.00, 0.95, 0.90
    ], dtype=np.float64),

    # Rural / Agricultural: Early morning irrigation pumping (04:00-08:00), afternoon lull, evening surge
    ConsumerType.RURAL: np.array([
        0.40, 0.38, 0.42, 0.65, 1.15, 1.55, 1.60, 1.40, 0.95, 0.80,
        0.75, 0.70, 0.68, 0.70, 0.72, 0.85, 1.05, 1.35, 1.40, 1.25,
        0.95, 0.70, 0.55, 0.45
    ], dtype=np.float64)
}

# Day of week multiplier (Monday=0, ..., Sunday=6)
WEEKDAY_MULTIPLIERS = {
    # Residential: Weekend is higher during daytime (occupants stay home)
    ConsumerType.RESIDENTIAL: np.array([0.95, 0.95, 0.96, 0.97, 1.02, 1.15, 1.12], dtype=np.float64),

    # Commercial: Saturday is half-day or retail heavy, Sunday is low standby
    ConsumerType.COMMERCIAL: np.array([1.10, 1.12, 1.12, 1.10, 1.08, 0.65, 0.35], dtype=np.float64),

    # Industrial: Saturday partial capacity, Sunday skeleton or maintenance
    ConsumerType.INDUSTRIAL: np.array([1.08, 1.10, 1.10, 1.10, 1.05, 0.75, 0.45], dtype=np.float64),

    # Rural: Relatively steady, slight weekend reduction
    ConsumerType.RURAL: np.array([1.02, 1.02, 1.00, 1.00, 1.00, 0.98, 0.95], dtype=np.float64)
}


@dataclass
class ConsumptionPatternEngine:
    """
    Engine combining diurnal curves, weekly shifts, seasonal cycles,
    solar self-generation, and stochastic noise.
    """
    summer_peak_doy: int = 200     # Mid-July
    winter_peak_doy: int = 15      # Mid-January
    seasonal_amplitude: float = 0.25 # Peak-to-trough seasonal variation (±25%)
    noise_std: float = 0.08        # Relative standard deviation of Gaussian noise

    def get_seasonal_factor(self, day_of_year: int) -> float:
        """
        Calculate bimodal seasonal demand factor using physical Heating & Cooling Degree Day model.
        Demand peaks during winter space-heating and summer air-conditioning,
        while dropping to baseline during mild spring and autumn shoulder seasons.
        """
        # Annual sinusoidal temperature curve peaked in mid-July (approx day 196)
        temp_c = 18.0 + 14.0 * math.sin(2 * math.pi * (day_of_year - 105) / 365.25)

        cooling_degrees = max(0.0, temp_c - 21.0)
        heating_degrees = max(0.0, 15.0 - temp_c)

        # Composite weather sensitivity
        weather_load = (cooling_degrees * 1.1 + heating_degrees * 1.3) / 14.0
        seasonal_factor = 1.0 + self.seasonal_amplitude * (weather_load - 0.40)
        return max(0.65, round(seasonal_factor, 4))

    def get_solar_generation(
        self,
        dt: datetime,
        solar_capacity_kw: float,
        rng: np.random.RandomState
    ) -> float:
        """
        Simulate rooftop photovoltaic generation curve during daylight hours.
        Peaked around 13:00, zero before sunrise and after sunset, modulated by cloud noise.
        """
        if solar_capacity_kw <= 0.0:
            return 0.0

        hour = dt.hour + dt.minute / 60.0
        # Daylight window: approx 06:00 to 19:00
        if hour < 6.0 or hour > 19.0:
            return 0.0

        # Bell-shaped solar irradiance curve peaked at 12.5 (12:30 PM)
        width = 3.2
        sun_profile = math.exp(-0.5 * ((hour - 12.5) / width) ** 2)

        # Seasonal solar adjustment (higher irradiance in summer)
        day_of_year = dt.timetuple().tm_yday
        seasonal_sun = 0.8 + 0.4 * math.sin(2 * math.pi * (day_of_year - 80) / 365.25)

        # Cloud cover variability (random attenuation between 0.4 and 1.0)
        cloud_factor = rng.uniform(0.65, 1.0)

        generation = solar_capacity_kw * sun_profile * seasonal_sun * cloud_factor
        return max(0.0, float(generation))

    def compute_hourly_consumption(
        self,
        dt: datetime,
        consumer_type: ConsumerType,
        base_load_kwh: float,
        rng: np.random.RandomState,
        solar_capacity_kw: float = 0.0
    ) -> float:
        """
        Compute legitimate, un-tampered hourly consumption in kWh.
        """
        hour = dt.hour
        day_of_week = dt.weekday()
        day_of_year = dt.timetuple().tm_yday

        # 1. Base Diurnal Factor
        diurnal_curve = DIURNAL_BASELINES[consumer_type]
        diurnal_factor = diurnal_curve[hour]

        # 2. Weekend / Day of Week Multiplier
        dow_multipliers = WEEKDAY_MULTIPLIERS[consumer_type]
        dow_factor = dow_multipliers[day_of_week]

        # Weekend residential profile adjustment (occupants wake up later, stay active afternoon)
        if consumer_type == ConsumerType.RESIDENTIAL and day_of_week >= 5:
            if 10 <= hour <= 16:
                diurnal_factor *= 1.35
            elif 6 <= hour <= 8:
                diurnal_factor *= 0.70

        # 3. Seasonal Factor
        seasonal_factor = self.get_seasonal_factor(day_of_year)

        # 4. Multiplicative Gaussian Noise + lognormal spike
        gaussian_noise = rng.normal(loc=0.0, scale=self.noise_std)
        # Occasional appliance spike (e.g. EV charger, pump, compressor)
        spike = rng.lognormal(mean=-4.0, sigma=0.6) if rng.rand() < 0.06 else 0.0

        gross_multiplier = max(0.05, (diurnal_factor * dow_factor * seasonal_factor * (1.0 + gaussian_noise)) + spike)
        gross_consumption = base_load_kwh * gross_multiplier

        # 5. Solar Self-Generation Offset
        if solar_capacity_kw > 0.0:
            solar_gen = self.get_solar_generation(dt, solar_capacity_kw, rng)
            net_consumption = max(0.0, gross_consumption - solar_gen)
            return round(net_consumption, 4)

        return round(max(0.0, gross_consumption), 4)

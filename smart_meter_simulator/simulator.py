"""
Smart Meter Fleet Simulator.
Generates high-fidelity synthetic hourly consumption telemetry for 500-2,000 pseudonymous
smart meters partitioned across regional distribution substations (client_A - client_E).
Simulates diurnal curves, weekly dynamics, seasonal shifts, stochastic noise,
legitimate zero consumption (vacancies, outages, disconnections, solar self-generation),
communication drops, sustained efficiency drops, and synthetic tamper events.
Exports reproducible CSVs and maintains strict separation between telemetry and simulated ground truth.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from smart_meter_simulator.consumer import (
    ConsumerProfile,
    ConsumerType,
    OccupancyState,
    generate_consumer_profile,
)
from smart_meter_simulator.consumption_patterns import ConsumptionPatternEngine
from smart_meter_simulator.meter import GridEventType, MeterStatus, SmartMeter
from smart_meter_simulator.tamper_events import TamperEvent, TamperType

logger = logging.getLogger("smart_meter_simulator")


# Distribution substation consumer mix definitions (matching grid topology)
SUBSTATION_MIXES = {
    "client_A": [ # Industrial Zone Substation
        (ConsumerType.INDUSTRIAL, 0.70),
        (ConsumerType.COMMERCIAL, 0.20),
        (ConsumerType.RESIDENTIAL, 0.10),
    ],
    "client_B": [ # Commercial District Substation
        (ConsumerType.COMMERCIAL, 0.60),
        (ConsumerType.RESIDENTIAL, 0.30),
        (ConsumerType.INDUSTRIAL, 0.10),
    ],
    "client_C": [ # Urban Residential Feeder 1
        (ConsumerType.RESIDENTIAL, 0.85),
        (ConsumerType.COMMERCIAL, 0.15),
    ],
    "client_D": [ # Urban Residential Feeder 2
        (ConsumerType.RESIDENTIAL, 0.90),
        (ConsumerType.COMMERCIAL, 0.10),
    ],
    "client_E": [ # Rural / Semi-Urban Feeder
        (ConsumerType.RURAL, 0.50),
        (ConsumerType.RESIDENTIAL, 0.40),
        (ConsumerType.COMMERCIAL, 0.10),
    ]
}


class SmartMeterSimulator:
    """
    Fleet-wide smart meter orchestrator.
    """

    def __init__(
        self,
        num_meters: int = 1000,
        days: int = 30,
        start_date: str = "2026-01-01 00:00:00",
        theft_rate: float = 0.08,
        vacancy_rate: float = 0.04,
        disconnection_rate: float = 0.02,
        outage_probability: float = 0.003,
        missing_reading_rate: float = 0.01,
        solar_adoption_rate: float = 0.12,
        sustained_drop_rate: float = 0.05,
        client_ids: Optional[List[str]] = None,
        seed: int = 42,
    ):
        """
        Initialize simulation configuration.

        Parameters
        ----------
        num_meters : int
            Total smart meters to instantiate (recommended 500-2,000).
        days : int
            Simulation duration in calendar days (e.g. 30 days = 720 hours).
        start_date : str
            Simulation timeline anchor in ISO format ("YYYY-MM-DD HH:MM:SS").
        theft_rate : float
            Fraction of meters subjected to synthetic tampering.
        vacancy_rate : float
            Fraction of residential/commercial meters representing vacant properties.
        disconnection_rate : float
            Fraction of meters disconnected by utility.
        outage_probability : float
            Hourly probability of feeder blackout occurrence.
        missing_reading_rate : float
            Hourly probability of communication packet drop per meter.
        solar_adoption_rate : float
            Fraction of meters with rooftop PV solar generation.
        sustained_drop_rate : float
            Fraction of meters with legitimate energy efficiency/sustained load drop.
        client_ids : Optional[List[str]]
            Substation identifier list (defaults to client_A through client_E).
        seed : int
            Random seed for deterministic reproducibility.
        """
        self.num_meters = num_meters
        self.days = days
        self.total_hours = days * 24
        self.start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        self.theft_rate = theft_rate
        self.vacancy_rate = vacancy_rate
        self.disconnection_rate = disconnection_rate
        self.outage_probability = outage_probability
        self.missing_reading_rate = missing_reading_rate
        self.solar_adoption_rate = solar_adoption_rate
        self.sustained_drop_rate = sustained_drop_rate
        self.client_ids = client_ids or ["client_A", "client_B", "client_C", "client_D", "client_E"]
        self.seed = seed

        self.rng = np.random.RandomState(seed)
        self.pattern_engine = ConsumptionPatternEngine()
        self.meters: List[SmartMeter] = []
        self.feeder_outage_schedules: Dict[str, List[Tuple[int, int]]] = {
            cid: [] for cid in self.client_ids
        }

    def _sample_consumer_type(self, client_id: str) -> ConsumerType:
        """Sample consumer type according to substation regional archetype mix."""
        mix = SUBSTATION_MIXES.get(client_id, SUBSTATION_MIXES["client_C"])
        types = [item[0] for item in mix]
        probs = [item[1] for item in mix]
        probs = np.array(probs) / np.sum(probs)
        idx = int(self.rng.choice(len(types), p=probs))
        return types[idx]

    def _generate_tamper_event(self, start_hour: int) -> TamperEvent:
        """Create a diverse synthetic tamper event."""
        # Realistic distribution across tamper modalities
        modalities = [
            TamperType.PARTIAL_BYPASS,
            TamperType.FULL_BYPASS,
            TamperType.REVERSE_CURRENT,
            TamperType.MAGNETIC_TAMPER,
            TamperType.CYBER_SCALING,
            TamperType.CYBER_FLATLINE,
            TamperType.PEAK_CLIPPING,
            TamperType.RANDOM_UNDER_REPORTING
        ]
        weights = [0.22, 0.12, 0.08, 0.15, 0.18, 0.08, 0.09, 0.08]
        weights = np.array(weights) / sum(weights)
        idx = int(self.rng.choice(len(modalities), p=weights))
        t_type = modalities[idx]

        # Duration of theft event: between 72h (or available horizon) and end of simulation
        max_duration = max(2, self.total_hours - start_hour)
        min_duration = min(72, max_duration)
        duration = int(self.rng.randint(min_duration, max_duration + 1))
        end_hour = min(self.total_hours - 1, start_hour + duration)

        severity = float(self.rng.uniform(0.35, 0.75))
        hw_alert_prob = 0.20
        if t_type == TamperType.MAGNETIC_TAMPER:
            hw_alert_prob = 0.85
        elif t_type == TamperType.REVERSE_CURRENT:
            hw_alert_prob = 0.75
        elif t_type in (TamperType.CYBER_SCALING, TamperType.CYBER_FLATLINE):
            hw_alert_prob = 0.05

        return TamperEvent(
            tamper_type=t_type,
            start_hour=start_hour,
            end_hour=end_hour,
            severity=severity,
            hardware_alert_prob=hw_alert_prob,
            flatline_value_kwh=float(self.rng.uniform(0.08, 0.25)),
            peak_cap_kwh=float(self.rng.uniform(0.80, 1.40))
        )

    def _schedule_feeder_outages(self) -> None:
        """
        Schedule realistic grid blackouts at feeder / substation level.
        Each blackout lasts 2 to 6 hours and affects all meters on that substation.
        """
        for cid in self.client_ids:
            # Expected number of outages across simulation period
            num_outages = self.rng.poisson(lam=max(1, int(self.total_hours * self.outage_probability)))
            for _ in range(num_outages):
                outage_duration = int(self.rng.randint(2, min(7, max(3, self.total_hours))))
                max_start = max(1, self.total_hours - outage_duration)
                outage_start = int(self.rng.randint(0, max_start))
                outage_end = min(self.total_hours - 1, outage_start + outage_duration)
                self.feeder_outage_schedules[cid].append((outage_start, outage_end))

    def initialize_fleet(self) -> None:
        """
        Instantiate pseudonymous smart meters and distribute across feeders.
        """
        self.meters = []
        self._schedule_feeder_outages()

        meters_per_client = self.num_meters // len(self.client_ids)
        remainder = self.num_meters % len(self.client_ids)

        meter_counter = 1
        for c_idx, client_id in enumerate(self.client_ids):
            count = meters_per_client + (1 if c_idx < remainder else 0)

            for _ in range(count):
                # Pseudonymous meter identifier with substation tag and hex hash
                meter_id = f"MTR-{client_id.split('_')[-1]}-{meter_counter:04d}"
                consumer_type = self._sample_consumer_type(client_id)

                # Probabilistic behavioral traits
                is_vacant = (self.rng.rand() < self.vacancy_rate) and (consumer_type in (ConsumerType.RESIDENTIAL, ConsumerType.COMMERCIAL))
                has_sustained_drop = (not is_vacant) and (self.rng.rand() < self.sustained_drop_rate)
                has_solar = (self.rng.rand() < self.solar_adoption_rate) and (consumer_type != ConsumerType.INDUSTRIAL)

                consumer_profile = generate_consumer_profile(
                    consumer_id=f"CONS-{meter_id}",
                    consumer_type=consumer_type,
                    rng=self.rng,
                    is_vacant=is_vacant,
                    has_sustained_drop=has_sustained_drop,
                    has_solar=has_solar,
                    total_hours=self.total_hours
                )

                # Schedule synthetic tamper event (if selected for theft evaluation)
                tamper_event = None
                if self.rng.rand() < self.theft_rate:
                    # Theft starts between 15% and 50% into simulation timeline
                    theft_start = int(self.rng.randint(self.total_hours // 6, self.total_hours // 2))
                    tamper_event = self._generate_tamper_event(theft_start)

                # Schedule utility disconnection
                is_disc = False
                disc_start = None
                disc_end = None
                if (not is_vacant) and (tamper_event is None) and (self.rng.rand() < self.disconnection_rate):
                    disc_start = int(self.rng.randint(self.total_hours // 3, self.total_hours * 2 // 3))
                    disc_dur = int(self.rng.randint(48, 168)) # 2 to 7 days
                    disc_end = min(self.total_hours - 1, disc_start + disc_dur)
                    is_disc = True

                meter = SmartMeter(
                    meter_id=meter_id,
                    client_id=client_id,
                    consumer=consumer_profile,
                    tamper_event=tamper_event,
                    is_disconnected=False,
                    disconnect_start_hour=disc_start if is_disc else None,
                    disconnect_end_hour=disc_end if is_disc else None,
                )
                self.meters.append(meter)
                meter_counter += 1

        logger.info(
            f"Initialized {len(self.meters)} pseudonymous meters across {len(self.client_ids)} substations."
        )

    def _is_feeder_in_outage(self, client_id: str, hour_idx: int) -> bool:
        """Check if substation feeder is currently experiencing a grid outage."""
        for start_h, end_h in self.feeder_outage_schedules.get(client_id, []):
            if start_h <= hour_idx <= end_h:
                return True
        return False

    def simulate(self) -> pd.DataFrame:
        """
        Execute fleet simulation over all hourly timestamps.

        Returns
        -------
        pd.DataFrame
            Complete hourly readings dataset with separated telemetry and simulated labels.
        """
        if not self.meters:
            self.initialize_fleet()

        records: List[Dict[str, Any]] = []
        timestamps = [self.start_dt + timedelta(hours=h) for h in range(self.total_hours)]

        logger.info(
            f"Executing simulation: {len(self.meters)} meters x {self.total_hours} hours = "
            f"{len(self.meters) * self.total_hours:,} readings..."
        )

        for hour_idx, dt in enumerate(timestamps):
            # Check outage status for each client at this hour
            outage_lookup = {
                cid: self._is_feeder_in_outage(cid, hour_idx)
                for cid in self.client_ids
            }

            for meter in self.meters:
                is_outage = outage_lookup[meter.client_id]
                # Communication drop is intermittent per meter reading
                is_comm_drop = (not is_outage) and (self.rng.rand() < self.missing_reading_rate)

                rec = meter.read_hour(
                    dt=dt,
                    hour_idx=hour_idx,
                    is_grid_outage=is_outage,
                    is_comm_drop=is_comm_drop,
                    pattern_engine=self.pattern_engine,
                    rng=self.rng
                )
                records.append(rec)

        df = pd.DataFrame(records)
        logger.info(f"Simulation completed. Total records synthesized: {len(df):,}")
        return df

    def get_metadata_catalog(self) -> pd.DataFrame:
        """
        Extract meter metadata catalog for regional substation partitioning.
        """
        metadata = []
        for m in self.meters:
            c = m.consumer
            metadata.append({
                "meter_id": m.meter_id,
                "client_id": m.client_id,
                "consumer_type": c.consumer_type.value,
                "base_load_kwh": c.base_load_kwh,
                "occupancy_state": c.occupancy_state.value,
                "has_solar": c.has_solar,
                "solar_capacity_kw": c.solar_capacity_kw,
                "has_sustained_drop": c.sustained_drop_ratio > 0.0,
                "has_disconnection": m.disconnect_start_hour is not None,
                "simulated_has_tamper": m.tamper_event is not None,
                "simulated_tamper_type": m.tamper_event.tamper_type.value if m.tamper_event else TamperType.NONE.value,
                "simulated_tamper_start_hour": m.tamper_event.start_hour if m.tamper_event else None,
                "simulated_tamper_end_hour": m.tamper_event.end_hour if m.tamper_event else None,
            })
        return pd.DataFrame(metadata)

    def export_dataset(
        self,
        output_dir: str = "data/synthetic",
        filename: str = "simulated_meter_data.csv",
        metadata_filename: str = "simulated_meter_metadata.csv",
        summary_filename: str = "simulation_summary.json"
    ) -> Tuple[Path, Path, Path, Dict[str, Any]]:
        """
        Export reproducible CSVs and summary statistics to disk.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        readings_df = self.simulate()
        metadata_df = self.get_metadata_catalog()

        csv_file = out_path / filename
        meta_file = out_path / metadata_filename
        sum_file = out_path / summary_filename

        readings_df.to_csv(csv_file, index=False)
        metadata_df.to_csv(meta_file, index=False)

        # Summary statistics calculation
        total_readings = len(readings_df)
        zero_consumption_mask = (readings_df["consumption_kwh"] == 0.0)
        zero_readings_count = int(zero_consumption_mask.sum())
        
        # Legitimate zeros: zero consumption with simulated_theft_label == 0
        legit_zero_mask = zero_consumption_mask & (readings_df["simulated_theft_label"] == 0)
        legit_zeros_count = int(legit_zero_mask.sum())

        # Theft zeros: zero or near-zero caused by full bypass / reverse current with simulated_theft_label == 1
        theft_zero_mask = zero_consumption_mask & (readings_df["simulated_theft_label"] == 1)
        theft_zeros_count = int(theft_zero_mask.sum())

        total_simulated_theft_readings = int((readings_df["simulated_theft_label"] == 1).sum())
        total_tamper_alerts = int((readings_df["hardware_tamper_flag"] == 1).sum())
        missing_readings_count = int(readings_df["consumption_kwh"].isna().sum())

        summary = {
            "simulation_metadata": {
                "project": "Blockchain-Based Byzantine-Resilient FL Smart Grid Energy Theft Detection",
                "engine": "SmartMeterSimulator v1.0",
                "random_seed": self.seed,
                "start_timestamp": self.start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "total_meters": len(self.meters),
                "total_days": self.days,
                "total_hours": self.total_hours,
                "total_records": total_readings,
                "substations": self.client_ids,
            },
            "consumption_statistics": {
                "mean_reported_kwh": float(readings_df["consumption_kwh"].dropna().mean()),
                "max_reported_kwh": float(readings_df["consumption_kwh"].dropna().max()),
                "zero_readings_total": zero_readings_count,
                "zero_readings_legitimate": legit_zeros_count,
                "zero_readings_simulated_theft": theft_zeros_count,
                "legitimate_zero_percentage_of_all_zeros": round(
                    (legit_zeros_count / max(1, zero_readings_count)) * 100, 2
                ),
                "missing_readings_telemetry_drops": missing_readings_count,
            },
            "event_breakdown": {
                k: int(v) for k, v in readings_df["event_type"].value_counts().to_dict().items()
            },
            "meter_status_breakdown": {
                k: int(v) for k, v in readings_df["meter_status"].value_counts().to_dict().items()
            },
            "synthetic_ground_truth": {
                "theft_rate_meter_level": round(
                    float(metadata_df["simulated_has_tamper"].mean()), 4
                ),
                "simulated_theft_reading_count": total_simulated_theft_readings,
                "simulated_theft_reading_percentage": round(
                    (total_simulated_theft_readings / total_readings) * 100, 3
                ),
                "hardware_tamper_alerts_raised": total_tamper_alerts,
                "total_simulated_unreported_kwh": round(
                    float(readings_df["simulated_unreported_kwh"].sum()), 2
                ),
                "tamper_modality_distribution": {
                    k: int(v)
                    for k, v in readings_df["simulated_theft_type"].value_counts().to_dict().items()
                }
            }
        }

        with open(sum_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset exported to: {csv_file}")
        logger.info(f"Metadata exported to: {meta_file}")
        logger.info(f"Summary exported to: {sum_file}")

        return csv_file, meta_file, sum_file, summary


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser for smart meter simulator."""
    parser = argparse.ArgumentParser(
        prog="smart_meter_simulator",
        description="High-Fidelity Synthetic Smart Meter Telemetry Simulator with Separated Ground Truth"
    )
    parser.add_argument(
        "--num-meters",
        type=int,
        default=1000,
        help="Number of pseudonymous smart meters to simulate (500-2,000 recommended, default: 1000)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Duration of simulation in calendar days (default: 30 days = 720 hours)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-01-01 00:00:00",
        help="Simulation start datetime (default: '2026-01-01 00:00:00')"
    )
    parser.add_argument(
        "--theft-rate",
        type=float,
        default=0.08,
        help="Proportion of meters subjected to synthetic tampering (default: 0.08)"
    )
    parser.add_argument(
        "--vacancy-rate",
        type=float,
        default=0.04,
        help="Proportion of meters with vacant home legitimate zero/standby usage (default: 0.04)"
    )
    parser.add_argument(
        "--disconnection-rate",
        type=float,
        default=0.02,
        help="Proportion of meters with legitimate utility disconnections (default: 0.02)"
    )
    parser.add_argument(
        "--outage-prob",
        type=float,
        default=0.003,
        help="Hourly probability of feeder blackout occurrence (default: 0.003)"
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.01,
        help="Hourly probability of communication packet drop (default: 0.01)"
    )
    parser.add_argument(
        "--solar-rate",
        type=float,
        default=0.12,
        help="Proportion of meters equipped with rooftop solar PV (default: 0.12)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Directory to export CSV and JSON artifacts (default: 'data/synthetic')"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="simulated_meter_data.csv",
        help="Target readings CSV filename (default: 'simulated_meter_data.csv')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational stdout logging"
    )
    return parser


def main() -> None:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("================================================================================")
    print(" [SMART GRID] SMART METER SYNTHETIC TELEMETRY SIMULATOR")
    print("================================================================================")
    print(f" Configured Meters : {args.num_meters}")
    print(f" Timeline Horizon  : {args.days} days ({args.days * 24} hours)")
    print(f" Start Anchor Date : {args.start_date}")
    print(f" Synthetic Theft % : {args.theft_rate * 100:.1f}%")
    print(f" Vacant Home %     : {args.vacancy_rate * 100:.1f}%")
    print(f" Disconnection %   : {args.disconnection_rate * 100:.1f}%")
    print(f" Missing Packet %  : {args.missing_rate * 100:.1f}%")
    print(f" Random Seed       : {args.seed} (Deterministic)")
    print(f" Output Directory  : {args.output_dir}")
    print("--------------------------------------------------------------------------------")

    sim = SmartMeterSimulator(
        num_meters=args.num_meters,
        days=args.days,
        start_date=args.start_date,
        theft_rate=args.theft_rate,
        vacancy_rate=args.vacancy_rate,
        disconnection_rate=args.disconnection_rate,
        outage_probability=args.outage_prob,
        missing_reading_rate=args.missing_rate,
        solar_adoption_rate=args.solar_rate,
        seed=args.seed,
    )

    csv_file, meta_file, sum_file, summary = sim.export_dataset(
        output_dir=args.output_dir,
        filename=args.output_file
    )

    print("\n Export Completed Successfully!")
    print(f"  * Time-Series Readings CSV : {csv_file}")
    print(f"  * Substation Metadata CSV  : {meta_file}")
    print(f"  * Statistical Summary JSON : {sum_file}")
    print("\n Simulation Summary Highlights:")
    print(f"  * Total Readings Generated : {summary['simulation_metadata']['total_records']:,}")
    print(f"  * Total Zero Readings      : {summary['consumption_statistics']['zero_readings_total']:,}")
    print(f"    - Legitimate Zero Usage  : {summary['consumption_statistics']['zero_readings_legitimate']:,} ({summary['consumption_statistics']['legitimate_zero_percentage_of_all_zeros']}% of all zeros)")
    print(f"    - Theft Bypass Zero      : {summary['consumption_statistics']['zero_readings_simulated_theft']:,}")
    print(f"  * Synthetic Theft Readings : {summary['synthetic_ground_truth']['simulated_theft_reading_count']:,} ({summary['synthetic_ground_truth']['simulated_theft_reading_percentage']}%)")
    print(f"  * Hardware Tamper Alerts   : {summary['synthetic_ground_truth']['hardware_tamper_alerts_raised']:,}")
    print(f"  * Total Stolen/Lost Energy : {summary['synthetic_ground_truth']['total_simulated_unreported_kwh']:,.2f} kWh")
    print("================================================================================\n")


if __name__ == "__main__":
    main()

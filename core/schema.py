# core/schema.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FlightRecord:
    flight_number: str
    date: datetime
    origin: str
    destination: str
    aircraft: Optional[str]
    flight_time_min: Optional[int]
    sched_dep: datetime
    act_dep: datetime
    sched_arr: datetime
    act_arr: datetime
    dep_delay_min: int
    arr_delay_min: int

"""TransformerConfig dataclass"""

from dataclasses import dataclass


@dataclass
class Config:
    d_in: int = 1024
    d_out: int = 1024

"""Paths and shared constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
REPORTS = ROOT / "reports"

SEED = 0
ALPHA = 0.05

# LaLonde / NSW. The experimental file is the Dehejia-Wahba subset (445 rows)
# that the observational literature benchmarks against.
NSW_URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
CONTROL_URLS = {
    "cps": "https://users.nber.org/~rdehejia/data/cps_controls.dta",
    "psid": "https://users.nber.org/~rdehejia/data/psid_controls.dta",
}
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
OUTCOME = "re78"
TREATMENT = "treat"

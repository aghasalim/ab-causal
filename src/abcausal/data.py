"""Fetch the LaLonde/NSW files from NBER.

Public data, no credentials, no licence acceptance -- which is deliberate: the
whole repo has to be reproducible by whoever reads it, and a dataset behind a
login would break that for the one result that anchors everything else.
"""
from __future__ import annotations

import urllib.request

from . import config


def download() -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    targets = {"nsw_dw.dta": config.NSW_URL}
    targets.update({f"{k}_controls.dta": v for k, v in config.CONTROL_URLS.items()})
    for name, url in targets.items():
        dest = config.RAW / name
        if dest.exists():
            print(f"  have {name}")
            continue
        urllib.request.urlretrieve(url, dest)
        print(f"  fetched {name} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    download()

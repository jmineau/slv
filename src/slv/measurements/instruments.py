from abc import ABC


class Instrument(ABC):
    name: str
    display_name: str
    sample_rate: str
    pollutants: dict[str, str]  # Mapping from pollutant to concentration column name


class LGR_UGGA(Instrument):
    name = "lgr_ugga"
    display_name = "LGR UGGA"
    sample_rate = "10s"
    samples_per_hour = 309  # (3600 sec/hr - (90s flush * 4 + 50s ref * 3)) / 10s
    pollutants = {
        "CH4": "CH4d_ppm_cal",
        "CO2": "CO2d_ppm_cal",
        "H2O": "H2O_ppm",
    }


class LGR_UGGA_Manual_Cal(LGR_UGGA):
    name = "lgr_ugga_manual_cal"
    calibrated = False
    pollutants = {
        "CH4": "CH4d_ppm",
        "CO2": "CO2d_ppm",
        "H2O": "H2O_ppm",
    }


class Picarro_G2307(Instrument):
    name = "picarro_g2307"
    display_name = "Picarro G2307"
    sample_rate = "1min"
    samples_per_hour = 44  # 60 min/hr - (5min zero + 11min flush))
    pollutants = {
        "CH4": "CH4d_ppm_cal",
        "H2CO": "H2COd_ppm_cal",
    }


class Picarro_G2401(Instrument):
    name = "picarro_g2401"
    display_name = "Picarro G2401"
    sample_rate = "2s"
    pollutants = {
        "CH4": "CH4_dry_sync",
        "CO2": "CO2_dry_sync",
        "CO": "CO_sync",
        "H2O": "H2O_Sync",
    }


REGISTRY: dict[str, type[Instrument]] = {
    cls.name: cls
    for cls in [LGR_UGGA, LGR_UGGA_Manual_Cal, Picarro_G2307, Picarro_G2401]
}

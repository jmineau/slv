from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd

from lair.config import GROUP_DIR
import uataq


def load_lines(meters=False) -> gpd.GeoDataFrame:
    lines = gpd.read_file(f'{GROUP_DIR}/spatial/transportation/light_rail/UTA_TRAX.geojson')
    if meters:
        # Convert to UTM zone 12 for meter units
        lines = lines.to_crs(ccrs.UTM(12).proj4_init)
    return lines


def load_obs(cache='data.parquet', time_range=None, num_processes=1,
             buffer=50,) -> pd.DataFrame:
    if isinstance(cache, str):
        cache = Path(cache)
    if isinstance(cache, Path) and cache.exists():
        print(f'Loading cached data from {cache}')
        data = pd.read_parquet(cache)
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Longitude_deg, data.Latitude_deg),
                                crs='EPSG:4326')
        return data

    # Read in LGR data
    print('Reading LGR data...')
    lgr = uataq.read_data('trx01', instruments='lgr_ugga', lvl='calibrated',
                          time_range=time_range, num_processes=num_processes)['lgr_ugga']
    lgr_manual = uataq.read_data('trx01', instruments='lgr_ugga_manual_cal', lvl='qaqc',
                                 time_range=time_range, num_processes=num_processes)['lgr_ugga_manual_cal']

    # Standardize column names
    lgr = lgr.rename(columns={'CH4d_ppm_cal': 'CH4'})
    lgr_manual = lgr_manual.rename(columns={'CH4d_ppm': 'CH4'})

    # Filter to relevant columns and valid data
    columns = ['CH4', 'ID_CH4', 'QAQC_Flag']
    lgr = lgr[columns]
    lgr_manual = lgr_manual[columns]

    # Filter to valid observations
    valid_flags = [
        2,  # filled from backup
        1,  # manually passed
        0,  # auto passed
        -64,  # cavity temperature out of range (5, 45)
        -140, # formaldehyde out of range
        ]
    lgr = lgr[lgr.QAQC_Flag.isin(valid_flags)
            & (lgr.ID_CH4 == -10)
            & lgr.CH4.notnull()]
    lgr = lgr.drop(columns=['QAQC_Flag', 'ID_CH4'])
    lgr_manual = lgr_manual[lgr_manual.QAQC_Flag.isin(valid_flags)
                            & (lgr_manual.ID_CH4 == -10)
                            & lgr_manual.CH4.notnull()]
    lgr_manual = lgr_manual.drop(columns=['QAQC_Flag', 'ID_CH4'])

    # Combine datasets
    print('Combining LGR datasets...')
    lgr: pd.DataFrame = pd.concat([lgr, lgr_manual]).sort_index()
    lgr = lgr.rename(columns={'CH4': 'CH4_ppm'})
    del lgr_manual

    # Filter to reasonable CH4 values
    print('Filtering to reasonable CH4 values...')
    lgr = lgr[(lgr.CH4_ppm > 1.82) & (lgr.CH4_ppm < 300)]

    # Read in GPS data
    print('Reading GPS data...')
    gps = uataq.read_data('trx01', instruments='gps', lvl='final',
                          time_range=time_range, num_processes=num_processes)['gps']
    gps = gpd.GeoDataFrame(gps, geometry=gpd.points_from_xy(gps.Longitude_deg, gps.Latitude_deg),
                            crs='EPSG:4326')

    # Trim altitude outliers
    print('Trimming altitude outliers...')
    gps = gps[(gps.Altitude_msl > gps.Altitude_msl.quantile(0.01))
              & (gps.Altitude_msl < gps.Altitude_msl.quantile(0.99))]

    # Filter to locations within buffer of TRAX lines
    print('Filtering GPS points near TRAX lines...')
    lines_m = load_lines(meters=True)
    lines_m_buff = lines_m.buffer(buffer).to_frame()
    lines_buff = lines_m_buff.to_crs('EPSG:4326')
    gps = gpd.sjoin(gps, lines_buff, how='inner', predicate='within').drop(columns=['index_right'])

    # Remove gps points within storage polygon
    print('Removing GPS points within storage polygon...')
    jrrsc = gpd.read_file(data_dir / 'jrrsc.geojson')
    gps = gpd.sjoin(gps, jrrsc, how='left', predicate='within')
    gps = gps[gps.index_right.isnull()].drop(columns=['index_right'])

    gps = gps.drop(columns=['geometry'])

    # Merge LGR and GPS data
    print('Merging LGR and GPS data...')
    data = uataq.sites.MobileSite.merge_gps(lgr.rename_axis('Pi_Time', axis=0),
                                            gps,
                                            on='Pi_Time').drop(columns=['Pi_Time'])

    if isinstance(cache, Path):
        print(f'Caching data to {cache}')
        data.drop(columns=['geometry']).to_parquet(cache)

    return data

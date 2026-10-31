"""Unchanged ID/rank/geometry functions extracted from the established 2018 audit.
Only these functions are imported; none of that source's outcome/IV routines run.
Source SHA256: 4240a3ed043d4cc3983a015452c4bf278cffe6db3d2379100cfb5c828b6b77e3
Legacy 2018 city-column labels are renamed only after the unchanged function returns.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import BallTree
EARTH_KM=6371.0088

def id_hash(values):
    return hashlib.sha256(('\n'.join(sorted(map(str, values))) + '\n').encode()).hexdigest()

def build_reach(frame):
    frame = frame.sort_values(['CBSA_GEOID', 'gmap_id']).reset_index(drop=True).copy()
    frame['T09_high90'] = False
    frame['T09_type_midrank_fraction'] = np.nan
    types = []
    for typ, ix in frame.groupby('place_type', sort=True).indices.items():
        v = frame.loc[ix, 'L3__T09'].to_numpy(float)
        pct = (rankdata(v, method='average') - .5) / len(ix)
        high = pct >= .9
        frame.loc[ix, 'T09_type_midrank_fraction'] = pct
        frame.loc[ix, 'T09_high90'] = high
        types.append({'place_type': str(typ), 'reference_n': len(ix), 'high_n': int(high.sum()),
            'score_min': float(v.min()), 'score_max': float(v.max()),
            'high_min_score': float(v[high].min()) if high.any() else None,
            'reference_place_ids_sha256': id_hash(frame.loc[ix, 'gmap_id'])})
    coords = np.radians(frame[['latitude', 'longitude']].to_numpy(float))
    frame['nearest_other_T09_high_km'] = np.inf
    frame['T09_reached_1km'] = False
    cities, direct_checks = [], []
    for city, ix in frame.groupby('CBSA_GEOID', sort=True).indices.items():
        hi = ix[frame.loc[ix, 'T09_high90'].to_numpy(bool)]
        nearest = np.full(len(ix), np.inf)
        if len(hi):
            tree = BallTree(coords[hi], metric='haversine')
            distances, destinations = tree.query(coords[ix], k=min(2, len(hi)))
            distances *= EARTH_KM
            distances[hi[destinations] == ix[:, None]] = np.inf
            nearest = distances.min(axis=1)
        frame.loc[ix, 'nearest_other_T09_high_km'] = nearest
        frame.loc[ix, 'T09_reached_1km'] = nearest <= 1.0
        cities.append({'CBSA_GEOID': str(city), 'native_2018_places': len(ix), 'T09_high90_places': len(hi),
            'T09_reached_places': int((nearest <= 1.0).sum()), 'T09_reach_1km_2018': float((nearest <= 1.0).mean()),
            'origin_place_ids_sha256': id_hash(frame.loc[ix, 'gmap_id'])})
        # Fixed first/middle/last ID checks per city; no stochastic case selection.
        for offset in sorted({0, len(ix) // 2, len(ix) - 1}):
            origin = ix[offset]
            dest = hi[hi != origin]
            if len(dest):
                delta = coords[dest] - coords[origin]
                h = np.sin(delta[:, 0] / 2)**2 + np.cos(coords[origin, 0]) * np.cos(coords[dest, 0]) * np.sin(delta[:, 1] / 2)**2
                direct = float((2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).min())
            else:
                direct = np.inf
            error = abs(direct - nearest[offset]) if np.isfinite(direct) else 0.0
            assert np.isfinite(direct) == np.isfinite(nearest[offset])
            assert error < 1e-7 and (direct <= 1.0) == (nearest[offset] <= 1.0)
            direct_checks.append({'CBSA_GEOID': str(city), 'gmap_id': str(frame.loc[origin, 'gmap_id']),
                'absolute_distance_error_km': error, 'same_reach_flag': True})
    return frame, pd.DataFrame(cities), pd.DataFrame(types), pd.DataFrame(direct_checks)

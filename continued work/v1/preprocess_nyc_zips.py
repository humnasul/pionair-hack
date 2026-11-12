"""
preprocess_nyc_zips.py

PURPOSE:
- Read the NYC "ZIP Code Tabulation Areas" CSV (boundary polygons) and write nyc_zip_lookup.json,
  a compact file that nyc_context.py uses to find which ZIP area a location is in WITHOUT a network call.
- This file is a location helper only. It is not used in the risk model.

WHAT THE CSV CONTAINS (checked against the file, not assumed):
- 221 ZIP areas that touch New York City. Their boundary polygons are CLIPPED TO THE CITY: for the 209 ZIPs with an
  NYC prefix the polygon area matches the ZIP's full area (ratio about 1.0), while for ZIPs that are mostly outside
  the city (Jersey City 07305, Mount Vernon 10550, Yonkers 10704/10705, Pelham 10803, Nassau ZIPs 11001, 11020,
  11021, 11040, 11580) the polygon is only the small sliver inside NYC (ratio 0.00 to 0.12).
  11004 and 11005 are almost entirely inside the city (ratio 0.99).
- Therefore a point inside ANY polygon in this file is inside NYC.
- The CSV has no borough column. Borough is taken from the ZIP prefix where it is unambiguous
  (100-102 Manhattan, 103 Staten Island, 104 Bronx, 112 Brooklyn, 111/113/114/116 Queens). For every other ZIP the
  borough is INFERRED from the nearest ZIP with a known borough and marked "inferred": true. Review the printout.
- "partial": true marks ZIPs with less than half of their area inside the city. nyc_context.py does not treat a
  typed partial ZIP as an NYC ZIP, but does treat a location inside its sliver as NYC.

USAGE:
    python preprocess_nyc_zips.py                                   # uses the default file name below
    python preprocess_nyc_zips.py path/to/ZIP_Code_Tabulation_Areas.csv
"""

import csv
import json
import math
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "ZIP_Code_Tabulation_Areas_20261108.csv"
OUT = "nyc_zip_lookup.json"
SIMPLIFY_TOLERANCE = 0.0001  # degrees (about 10 m): shrinks the file; far finer than a ZIP-level lookup needs
DECIMALS = 5
PARTIAL_BELOW = 0.5  # polygon area / full ZIP area under this means the ZIP is mostly outside NYC

PREFIX_TO_BOROUGH = {
    "100": "Manhattan", "101": "Manhattan", "102": "Manhattan",
    "103": "Staten Island",
    "104": "Bronx",
    "112": "Brooklyn",
    "111": "Queens", "113": "Queens", "114": "Queens", "116": "Queens",
}

csv.field_size_limit(sys.maxsize)  # the Geometry column is very large


def parse_rings(wkt):
    """Return every ring of a POLYGON/MULTIPOLYGON as a list of (lon, lat) points.

    Rings are not grouped by polygon: nyc_context.py uses an even-odd test, which handles holes correctly
    for non-overlapping polygons.
    """
    rings = []
    for chunk in re.findall(r"\(([^()]+)\)", wkt):
        pts = []
        for pair in chunk.split(","):
            x, y = pair.split()[:2]
            pts.append((float(x), float(y)))
        if len(pts) >= 4:
            rings.append(pts)
    return rings


def ring_area_m2(pts):
    """Approximate area of a ring in square meters (local equirectangular projection)."""
    lat0 = sum(p[1] for p in pts) / len(pts)
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110574
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i][0] * kx, pts[i][1] * ky
        x2, y2 = pts[(i + 1) % len(pts)][0] * kx, pts[(i + 1) % len(pts)][1] * ky
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def simplify(points, tol):
    """Ramer-Douglas-Peucker simplification (iterative) of a closed ring."""
    n = len(points)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        (ax, ay), (bx, by) = points[a], points[b]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        best, idx = 0.0, None
        for i in range(a + 1, b):
            px, py = points[i]
            if denom == 0:
                d = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
                d = ((px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2) ** 0.5
            if d > best:
                best, idx = d, i
        if idx is not None and best > tol:
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [p for p, k in zip(points, keep) if k]


def main():
    zips = {}
    with open(SRC, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            z = row["Zip Code"].strip()
            rings = parse_rings(row["Geometry"])
            if not rings:
                continue
            xs = [p[0] for r in rings for p in r]
            ys = [p[1] for r in rings for p in r]
            full_area = float(row["Land Area"]) + float(row["Water Area"])
            inside_area = sum(ring_area_m2(r) for r in rings)
            slim = []
            for r in rings:
                s = simplify(r, SIMPLIFY_TOLERANCE)
                if len(s) >= 4:
                    slim.append([[round(x, DECIMALS), round(y, DECIMALS)] for x, y in s])
            coverage = round(inside_area / full_area, 3) if full_area else 0.0
            entry = {
                "bbox": [round(min(xs), DECIMALS), round(min(ys), DECIMALS), round(max(xs), DECIMALS), round(max(ys), DECIMALS)],
                "rings": slim,
                "coverage": coverage,
            }
            borough = PREFIX_TO_BOROUGH.get(z[:3])
            if borough:
                entry["borough"] = borough
            if coverage < PARTIAL_BELOW:
                entry["partial"] = True
            zips[z] = entry

    # infer a borough for ZIPs without an unambiguous prefix, from the nearest ZIP with a known borough
    known = {z: e for z, e in zips.items() if "borough" in e}
    for z, e in zips.items():
        if "borough" in e:
            continue
        pts = [p for r in e["rings"] for p in r]
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        best, best_b, best_z = 1e9, None, None
        for kz, ke in known.items():
            for r in ke["rings"]:
                for x, y in r:
                    d = (x - cx) ** 2 + (y - cy) ** 2
                    if d < best:
                        best, best_b, best_z = d, ke["borough"], kz
        e["borough"] = best_b
        e["inferred"] = True
        print(f"  inferred borough for {z}: {best_b} (nearest known ZIP {best_z}, "
              f"{math.sqrt(best) * 111:.1f} km away, coverage {e['coverage']})")

    assert len(zips) > 150, f"Only {len(zips)} ZIPs parsed - is this the right file?"
    payload = {
        "source": "NYC Open Data - ZIP Code Tabulation Areas (boundaries clipped to NYC).",
        "simplify_tolerance_deg": SIMPLIFY_TOLERANCE,
        "zips": zips,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    partial = sorted(z for z, e in zips.items() if e.get("partial"))
    inferred = sorted(z for z, e in zips.items() if e.get("inferred"))
    print(f"Parsed {len(zips)} ZIP areas. Borough from prefix: {len(zips) - len(inferred)}; inferred: {len(inferred)}.")
    print("Mostly outside NYC (partial):", partial)
    print(f"[SUCCESS] Wrote {OUT}")


if __name__ == "__main__":
    main()

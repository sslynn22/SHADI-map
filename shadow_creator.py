"""
- pysolar로 시각(STAMP)에 맞는 그림자 생성
- PostGIS에 append + dedup(md5 WKB) 적재
- building/tree/shelter → union 테이블 생성/갱신

[변경점]
- STAMP가 'YYYYMMDD'면 08:00~19:00 매 정시에 대해 일괄 생성/적재.
  (기존 'YYYYMMDD_HHMM'은 단일 시각 처리 그대로 유지)
"""

import os, re, math, time, warnings, hashlib, datetime, pytz
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection, box
from shapely.affinity import translate, scale, rotate
from shapely.ops import unary_union, transform, snap as shp_snap
from shapely.validation import make_valid
from pyproj import Transformer
from pysolar.solar import get_altitude, get_azimuth

# -------------------- 환경 변수 --------------------
STAMP       = os.getenv("STAMP", "20250818")          # 'yyyyMMdd_HHmm' 또는 'yyyyMMdd' (날짜만 → 08~19시 일괄)
PG_URL      = os.getenv("PG_URL", "postgresql://postgres:0202@localhost:5432/shadi")
SCHEMA      = os.getenv("SCHEMA", "public")
SRID        = int(os.getenv("SRID", "4326"))
REBUILD_UNION = bool(int(os.getenv("REBUILD_UNION", "0"))) # 1이면 union 재생성

TREES_CSV   = os.getenv("TREES_CSV",  "data/대전광역시_가로수 현황_20221201.csv")
SHELTER_CSV = os.getenv("SHELTER_CSV","data/대전광역시 유성구_그늘막쉼터_20240920.csv")
BUILDING_SHP= os.getenv("BUILDING_SHP","data/CH_D010_00_20250731.shp")

CENTER_CNU  = (36.36917, 127.34515)  # (lat, lon)
DIST_M      = int(os.getenv("DIST_M","2000"))
USE_OSM_BUILDING = bool(int(os.getenv("USE_OSM_BUILDING","1")))
TIMEZONE    = "Asia/Seoul"

# -------------------- 기본 설정 --------------------
tz = pytz.timezone(TIMEZONE)
warnings.filterwarnings("ignore", message="I don't know about leap seconds")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
to4326 = Transformer.from_crs(5179, 4326, always_xy=True)

def parse_stamp_to_dt(stamp: str) -> datetime.datetime:
    dt = datetime.datetime.strptime(stamp, "%Y%m%d_%H%M")
    return tz.localize(dt)

# 날짜만 들어올 수 있으므로 전역 NOW는 루프에서 설정
NOW = None

def deg_buffer(center_latlon, dist_m):
    deg = dist_m / 111_320.0
    lat, lon = center_latlon
    return box(lon-deg, lat-deg, lon+deg, lat+deg)
BBOX_POLY = deg_buffer(CENTER_CNU, DIST_M)

# -------------------- 유틸/그림자 함수 --------------------
def shadow_len(h, alt_deg): return 0 if alt_deg <= 0 else h / math.tan(math.radians(alt_deg))

def tree_shadow_ellipse(lat, lon, r_m, alt, azi):
    if alt <= 0: return Polygon()
    x0, y0 = to5179.transform(lon, lat)
    circle = Point(x0, y0).buffer(r_m)
    stretch = 1 / math.tan(math.radians(alt))
    ellip   = scale(circle, 1, stretch, origin=(x0, y0))
    ellip   = rotate(ellip, (azi + 90) % 360, origin=(x0, y0))
    L   = shadow_len(r_m, alt)
    dx  = (L/2) * math.sin(math.radians(azi))
    dy  = (L/2) * math.cos(math.radians(azi))
    ellip = translate(ellip, xoff=dx, yoff=dy)
    shadow = transform(lambda x,y,z=None: to4326.transform(x,y), ellip)
    return make_valid(shadow)

def shelter_shadow_octagon(lat, lon, diameter_m, height_m, alt, azi):
    if alt <= 0: return Polygon()
    r = max(0.1, diameter_m / 2.0)
    angles = [math.radians(22.5 + 45*i) for i in range(8)]
    base_pts = [(r*math.cos(th), r*math.sin(th)) for th in angles]
    base = Polygon(base_pts)
    L = shadow_len(height_m, alt)
    stretch = (L / r) if r>0 else 1.0
    sh = scale(base, 1, stretch, origin=(0, 0))
    sh = rotate(sh, (azi + 90) % 360, origin=(0, 0))
    dx, dy = (L/2)*math.sin(math.radians(azi)), (L/2)*math.cos(math.radians(azi))
    sh = translate(sh, xoff=dx, yoff=dy)
    cx, cy = to5179.transform(lon, lat)
    sh = translate(sh, xoff=cx, yoff=cy)
    return make_valid(transform(lambda x,y,z=None: to4326.transform(x,y), sh))

def polygon_parts(geom):
    if geom is None or geom.is_empty: return []
    if isinstance(geom, Polygon): return [geom]
    if isinstance(geom, MultiPolygon): return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out=[]
        for g in geom.geoms:
            if isinstance(g, Polygon): out.append(g)
            elif isinstance(g, MultiPolygon): out.extend(list(g.geoms))
        return out
    return []

def building_shadow_polygon(geom, h, alt, azi):
    if alt <= 0: return Polygon()
    polys = [make_valid(p) for p in polygon_parts(geom)]
    polys = [p for p in polys if (p and not p.is_empty)]
    if not polys: return Polygon()
    L  = shadow_len(h, alt)
    ref_lat = polys[0].centroid.y
    dy =  L*math.cos(math.radians(azi)) / 111_320.0
    dx = (L*math.sin(math.radians(azi))
          / (40075_000*math.cos(math.radians(ref_lat))/360.0))
    def _shadow(p):
        src  = list(p.exterior.coords)
        dest = [(x+dx, y+dy) for x,y in src]
        quads = [Polygon([src[i], src[i+1], dest[i+1], dest[i]]) for i in range(len(src)-1)]
        return unary_union([Polygon(dest), *quads])
    pieces = [_shadow(p) for p in polys]
    if not pieces: return Polygon()
    sh = unary_union(pieces)
    sh = sh.difference(unary_union(polys))
    return make_valid(sh)

def to_float_or_none(val):
    num = re.sub(r"[^0-9.]", "", str(val)); return float(num) if num else None

def canonicalize(g):
    if g is None or g.is_empty: return None
    g5179 = transform(lambda x,y,z=None: to5179.transform(x,y), g)
    g5179 = make_valid(g5179)
    g5179 = shp_snap(g5179, g5179, 0)
    g5179 = transform(lambda x,y,z=None: (round(x/0.05)*0.05, round(y/0.05)*0.05), g5179)
    g5179 = g5179.buffer(0)
    g4326 = transform(lambda x,y,z=None: to4326.transform(x,y), g5179)
    return make_valid(g4326)

def md5_wkb(g): return hashlib.md5(g.wkb).hexdigest()

# -------------------- 데이터 로드 --------------------
def load_trees():
    if not os.path.exists(TREES_CSV):
        print(f"[warn] trees csv not found: {TREES_CSV}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    trees = pd.read_csv(TREES_CSV, encoding="euc-kr")
    trees["위도"] = pd.to_numeric(trees["위도"], errors="coerce")
    trees["경도"] = pd.to_numeric(trees["경도"], errors="coerce")
    gdf = gpd.GeoDataFrame(
        trees.dropna(subset=["경도","위도"]),
        geometry=[Point(xy) for xy in zip(trees["경도"], trees["위도"])],
        crs="EPSG:4326",
    )
    return gpd.clip(gdf, BBOX_POLY)

def load_shelters():
    if not os.path.exists(SHELTER_CSV):
        print(f"[warn] shelter csv not found: {SHELTER_CSV}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    shel = pd.read_csv(SHELTER_CSV, encoding="euc-kr")
    shel["위도"] = pd.to_numeric(shel["위도"], errors="coerce")
    shel["경도"] = pd.to_numeric(shel["경도"], errors="coerce")
    looks_swapped = (shel["위도"].between(120, 140).mean() > 0.5) and (shel["경도"].between(30, 45).mean() > 0.5)
    lon = shel["위도"] if looks_swapped else shel["경도"]
    lat = shel["경도"] if looks_swapped else shel["위도"]
    gdf = gpd.GeoDataFrame(
        shel.assign(lon=lon, lat=lat).dropna(subset=["lon","lat"]),
        geometry=[Point(xy) for xy in zip(lon, lat)],
        crs="EPSG:4326"
    )
    return gpd.clip(gdf, BBOX_POLY)

def _filter_polygonish(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or len(gdf) == 0: return gdf
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon","MultiPolygon","GeometryCollection","MultiSurface"])]
    return gdf

def load_buildings():
    shp = None
    if os.path.exists(BUILDING_SHP):
        shp = gpd.read_file(BUILDING_SHP, encoding="euc-kr").to_crs(4326)
        if "A4" in shp.columns:
            shp = shp[shp["A4"].astype(str).str.contains("대전", na=False)]
        shp = gpd.clip(shp, BBOX_POLY)
        shp = _filter_polygonish(shp)
        print(f"  - shapefile buildings: {len(shp)}")
    else:
        print(f"[warn] building shapefile not found: {BUILDING_SHP}")
    return shp

# -------------------- 그림자 생성 --------------------
def generate_tree_shadows(trees_gdf: gpd.GeoDataFrame):
    out = []
    for _, r in trees_gdf.iterrows():
        lat, lon = r["위도"], r["경도"]
        alt = get_altitude(lat, lon, NOW)
        azi = (get_azimuth(lat, lon, NOW) + 180) % 360
        poly = tree_shadow_ellipse(lat, lon, r_m=3.0, alt=alt, azi=azi)
        if not poly.is_empty: out.append(poly)
    return out

def generate_shelter_shadows(shel_gdf: gpd.GeoDataFrame):
    out = []
    for _, r in shel_gdf.iterrows():
        lat, lon = r["lat"], r["lon"]
        canopy_d  = to_float_or_none(r.get("펼침지름")) or 3.0
        height_m  = to_float_or_none(r.get("전체높이")) or 3.0
        alt = get_altitude(lat, lon, NOW)
        azi = (get_azimuth(lat, lon, NOW) + 180) % 360
        poly = shelter_shadow_octagon(lat, lon, canopy_d, height_m, alt, azi)
        if not poly.is_empty: out.append(poly)
    return out

def generate_building_shadows(shp_gdf: gpd.GeoDataFrame, include_osm=True):
    layers = []
    shp_union = None
    if shp_gdf is not None and len(shp_gdf) > 0:
        for _, r in shp_gdf.iterrows():
            poly = make_valid(r.geometry)
            if poly.is_empty: continue
            floors = pd.to_numeric(r.get("A25"), errors="coerce")
            h = floors*3 if not pd.isna(floors) else 10.0
            c = poly.centroid
            alt = get_altitude(c.y, c.x, NOW)
            azi = (get_azimuth(c.y, c.x, NOW) + 180) % 360
            s_poly = building_shadow_polygon(poly, h, alt, azi)
            if not s_poly.is_empty: layers.append(s_poly)
        shp_union = unary_union(layers) if layers else None

    if include_osm:
        try:
            osm = ox.features_from_polygon(BBOX_POLY, tags={"building": True}).to_crs(4326)
        except Exception:
            osm = ox.features_from_point(CENTER_CNU, dist=DIST_M, tags={"building": True}).to_crs(4326)
        osm = _filter_polygonish(osm)
        print(f"  - osm buildings: {len(osm)}")
        for _, row in osm.iterrows():
            poly = make_valid(row.geometry)
            if poly.is_empty: continue
            if shp_union is not None and poly.intersects(shp_union): continue
            h_height = to_float_or_none(row.get("height"))
            raw_lv   = row.get("building:levels")
            lv_list  = [int(x) for x in re.findall(r'\d+', str(raw_lv) if raw_lv else "")]
            h_levels = (max(lv_list)*3) if lv_list else None
            h = max([v for v in (h_height, h_levels) if v is not None], default=10.0)
            c = poly.centroid
            alt = get_altitude(c.y, c.x, NOW)
            azi = (get_azimuth(c.y, c.x, NOW) + 180) % 360
            s_poly = building_shadow_polygon(poly, h, alt, azi)
            if not s_poly.is_empty: layers.append(s_poly)
    return layers

# -------------------- DB 적재(append + dedup) --------------------
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import sql

def table_name(base: str) -> str:
    return f"{base}_{STAMP}"

def ensure_table(conn, schema: str, table: str):
    idx_uhix = f"{schema}_{table}_uhix"
    idx_gix  = f"{schema}_{table}_gix"
    with conn.cursor() as cur:
        # 1) 테이블(geometry만) 보장
        cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} (geometry geometry(Geometry, %s));")
                    .format(sql.Identifier(schema), sql.Identifier(table)),
                    (SRID,))
        # 2) ghash 컬럼 보장 + 채우기
        cur.execute(sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS ghash text;")
                    .format(sql.Identifier(schema), sql.Identifier(table)))
        cur.execute(sql.SQL("UPDATE {}.{} SET ghash = md5(ST_AsEWKB(geometry)) WHERE ghash IS NULL AND geometry IS NOT NULL;")
                    .format(sql.Identifier(schema), sql.Identifier(table)))
        # 3) ghash 중복 제거(있다면 앞 항목만 유지)
        cur.execute(sql.SQL("""
            DELETE FROM {}.{} a
            USING {}.{} b
            WHERE a.ctid < b.ctid
              AND a.ghash = b.ghash
              AND a.ghash IS NOT NULL;
        """).format(sql.Identifier(schema), sql.Identifier(table),
                    sql.Identifier(schema), sql.Identifier(table)))
        # 4) 인덱스들
        cur.execute(sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{} (ghash);")
                    .format(sql.Identifier(idx_uhix), sql.Identifier(schema), sql.Identifier(table)))
        cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} USING GIST (geometry);")
                    .format(sql.Identifier(idx_gix), sql.Identifier(schema), sql.Identifier(table)))

def append_geoms(conn, schema: str, table: str, geoms):
    if not geoms:
        print(f"[{schema}.{table}] nothing to insert.")
        return 0
    rows = []
    for g in geoms:
        gc = canonicalize(g)
        if gc is None or gc.is_empty: continue
        rows.append( (gc.wkb_hex, md5_wkb(gc)) )
    if not rows:
        print(f"[{schema}.{table}] all filtered out after canonicalize.")
        return 0
    with conn.cursor() as cur:
        cur.execute("SET LOCAL synchronous_commit = OFF;")
        cur.execute("SET LOCAL jit = OFF;")
        cur.execute("SET LOCAL work_mem = '256MB';")
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tmp_ins (wkb_hex text, ghash text) ON COMMIT DROP;")
        cur.execute("TRUNCATE _tmp_ins;")
        execute_values(cur, "INSERT INTO _tmp_ins (wkb_hex, ghash) VALUES %s", rows, page_size=1000)
        cur.execute(sql.SQL("""
            INSERT INTO {}.{}(geometry, ghash)
            SELECT ST_SetSRID(ST_GeomFromWKB(decode(wkb_hex,'hex')),%s) AS g, ghash
            FROM _tmp_ins
            ON CONFLICT (ghash) DO NOTHING;
        """).format(sql.Identifier(schema), sql.Identifier(table)), (SRID,))
        cur.execute("SELECT COUNT(*) FROM _tmp_ins;")
        return cur.fetchone()[0]

def rebuild_or_keep_union(conn, schema: str, table_u: str, sources: list):
    idx_gix = f"{schema}_{table_u}_gix"
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} (geometry geometry(Geometry, %s));")
                    .format(sql.Identifier(schema), sql.Identifier(table_u)), (SRID,))
        cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} USING GIST (geometry);")
                    .format(sql.Identifier(idx_gix), sql.Identifier(schema), sql.Identifier(table_u)))
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}.{};").format(sql.Identifier(schema), sql.Identifier(table_u)))
        union_rows = cur.fetchone()[0]

        if REBUILD_UNION:
            print("[union] REBUILD_UNION=1 → refreshing union table")
            cur.execute(sql.SQL("DELETE FROM {}.{};").format(sql.Identifier(schema), sql.Identifier(table_u)))
            union_rows = 0

        if union_rows == 0:
            parts = []
            for (s, t) in sources:
                cur.execute("SELECT to_regclass(%s);", (f"{s}.{t}",))
                if cur.fetchone()[0] is None:
                    continue
                parts.append(sql.SQL("""
                    SELECT ST_CollectionExtract(
                             ST_Buffer(
                               ST_MakeValid(
                                 ST_SnapToGrid(ST_Transform(geometry, 5179), 0.05)
                               ), 0
                             ), 3
                           ) AS g5179
                    FROM {}.{}
                """).format(sql.Identifier(s), sql.Identifier(t)))
            if not parts:
                print("[union] no sources -> skip")
                return
            union_sql = sql.SQL("""
                WITH all_shadows AS (
                  {parts}
                ),
                nonempty AS (
                  SELECT g5179 FROM all_shadows
                  WHERE g5179 IS NOT NULL AND NOT ST_IsEmpty(g5179)
                ),
                u AS (
                  SELECT ST_UnaryUnion(ST_Collect(g5179)) AS g
                  FROM nonempty
                )
                INSERT INTO {schema}.{table}(geometry)
                SELECT ST_Transform(g, %s) FROM u;
            """).format(
                parts=sql.SQL(" UNION ALL ").join(parts),
                schema=sql.Identifier(schema),
                table=sql.Identifier(table_u),
            )
            cur.execute(union_sql, (SRID,))
            cur.execute(sql.SQL("ANALYZE {}.{};").format(sql.Identifier(schema), sql.Identifier(table_u)))
            print("[union] built.")
        else:
            print(f"[union] keep existing rows: {union_rows} (set REBUILD_UNION=1 to refresh)")

# -------------------- 보조: 날짜→시간리스트 --------------------
def _stamps_for_date(date_str: str, start_h=8, end_h=19):
    """'YYYYMMDD' → ['YYYYMMDD_0800', ..., 'YYYYMMDD_1900']"""
    return [f"{date_str}_{h:02d}00" for h in range(start_h, end_h+1)]

def _is_date_only(s: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", s))

def _is_full_stamp(s: str) -> bool:
    return bool(re.fullmatch(r"\d{8}_\d{4}", s))

# -------------------- 메인 --------------------
def main():
    global NOW, STAMP  # 전역을 이 함수에서 갱신할 것이므로 참조 전에 선언
    t_all = time.time()

    # STAMP 해석: 날짜만 → 시간 리스트, 완전한 스탬프 → 단일
    if _is_date_only(STAMP):
        stamps = _stamps_for_date(STAMP, 8, 19)
    elif _is_full_stamp(STAMP):
        stamps = [STAMP]
    else:
        raise ValueError("STAMP 형식이 올바르지 않습니다. 'YYYYMMDD' 또는 'YYYYMMDD_HHMM'")

    import psycopg2
    with psycopg2.connect(PG_URL) as conn:
        conn.autocommit = True

        for st in stamps:
            # 각 시각별 실행 컨텍스트 설정
            STAMP = st
            NOW = parse_stamp_to_dt(st)

            t0 = time.time()
            print(f"\n▶ START make_and_load_shadows (STAMP={STAMP}, SRID={SRID})")

            trees_gdf   = load_trees()
            shelters_gdf= load_shelters()
            shp_gdf     = load_buildings()

            print(f" - trees in bbox: {len(trees_gdf)}")
            print(f" - shelters in bbox: {len(shelters_gdf)}")
            print(f" - shapefile buildings in bbox: {0 if shp_gdf is None else len(shp_gdf)}")

            print("• generating shadows with pysolar …")
            tree_geoms    = generate_tree_shadows(trees_gdf) if len(trees_gdf)>0 else []
            shelter_geoms = generate_shelter_shadows(shelters_gdf) if len(shelters_gdf)>0 else []
            building_geoms= generate_building_shadows(shp_gdf, include_osm=USE_OSM_BUILDING)
            print(f"   → tree: {len(tree_geoms)}, shelter: {len(shelter_geoms)}, building: {len(building_geoms)}")

            tbl_b = table_name("shadow_building")
            tbl_t = table_name("shadow_tree")
            tbl_s = table_name("shadow_shelter")
            tbl_u = table_name("shadow_union")

            ensure_table(conn, SCHEMA, tbl_b)
            ensure_table(conn, SCHEMA, tbl_t)
            ensure_table(conn, SCHEMA, tbl_s)

            ins_b = append_geoms(conn, SCHEMA, tbl_b, building_geoms)
            ins_t = append_geoms(conn, SCHEMA, tbl_t, tree_geoms)
            ins_s = append_geoms(conn, SCHEMA, tbl_s, shelter_geoms)
            print(f" • appended (attempted rows): building={ins_b}, tree={ins_t}, shelter={ins_s} (dedup by ghash)")

            rebuild_or_keep_union(conn, SCHEMA, tbl_u, [(SCHEMA, tbl_b), (SCHEMA, tbl_t), (SCHEMA, tbl_s)])

            print(f"DONE STAMP {STAMP} in {time.time()-t0:.1f}s")

    print(f"\nALL DONE in {time.time()-t_all:.1f}s")

if __name__ == "__main__":
    main()
    print(f"KAKAO_JS_KEY={KAKAO_JS_KEY}")
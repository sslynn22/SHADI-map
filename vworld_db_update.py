# vworld_db_update.py
import os, re, math, hashlib, requests, datetime, warnings
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point
from shapely import affinity, wkb
from shapely.ops import unary_union, transform
from shapely.validation import make_valid
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry as GA_Geom
from pysolar.solar import get_altitude, get_azimuth
from pyproj import Transformer
import pytz


# ───────── optional: .env 지원 ─────────
try:
    from dotenv import load_dotenv
    load_dotenv("api_keys.env")
except Exception:
    pass

warnings.filterwarnings("ignore", message="I don't know about leap seconds")

# ───────── 설정 ─────────
VWORLD_KEY    = os.getenv("VWORLD_KEY")
VWORLD_DOMAIN = os.getenv("VWORLD_DOMAIN", "127.0.0.1")
if not VWORLD_KEY:
    raise SystemExit("❌ 환경변수/환경파일에 VWORLD_KEY가 필요합니다.")

PG_URL = os.getenv("PG_URL", "postgresql://postgres:804009@localhost:5432/shadi")

TYPENAME = "lt_c_bldginfo"

# 예시 BBOX (원하는 범위로 바꾸세요)
MINX, MINY, MAXX, MAXY = 127.345538, 36.360842, 127.354279, 36.367062

# 분석 타임스탬프(모든 테이블 이름 재사용) — 기본값 예: 20240731_1800
# 'YYYYMMDD_HHMM' 또는 'YYYYMMDD'(날짜만 → 08~19시 일괄)
STAMP       = os.getenv("STAMP", "20250818")
TB_BUILDING = f"shadow_building_{STAMP}"
TB_TREE     = f"shadow_tree_{STAMP}"
TB_SHELTER  = f"shadow_shelter_{STAMP}"
TB_UNION    = f"shadow_union_{STAMP}"

# 높이 추정 규칙
DEFAULT_FLOOR_H = 3.0   # m
DEFAULT_HEIGHT  = 10.0  # m

# sweep 샘플링(값↑ → 경계 부드러움↑, 계산량↑)
SWEEP_STEPS = 10

# 건물 바닥(footprint) 지우기용 미세 버퍼 (단위: m, EPSG:5179 기준)
BASE_ERASE_BUFFER_M = 0.05

# IoU 중복 판정 임계값 (0~1): 기존 그림자와 0.8 이상 겹치면 중복으로 간주해 skip
IOU_DUP_THRESH = 0.80

# 좌표변환기
to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
to4326 = Transformer.from_crs(5179, 4326, always_xy=True)


# ───────── 유틸 ─────────
def _geom_md5(g):
    if g is None or g.is_empty:
        return None
    return hashlib.md5(wkb.dumps(g, hex=False)).hexdigest()

def _to_multi(g):
    if g is None or g.is_empty:
        return None
    if isinstance(g, MultiPolygon):
        return g
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, GeometryCollection):
        polys=[]
        for sub in g.geoms:
            if isinstance(sub, Polygon):
                polys.append(sub)
            elif isinstance(sub, MultiPolygon):
                polys.extend(list(sub.geoms))
        return MultiPolygon(polys) if polys else None
    return None

def parse_stamp_to_dt(stamp: str) -> datetime.datetime:
    # "yyyyMMdd_HHmm" → tz-aware datetime(Asia/Seoul)
    kst = pytz.timezone("Asia/Seoul")
    dt = datetime.datetime.strptime(stamp, "%Y%m%d_%H%M")
    return kst.localize(dt)

def _stamps_for_date(date_str: str, start_h=8, end_h=19):
    """'YYYYMMDD' → ['YYYYMMDD_0800', ..., 'YYYYMMDD_1900']"""
    return [f"{date_str}_{h:02d}00" for h in range(start_h, end_h+1)]

def _is_date_only(s: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", s))

def _is_full_stamp(s: str) -> bool:
    return bool(re.fullmatch(r"\d{8}_\d{4}", s))


# ───────── VWorld 건물 수집 ─────────
def fetch_vworld_bldg(minx, miny, maxx, maxy):
    from urllib.parse import urlencode
    base = "https://api.vworld.kr/req/wfs"
    params = {
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typeName": TYPENAME, "srsName": "CRS:84",
        "bbox": f"{minx},{miny},{maxx},{maxy},CRS:84",
        "outputFormat": "application/json", "count": 1000,
        "key": VWORLD_KEY, "domain": VWORLD_DOMAIN,
    }
    url = f"{base}?{urlencode(params)}"
    print("WFS URL:", url)

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()

    feats = js.get("features") if isinstance(js, dict) else None
    if not feats:
        print("ℹ️ VWorld features=0")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame.from_features(feats)

    # GeoSeries → GeoDataFrame 승격
    if isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.to_frame(name="__geom__")
        gdf = gpd.GeoDataFrame(gdf, geometry="__geom__", crs="EPSG:4326")

    # 활성 geometry 확정
    if not hasattr(gdf, "geometry") or not isinstance(getattr(gdf, "geometry"), gpd.GeoSeries):
        cand = [c for c in gdf.columns if str(getattr(gdf[c], "dtype", "")) == "geometry"]
        if not cand:
            for c in gdf.columns:
                try:
                    if gdf[c].apply(lambda x: hasattr(x, "geom_type")).any():
                        cand = [c]; break
                except Exception:
                    pass
        if not cand:
            print("DEBUG columns:", list(gdf.columns))
            raise SystemExit("❌ geometry 열을 찾지 못했습니다.")
        gdf = gpd.GeoDataFrame(gdf, geometry=cand[0], crs="EPSG:4326")

    # CRS 정리
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    else:
        crs_txt = str(gdf.crs).upper()
        if crs_txt in ("CRS:84", "OGC:CRS84"):
            gdf = gdf.to_crs(4326)

    # 유효성 & 멀티폴리곤화
    try:
        gdf = gdf.set_geometry(gdf.make_valid().geometry)
    except Exception:
        gdf = gdf.set_geometry(gdf.buffer(0))

    gdf = gdf.set_geometry(gdf.geometry.apply(_to_multi))
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    print(f"VWorld buildings: {len(gdf)} | geom_name: {gdf.geometry.name} | crs: {gdf.crs}")
    return gdf


# ───────── 태양 위치 ─────────
def compute_sun(dt_local, lat, lon):
    kst = pytz.timezone("Asia/Seoul")
    dt_kst = dt_local if dt_local.tzinfo else kst.localize(dt_local)
    dt_utc = dt_kst.astimezone(pytz.utc)
    alt = get_altitude(lat, lon, dt_utc)
    azi = get_azimuth(lat, lon, dt_utc)
    print(f"Sun alt={alt:.2f}°, az={azi:.2f}° (KST {dt_kst}, UTC {dt_utc})")
    return alt, azi


# ───────── 그림자 생성 ─────────
def sweep_shadow(geom5179, L, dir_deg, n=SWEEP_STEPS):
    if L <= 0:
        return None
    rad = math.radians(dir_deg)
    dx, dy = L*math.sin(rad), L*math.cos(rad)  # x=East, y=North
    geoms = [affinity.translate(geom5179, xoff=dx*i/n, yoff=dy*i/n) for i in range(n+1)]
    out = geoms[0]
    for g in geoms[1:]:
        out = out.union(g)
    return out

def footprints_to_shadows(gdf4326, alt_deg, az_deg):
    if alt_deg <= 0:
        print("⚠️ 태양고도 ≤ 0 → 그림자 없음")
        return gpd.GeoDataFrame({"geometry":[]}, geometry="geometry", crs="EPSG:4326")

    dir_deg = (az_deg + 180.0) % 360.0
    gdf = gdf4326.copy()

    # 높이 추정
    def _est_h(row):
        h = None
        if "height" in row and row["height"] not in (None, "", 0):
            try: h = float(row["height"])
            except: h = None
        if h is None and "grnd_flr" in row and row["grnd_flr"] not in (None, "", 0):
            try: h = float(row["grnd_flr"]) * DEFAULT_FLOOR_H
            except: h = None
        return h if (h and h > 0) else DEFAULT_HEIGHT
    gdf["est_h"] = gdf.apply(_est_h, axis=1)

    # 5179로 변환
    g5179 = gdf.to_crs(5179)

    # 바닥 유니온(살짝 확장) → 그림자에서 제거
    fp_union = g5179.geometry.unary_union.buffer(BASE_ERASE_BUFFER_M)

    # 그림자 길이
    alt_rad = math.radians(alt_deg)
    g5179["L"] = gdf["est_h"].apply(lambda h: h / max(math.tan(alt_rad), 1e-6))

    shadows = []
    for geom, L in zip(g5179.geometry, g5179["L"]):
        if geom is None or geom.is_empty:
            continue
        sh = sweep_shadow(geom, L, dir_deg, n=SWEEP_STEPS)
        if sh is None or sh.is_empty:
            continue
        sh = sh.difference(fp_union)  # 바닥 제거
        if sh.is_empty:
            continue
        shadows.append(sh)

    if not shadows:
        return gpd.GeoDataFrame({"geometry":[]}, geometry="geometry", crs="EPSG:4326")

    out5179 = gpd.GeoDataFrame(geometry=shadows, crs=5179)
    out4326 = out5179.to_crs(4326)

    # 정리
    try:
        out4326 = out4326.set_geometry(out4326.make_valid().geometry)
    except Exception:
        out4326 = out4326.set_geometry(out4326.buffer(0))
    out4326 = out4326.set_geometry(out4326.geometry.apply(_to_multi))
    out4326 = out4326[out4326.geometry.notna()]
    out4326 = out4326[~out4326.geometry.is_empty]
    if out4326.geometry.name != "geometry":
        out4326 = out4326.rename(columns={out4326.geometry.name: "geometry"}).set_geometry("geometry")

    print(f"Shadows created (footprint removed): {len(out4326)}")
    return out4326[["geometry"]]


# ───────── DB 준비/적재 ─────────
def ensure_tables(engine):
    """
    운영 테이블은 DROP/TRUNCATE 하지 않음. 없으면 생성만.
    과거에 UNLOGGED로 만들어졌던 경우를 대비해 SET LOGGED도 실행.
    """
    with engine.begin() as conn:
        for tb in (TB_BUILDING, TB_TREE, TB_SHELTER, TB_UNION):
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {tb}(
                    geometry geometry(MULTIPOLYGON,4326) NOT NULL
                );
            """))
            # LOGGED 보장 (비정상 종료시 데이터 유실 방지)
            conn.execute(text(f"ALTER TABLE {tb} SET LOGGED;"))
            # 빌딩/트리/쉼터는 중복제거용 해시도 둔다(UNION은 불필요)
            if tb != TB_UNION:
                conn.execute(text(f"ALTER TABLE {tb} ADD COLUMN IF NOT EXISTS geom_hash TEXT;"))
                conn.execute(text(f"UPDATE {tb} SET geom_hash = md5(ST_AsEWKB(geometry)) WHERE geom_hash IS NULL;"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {tb}_geom_hash_idx ON {tb}(geom_hash);"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {tb}_gix ON {tb} USING GIST(geometry);"))

def insert_append_building(engine, gdf, iou_thresh=IOU_DUP_THRESH):
    """빌딩 테이블에 append(중복 방지). 운영 테이블은 DROP/TRUNCATE 안 함."""
    if gdf.empty:
        print(f"ℹ️ No geometries to insert for {TB_BUILDING}")
        return

    STAGE = f"_stage_{TB_BUILDING}"
    gdf2 = gdf.copy()
    gdf2["geom_hash"] = gdf2["geometry"].apply(_geom_md5)

    # stage는 replace로 매번 갈아끼워도 운영에 영향 없음
    gdf2.to_postgis(
        STAGE, engine,
        if_exists="replace", index=False,
        dtype={"geometry": GA_Geom("MULTIPOLYGON", srid=4326)}
    )

    with engine.begin() as conn:
        # 해시 보장
        conn.execute(text(f"UPDATE {STAGE} SET geom_hash = md5(ST_AsEWKB(geometry)) WHERE geom_hash IS NULL;"))

        # IoU 기반 중복 제거(운영 테이블과 겹치면 버림)
        conn.execute(text(f"DROP TABLE IF EXISTS _keep_{TB_BUILDING};"))
        conn.execute(text(f"""
            CREATE TEMP TABLE _keep_{TB_BUILDING} AS
            WITH cand AS (SELECT * FROM {STAGE}),
                 ex   AS (SELECT geometry FROM {TB_BUILDING})
            SELECT c.*
            FROM cand c
            LEFT JOIN LATERAL (
               SELECT
                 ST_Area(ST_Intersection(c.geometry, e.geometry)) /
                 NULLIF(ST_Area(ST_Union(c.geometry, e.geometry)), 0) AS iou
               FROM ex e
               WHERE ST_DWithin(c.geometry, e.geometry, 0.00010) -- ~10m
                 AND ST_Intersects(c.geometry, e.geometry)
               ORDER BY iou DESC
               LIMIT 1
            ) m ON TRUE
            WHERE (m.iou IS NULL OR m.iou < {iou_thresh});
        """))
        conn.execute(text(f"TRUNCATE {STAGE};"))
        conn.execute(text(f"INSERT INTO {STAGE} SELECT * FROM _keep_{TB_BUILDING};"))

        # 운영 테이블에 UPSERT(해시 기준)
        conn.execute(text(f"""
            INSERT INTO {TB_BUILDING}(geometry, geom_hash)
            SELECT s.geometry, s.geom_hash
            FROM {STAGE} s
            LEFT JOIN {TB_BUILDING} t ON t.geom_hash = s.geom_hash
            WHERE t.geom_hash IS NULL;
        """))

        added = conn.execute(text(f"""
            SELECT COUNT(*) FROM {STAGE} s
            LEFT JOIN {TB_BUILDING} t ON t.geom_hash = s.geom_hash
            WHERE t.geom_hash IS NULL;
        """)).scalar()
        total = conn.execute(text(f"SELECT COUNT(*) FROM {TB_BUILDING};")).scalar()
        print(f"✅ {TB_BUILDING}: 새로 추가 {added}개 / 총 {total}개")

def rebuild_union_from_components(engine):
    """
    UNION 테이블은 DROP 안 하고,
    임시 테이블에 새 결과를 만든 후 같은 트랜잭션에서 TRUNCATE→INSERT.
    인덱스/권한 유지.
    """
    with engine.begin() as conn:
        # 임시 테이블에 먼저 빌드
        tmp = f"_tmp_{TB_UNION}"
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp};"))
        conn.execute(text(f"CREATE TEMP TABLE {tmp}(geometry geometry(MULTIPOLYGON,4326));"))

        conn.execute(text(f"""
            INSERT INTO {tmp}(geometry)
            WITH all_shadows AS (
              SELECT ST_CollectionExtract(
                       ST_Buffer(
                         ST_MakeValid(ST_SnapToGrid(ST_Transform(geometry, 5179), 0.05)), 0
                       ), 3
                     ) AS g5179
              FROM {TB_BUILDING}
              UNION ALL
              SELECT ST_CollectionExtract(
                       ST_Buffer(
                         ST_MakeValid(ST_SnapToGrid(ST_Transform(geometry, 5179), 0.05)), 0
                       ), 3
                     )
              FROM {TB_TREE}
              UNION ALL
              SELECT ST_CollectionExtract(
                       ST_Buffer(
                         ST_MakeValid(ST_SnapToGrid(ST_Transform(geometry, 5179), 0.05)), 0
                       ), 3
                     )
              FROM {TB_SHELTER}
            ),
            nonempty AS (
              SELECT g5179
              FROM all_shadows
              WHERE g5179 IS NOT NULL AND NOT ST_IsEmpty(g5179)
            ),
            u AS (
              SELECT ST_UnaryUnion(ST_Collect(g5179)) AS g
              FROM nonempty
            )
            SELECT ST_Transform(g, 4326) AS geometry
            FROM u;
        """))

        # 운영 UNION 내용만 교체(객체는 유지)
        conn.execute(text(f"TRUNCATE {TB_UNION};"))
        conn.execute(text(f"INSERT INTO {TB_UNION}(geometry) SELECT geometry FROM {tmp};"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {TB_UNION}_gix ON {TB_UNION} USING GIST(geometry);"))
        conn.execute(text(f"ANALYZE {TB_UNION};"))

        total = conn.execute(text(f"SELECT COUNT(*) FROM {TB_UNION};")).scalar()
        print(f"🔄 {TB_UNION} 재빌드 완료 / 총 {total}개")


# ───────── 메인 ─────────
def main():
    global STAMP, TB_BUILDING, TB_TREE, TB_SHELTER, TB_UNION
    print(f"▶ START vworld_db_update (STAMP={STAMP})")

    # 날짜만 주어지면 08:00~19:00 일괄, 아니면 단일 시각
    if _is_date_only(STAMP):
        stamps = _stamps_for_date(STAMP, 8, 19)
    elif _is_full_stamp(STAMP):
        stamps = [STAMP]
    else:
        raise SystemExit("❌ STAMP 형식은 'YYYYMMDD' 또는 'YYYYMMDD_HHMM' 이어야 합니다.")

    for st in stamps:
        # 각 시각별 테이블명/타임스탬프 갱신
        STAMP = st
        TB_BUILDING = f"shadow_building_{STAMP}"
        TB_TREE     = f"shadow_tree_{STAMP}"
        TB_SHELTER  = f"shadow_shelter_{STAMP}"
        TB_UNION    = f"shadow_union_{STAMP}"
        print(f"\n▶▶ Processing STAMP={STAMP}")

        # 1) VWorld에서 footprint 수집
        gdf_b = fetch_vworld_bldg(MINX, MINY, MAXX, MAXY)
        if gdf_b.empty:
            print("❌ VWorld 건물 없음 → 이 시각은 건너뜀")
            continue  # 배치에서는 다음 시각으로

        # 2) 태양 각도(분석 시각은 STAMP에 맞춰 계산)
        dt_local = parse_stamp_to_dt(STAMP)
        lat_c = (MINY + MAXY)/2.0
        lon_c = (MINX + MAXX)/2.0
        alt, az = compute_sun(dt_local, lat_c, lon_c)
        if alt <= 0:
            print("❌ 해당 시각에 태양고도 ≤ 0 → 이 시각 건너뜀")
            continue  # 배치에서는 다음 시각으로

        # 3) 그림자 생성
        gdf_shadow = footprints_to_shadows(gdf_b, alt, az)
        if gdf_shadow.empty:
            print("❌ 그림자 생성 0개 → 이 시각 건너뜀")
            continue  # 배치에서는 다음 시각으로

        # 4) DB 반영
        engine = create_engine(PG_URL, pool_pre_ping=True)
        ensure_tables(engine)

        # 4-1) 빌딩 그림자 append(+중복제거)
        insert_append_building(engine, gdf_shadow, iou_thresh=IOU_DUP_THRESH)

        # 4-2) UNION은 트랜잭션 내에서 임시테이블→TRUNCATE→INSERT로 재빌드
        rebuild_union_from_components(engine)

        print(f"✅ done for STAMP={STAMP}")

    print("\n✅ all done.")


if __name__ == "__main__":
    main()

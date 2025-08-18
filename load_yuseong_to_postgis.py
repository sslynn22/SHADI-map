# load_yuseong_to_postgis.py
import osmnx as ox
import geopandas as gpd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from urllib.parse import quote_plus
from geoalchemy2 import Geometry
from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely.ops import linemerge

# ───────── 사용자 설정 ─────────
PLACE = "Yuseong-gu, Daejeon, South Korea"

DB_USER = "postgres"
DB_PASS = "0202"          # 특수문자 포함 가능 → quote_plus 로 인코딩
DB_HOST = "127.0.0.1"     # localhost 대신 IP 권장
DB_PORT = 5432
DB_NAME = "shadi"

# ───────── 유틸: 어떤 경우든 LINESTRING으로 맞추기 ─────────
def to_linestring(g):
    if g is None:
        return None
    if isinstance(g, LineString):
        return g
    if isinstance(g, MultiLineString):
        try:
            m = linemerge(g)
            if isinstance(m, MultiLineString):
                # 여러 선이 남으면 가장 긴 선만 사용
                return max(list(m.geoms), key=lambda x: x.length)
            return m
        except Exception:
            return max(list(g.geoms), key=lambda x: x.length)
    if isinstance(g, GeometryCollection):
        lines = [geom for geom in g.geoms if isinstance(geom, (LineString, MultiLineString))]
        if not lines:
            return None
        merged = linemerge(lines)
        if isinstance(merged, MultiLineString):
            return max(list(merged.geoms), key=lambda x: x.length)
        return merged
    return g

def main():
    # 0) 엔진 만들기
    print("▶ Connecting to PostGIS…")
    safe_url = URL.create(
        "postgresql+psycopg2",
        username=DB_USER,
        password=quote_plus(DB_PASS),
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        query={"client_encoding": "utf8"}
    )
    engine = create_engine(safe_url, pool_pre_ping=True, future=True)

    # 1) 네트워크 다운로드
    print("▶ Downloading walking network for:", PLACE)
    G = ox.graph_from_place(PLACE, network_type="walk", simplify=True)

    # 2) GDF 변환
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    edges_gdf = edges_gdf.reset_index()  # u, v, key 등 풀려있게
    edges_gdf = edges_gdf.to_crs(epsg=4326)
    nodes_gdf = nodes_gdf.to_crs(epsg=4326)

    # 3) geometry 보정 + 길이 계산
    print("▶ Coercing edge geometries to LINESTRING...")
    edges_gdf["geometry"] = edges_gdf["geometry"].apply(to_linestring)
    edges_gdf = edges_gdf[edges_gdf["geometry"].notnull()].copy()
    edges_gdf["len_m"] = edges_gdf.geometry.to_crs(5179).length

    # 4) 컬럼명/지오메트리 지정
    edges_gdf = edges_gdf.rename(columns={"geometry": "geom"})
    edges_gdf = gpd.GeoDataFrame(edges_gdf, geometry="geom", crs="EPSG:4326")
    nodes_gdf = nodes_gdf.rename(columns={"geometry": "geom"})
    nodes_gdf = gpd.GeoDataFrame(nodes_gdf, geometry="geom", crs="EPSG:4326")

    # 5) DB 저장
    print("▶ Writing edges (ways_raw) to PostGIS…")
    edges_gdf.to_postgis(
        "ways_raw",
        engine,
        if_exists="replace",
        index=False,
        dtype={"geom": Geometry("LINESTRING", 4326)}
    )

    print("▶ Writing nodes (nodes_raw) to PostGIS…")
    nodes_gdf.to_postgis(
        "nodes_raw",
        engine,
        if_exists="replace",
        index=False,
        dtype={"geom": Geometry("POINT", 4326)}
    )

    # 6) pgRouting 토폴로지 / 인덱스 / 코스트 자동화
    print("▶ Post-processing with pgRouting topology…")
    with engine.begin() as conn:
        # 확장 보장
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgrouting;"))

        # geometry 타입 합치기(혹시 모를 혼합 대비)
        conn.execute(text("""
            ALTER TABLE ways_raw
              ALTER COLUMN geom TYPE geometry(LineString,4326)
              USING ST_LineMerge(ST_CollectionExtract(geom,2));
        """))

        # PK(id) 보장
        conn.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid='ways_raw'::regclass AND attname='id') THEN
            ALTER TABLE ways_raw ADD COLUMN id bigserial;
          END IF;
          IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='ways_raw_pkey') THEN
            ALTER TABLE ways_raw ADD CONSTRAINT ways_raw_pkey PRIMARY KEY (id);
          END IF;
        END$$;
        """))

        # source/target 컬럼 보장 (없으면 추가)
        conn.execute(text("""
            DO $$
            BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid='ways_raw'::regclass AND attname='source') THEN
                ALTER TABLE ways_raw ADD COLUMN source bigint;
              END IF;
              IF NOT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid='ways_raw'::regclass AND attname='target') THEN
                ALTER TABLE ways_raw ADD COLUMN target bigint;
              END IF;
            END$$;
        """))

        # pgr_createTopology: source/target 채우기 + vertices 테이블 생성
        # ※ v3.8에서 deprecated 경고는 무시해도 됩니다.
        conn.execute(text("""
            SELECT pgr_createTopology(
              'ways_raw',              -- edge table
              0.00001,                 -- tolerance (약 1cm 정도)
              'geom', 'id',            -- geom, id
              'source','target',       -- source/target 컬럼
              rows_where := 'true',
              clean := true            -- 깨끗하게 생성
            );
        """))

        # 인덱스
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ways_raw_geom_gix   ON ways_raw USING GIST (geom);
            CREATE INDEX IF NOT EXISTS ways_raw_source_idx ON ways_raw(source);
            CREATE INDEX IF NOT EXISTS ways_raw_target_idx ON ways_raw(target);
        """))

        # cost 컬럼(코드에서는 len_m 사용하지만 가끔 pgr 함수에 대비)
        conn.execute(text("""
            ALTER TABLE ways_raw ADD COLUMN IF NOT EXISTS cost double precision;
            ALTER TABLE ways_raw ADD COLUMN IF NOT EXISTS reverse_cost double precision;
            UPDATE ways_raw SET cost = len_m, reverse_cost = len_m;
        """))

        # 간단 검증 로그
        res1 = conn.execute(text("SELECT COUNT(*) FROM public.ways_raw_vertices_pgr;")).scalar()
        res2 = conn.execute(text("SELECT COUNT(*) FROM ways_raw WHERE source IS NOT NULL AND target IS NOT NULL;")).scalar()
        print(f"▶ vertices: {res1}, edges with s/t: {res2}")

    print("✅ saved: ways_raw, nodes_raw + pgRouting topology ready")

if __name__ == "__main__":
    main()

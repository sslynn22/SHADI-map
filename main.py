# main.py
import os, re, json, datetime, math
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import csv  # [ADD] 무더위쉼터 CSV 읽기
from ai_chat.blueprint import make_chat_blueprint

# ── api_keys.env 로부터 환경변수 로드  ──
ENV_FILE = os.getenv("API_KEYS_ENV_PATH", "api_keys.env")
def _load_env_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except FileNotFoundError:
        pass
_load_env_file(ENV_FILE)

app = Flask(__name__)
CORS(app)
app.register_blueprint(make_chat_blueprint(), url_prefix="/chat")  # <-- "/chat" 로 고정

# ───────────────────────────────────────────────────────────────────────

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
PG_URL   = os.getenv("PG_URL",  "postgresql://postgres:0202@localhost:5432/shadi")

DEFAULT_STAMP = "20250818_1400"
DEFAULT_COOL_WEIGHT = float(os.getenv("COOL_WEIGHT", "0.8"))
KAKAO_JS_KEY = os.getenv("KAKAO_JS_KEY", "")

HOT_SHELTER_CSV = os.getenv("HOT_SHELTER_CSV", "data/유성구_무더위쉼터_20250817.csv")  # [ADD]

def _parse_coord_pair(s: str):
    if not s: return None
    m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$', s.strip())
    if not m: return None
    a, b = float(m.group(1)), float(m.group(2))
    if abs(a) <= 90 and abs(b) <= 180 and abs(a) < abs(b): lat, lon = a, b
    else: lon, lat = a, b
    return (lon, lat)

def _stamp_from_time(t: str | None) -> str:
    if not t: return DEFAULT_STAMP
    s = t.strip().replace("T", " ").replace("/", "-")
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").strftime("%Y%m%d_%H%M")
    except Exception:
        pass
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s*(오전|오후)\s*(\d{1,2}):(\d{2})$", s)
    if m:
        date, ap, hh, mm = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if ap == "오후" and hh < 12: hh += 12
        if ap == "오전" and hh == 12: hh = 0
        return f"{date.replace('-','')}_{hh:02d}{mm:02d}"
    try:
        return datetime.datetime.fromisoformat(s).strftime("%Y%m%d_%H%M")
    except Exception:
        return DEFAULT_STAMP

def _validate_table(prefix: str, stamp: str) -> str:
    if prefix not in {"shadow_union","shadow_building","shadow_tree","shadow_shelter"}:
        prefix = "shadow_union"
    if not re.fullmatch(r"\d{8}_\d{4}", stamp):
        stamp = DEFAULT_STAMP
    return f"{prefix}_{stamp}"

# ───────────────────────── 기존 경로(최단/시원) ─────────────────────────
def _fetch_routes(conn_dsn: str, src: tuple, dst: tuple, union_table: str, cool_weight: float):
    union_table = _validate_table("shadow_union", union_table.split("_", 2)[-1])
    conninfo = conn_dsn + "?application_name=shadi_route_srv"
    with psycopg2.connect(conninfo) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SET LOCAL statement_timeout = '60s';
            SET LOCAL idle_in_transaction_session_timeout = '30s';
            SET LOCAL jit = OFF;
            SET LOCAL work_mem = '256MB';
        """)
        q_edges = f"""
DROP TABLE IF EXISTS edges_tmp;
CREATE TEMP TABLE edges_tmp AS
WITH
params AS (
  SELECT 4.5::float8 AS shade_tol_m,
         120::float8 AS near_m,
         250::float8 AS corridor_m
),
u5179 AS (
  SELECT ST_Subdivide(
           ST_UnaryUnion(ST_MakeValid(ST_Transform(geometry, 5179))), 256
         ) AS g5179
  FROM {union_table}
),
srcpt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g4326),
dstpt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g4326),
corridor AS (
  SELECT ST_Buffer(
           ST_Transform(
             ST_MakeLine((SELECT g4326 FROM srcpt),(SELECT g4326 FROM dstpt)),
             5179
           ),
           (SELECT corridor_m FROM params)
         ) AS g5179
),
u_clip AS (
  SELECT u.g5179
  FROM u5179 u
  JOIN corridor c ON ST_Intersects(u.g5179, c.g5179)
),
edges AS (
  SELECT
    w.id, w.source, w.target, w.geom, w.len_m,
    LEAST(
      COALESCE(
        SUM(
          ST_Length(
            ST_Intersection(
              ST_SnapToGrid(ST_Transform(w.geom, 5179), 0.05),
              ST_Buffer(u.g5179, (SELECT shade_tol_m FROM params))
            )
          )
        ) / NULLIF(
          ST_Length(ST_SnapToGrid(ST_Transform(w.geom, 5179), 0.05)), 0
        ),
        0
      ),
      1.0
    ) AS shade_ratio
  FROM ways_raw w
  JOIN corridor c ON ST_Intersects(ST_Transform(w.geom, 5179), c.g5179)
  LEFT JOIN u_clip u ON ST_DWithin(ST_Transform(w.geom, 5179), u.g5179, (SELECT near_m FROM params))
  GROUP BY w.id, w.source, w.target, w.geom, w.len_m
)
SELECT id, source, target, geom, len_m, shade_ratio
FROM edges;
"""
        cur.execute(q_edges, (src[0], src[1], dst[0], dst[1]))
        cur.execute("CREATE INDEX IF NOT EXISTS edges_tmp_id_idx ON edges_tmp(id);")
        cur.execute("CREATE INDEX IF NOT EXISTS edges_tmp_st_idx ON edges_tmp(source, target);")
        sql_route = f"""
WITH
ok_v AS (
  SELECT v.id, v.the_geom
  FROM ways_raw_vertices_pgr v
  JOIN (SELECT source AS vid FROM ways_raw UNION SELECT target AS vid FROM ways_raw) ok ON ok.vid = v.id
),
src AS (SELECT id FROM ok_v ORDER BY the_geom <-> ST_SetSRID(ST_Point(%s,%s), 4326) LIMIT 1),
dst AS (SELECT id FROM ok_v ORDER BY the_geom <-> ST_SetSRID(ST_Point(%s,%s), 4326) LIMIT 1),
shortest AS (
  SELECT * FROM pgr_dijkstra(
    $$SELECT id, source, target, len_m AS cost, len_m AS reverse_cost FROM edges_tmp$$,
    (SELECT id FROM src), (SELECT id FROM dst), false
  )
),
coolest AS (
  SELECT * FROM pgr_dijkstra(
    $$SELECT id, source, target,
             GREATEST(len_m * (1 - {cool_weight} * shade_ratio), 0.1) AS cost,
             GREATEST(len_m * (1 - {cool_weight} * shade_ratio), 0.1) AS reverse_cost
      FROM edges_tmp$$,
    (SELECT id FROM src), (SELECT id FROM dst), false
  )
),
shortest_path AS (
  SELECT ST_LineMerge(ST_Union(e.geom)) AS geom, SUM(e.len_m) AS total_m, AVG(e.shade_ratio) AS avg_shade_ratio
  FROM shortest s JOIN edges_tmp e ON s.edge = e.id WHERE s.edge <> -1
),
coolest_path AS (
  SELECT ST_LineMerge(ST_Union(e.geom)) AS geom, SUM(e.len_m) AS total_m, AVG(e.shade_ratio) AS avg_shade_ratio
  FROM coolest c  JOIN edges_tmp e ON c.edge = e.id WHERE c.edge <> -1
)
SELECT 'shortest' AS kind, ST_AsGeoJSON(geom) AS gj, total_m, avg_shade_ratio FROM shortest_path
UNION ALL
SELECT 'coolest'  AS kind, ST_AsGeoJSON(geom) AS gj, total_m, avg_shade_ratio FROM coolest_path;
"""
        cur.execute(sql_route, (src[0], src[1], dst[0], dst[1]))
        rows = cur.fetchall()
    out = {}
    for r in rows:
        gj = json.loads(r["gj"]) if r["gj"] else None
        out[r["kind"]] = {
            "gj": gj,
            "total_m": float(r["total_m"]) if r["total_m"] is not None else None,
            "avg_shade_ratio": float(r["avg_shade_ratio"]) if r["avg_shade_ratio"] is not None else None
        }
    return out

# ───────────────────── 쉼터 우선 경로(추가) ─────────────────────
def _read_hot_shelters():
    rows = []
    if not os.path.exists(HOT_SHELTER_CSV):
        return rows
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(HOT_SHELTER_CSV, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        lat = float(str(r.get("위도", "")).strip())
                        lng = float(str(r.get("경도", "")).strip())
                        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                            continue
                        rows.append({"lat":lat, "lng":lng})
                    except Exception:
                        continue
            break
        except Exception:
            continue
    return rows

def _fetch_route_via_shelter(conn_dsn: str, src: tuple, dst: tuple, union_table: str, max_candidates:int=6):
    """
    src,dst: (lon,lat)
    '쉼터 경유' 최단 경로. 후보 쉼터 몇 개를 골라 출발→쉼터 + 쉼터→도착 두 번의 dijkstra 합산이 가장 짧은 경로를 반환.
    """
    union_table = _validate_table("shadow_union", union_table.split("_", 2)[-1])
    shelters = _read_hot_shelters()
    if not shelters:
        return None

    # 간단 후보 추리: src/dst 경계 + 여유(약 0.02° ≈ 2km)
    minx = min(src[0], dst[0]); maxx = max(src[0], dst[0])
    miny = min(src[1], dst[1]); maxy = max(src[1], dst[1])
    padx = max(0.02, abs(maxx-minx)*0.3)
    pady = max(0.02, abs(maxy-miny)*0.3)
    cand = [s for s in shelters if (minx-padx)<=s["lng"]<=(maxx+padx) and (miny-pady)<=s["lat"]<=(maxy+pady)]
    if not cand:
        # 없으면 전체에서 src-dst 중점에 가까운 순 상위
        mid_lon = (src[0]+dst[0])/2; mid_lat = (src[1]+dst[1])/2
        cand = sorted(shelters, key=lambda s:(s["lng"]-mid_lon)**2+(s["lat"]-mid_lat)**2)

    cand = cand[:max_candidates]

    best = None
    conninfo = conn_dsn + "?application_name=shadi_route_via_shelter"
    with psycopg2.connect(conninfo) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SET LOCAL statement_timeout = '60s';
            SET LOCAL idle_in_transaction_session_timeout = '30s';
            SET LOCAL jit = OFF;
            SET LOCAL work_mem = '256MB';
        """)
        # 반복: 후보 쉼터 하나씩 평가
        for s in cand:
            via = (s["lng"], s["lat"])
            # 1) 두 구간의 코리도어를 합친 edges_tmp_via 생성
            q_edges_via = f"""
DROP TABLE IF EXISTS edges_tmp_via;
CREATE TEMP TABLE edges_tmp_via AS
WITH
params AS (
  SELECT 4.5::float8 AS shade_tol_m,
         120::float8 AS near_m,
         250::float8 AS corridor_m
),
u5179 AS (
  SELECT ST_Subdivide(
           ST_UnaryUnion(ST_MakeValid(ST_Transform(geometry, 5179))), 256
         ) AS g5179
  FROM {union_table}
),
srcpt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g4326),
viapt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g4326),
dstpt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g4326),
buf1  AS (
  SELECT ST_Buffer(ST_Transform(ST_MakeLine((SELECT g4326 FROM srcpt),(SELECT g4326 FROM viapt)),5179),
                   (SELECT corridor_m FROM params)) AS g5179
),
buf2  AS (
  SELECT ST_Buffer(ST_Transform(ST_MakeLine((SELECT g4326 FROM viapt),(SELECT g4326 FROM dstpt)),5179),
                   (SELECT corridor_m FROM params)) AS g5179
),
corridor AS (
  SELECT ST_Union((SELECT g5179 FROM buf1),(SELECT g5179 FROM buf2)) AS g5179
),
u_clip AS (
  SELECT u.g5179
  FROM u5179 u
  JOIN corridor c ON ST_Intersects(u.g5179, c.g5179)
),
edges AS (
  SELECT
    w.id, w.source, w.target, w.geom, w.len_m,
    LEAST(
      COALESCE(
        SUM(
          ST_Length(
            ST_Intersection(
              ST_SnapToGrid(ST_Transform(w.geom, 5179), 0.05),
              ST_Buffer(u.g5179, (SELECT shade_tol_m FROM params))
            )
          )
        ) / NULLIF(
          ST_Length(ST_SnapToGrid(ST_Transform(w.geom, 5179), 0.05)), 0
        ),
        0
      ),
      1.0
    ) AS shade_ratio
  FROM ways_raw w
  JOIN corridor c ON ST_Intersects(ST_Transform(w.geom, 5179), c.g5179)
  LEFT JOIN u_clip u ON ST_DWithin(ST_Transform(w.geom, 5179), u.g5179, (SELECT near_m FROM params))
  GROUP BY w.id, w.source, w.target, w.geom, w.len_m
)
SELECT id, source, target, geom, len_m, shade_ratio FROM edges;
"""
            cur.execute(q_edges_via, (src[0], src[1], via[0], via[1], dst[0], dst[1]))
            cur.execute("CREATE INDEX IF NOT EXISTS edges_tmp_via_id_idx ON edges_tmp_via(id);")
            cur.execute("CREATE INDEX IF NOT EXISTS edges_tmp_via_st_idx ON edges_tmp_via(source, target);")

            # 2) 출발→쉼터, 쉼터→도착 두 구간 최단으로 연결
            q_route_via = """
WITH
ok_v AS (
  SELECT v.id, v.the_geom
  FROM ways_raw_vertices_pgr v
  JOIN (SELECT source AS vid FROM ways_raw UNION SELECT target AS vid FROM ways_raw) ok ON ok.vid = v.id
),
srcpt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g),
viapt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g),
dstpt AS (SELECT ST_SetSRID(ST_Point(%s,%s), 4326) AS g),
src AS (SELECT id FROM ok_v ORDER BY the_geom <-> (SELECT g FROM srcpt) LIMIT 1),
via AS (SELECT id FROM ok_v ORDER BY the_geom <-> (SELECT g FROM viapt) LIMIT 1),
dst AS (SELECT id FROM ok_v ORDER BY the_geom <-> (SELECT g FROM dstpt) LIMIT 1),

leg1 AS (
  SELECT * FROM pgr_dijkstra(
    $$SELECT id, source, target, len_m AS cost, len_m AS reverse_cost FROM edges_tmp_via$$,
    (SELECT id FROM src), (SELECT id FROM via), false
  )
),
leg2 AS (
  SELECT * FROM pgr_dijkstra(
    $$SELECT id, source, target, len_m AS cost, len_m AS reverse_cost FROM edges_tmp_via$$,
    (SELECT id FROM via), (SELECT id FROM dst), false
  )
),
p1 AS (
  SELECT ST_LineMerge(ST_Union(e.geom)) AS geom, SUM(e.len_m) AS total_m, AVG(e.shade_ratio) AS avg_shade_ratio
  FROM leg1 s JOIN edges_tmp_via e ON s.edge = e.id WHERE s.edge <> -1
),
p2 AS (
  SELECT ST_LineMerge(ST_Union(e.geom)) AS geom, SUM(e.len_m) AS total_m, AVG(e.shade_ratio) AS avg_shade_ratio
  FROM leg2 s JOIN edges_tmp_via e ON s.edge = e.id WHERE s.edge <> -1
)
SELECT
  ST_AsGeoJSON(ST_LineMerge(ST_Union(ARRAY[(SELECT geom FROM p1),(SELECT geom FROM p2)]))) AS gj,
  COALESCE((SELECT total_m FROM p1),0) + COALESCE((SELECT total_m FROM p2),0) AS total_m,
  CASE
    WHEN (COALESCE((SELECT total_m FROM p1),0) + COALESCE((SELECT total_m FROM p2),0)) > 0
    THEN (
      COALESCE((SELECT avg_shade_ratio FROM p1),0) * COALESCE((SELECT total_m FROM p1),0)
      + COALESCE((SELECT avg_shade_ratio FROM p2),0) * COALESCE((SELECT total_m FROM p2),0)
    ) / NULLIF(COALESCE((SELECT total_m FROM p1),0) + COALESCE((SELECT total_m FROM p2),0), 0)
    ELSE NULL
  END AS avg_shade_ratio;
"""
            cur.execute(q_route_via, (src[0], src[1], via[0], via[1], dst[0], dst[1]))
            row = cur.fetchone()
            gj = json.loads(row["gj"]) if row and row["gj"] else None
            total_m = float(row["total_m"]) if row and row["total_m"] is not None else None
            avg_shade = float(row["avg_shade_ratio"]) if row and row["avg_shade_ratio"] is not None else None

            if gj and total_m:
                cand_out = {"gj": gj, "total_m": total_m, "avg_shade_ratio": avg_shade, "via": {"lon": via[0], "lat": via[1]}}
                if (best is None) or (total_m < best["total_m"]):
                    best = cand_out

    return best

# ───────────────────────── 그림자 GeoJSON ─────────────────────────
def _fetch_shadow_any(conn_dsn: str, table_prefix: str, stamp: str, bbox: tuple|None, simplify_tol_m: float = 0.7):
    table = _validate_table(table_prefix, stamp)
    conninfo = conn_dsn + "?application_name=shadi_shadow_srv"
    with psycopg2.connect(conninfo) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SET LOCAL statement_timeout = '60s';
            SET LOCAL jit = OFF;
            SET LOCAL work_mem = '256MB';
        """)
        if bbox:
            minx, miny, maxx, maxy = bbox
            q = f"""
WITH bb AS (SELECT ST_MakeEnvelope(%s,%s,%s,%s,4326) AS env4326),
     src AS (SELECT ST_MakeValid(geometry) AS g4326 FROM {table}),
     clip AS (
       SELECT ST_Intersection(s.g4326, bb.env4326) AS g4326
       FROM src s, bb WHERE ST_Intersects(s.g4326, bb.env4326)
     ),
     fix AS (
       SELECT ST_CollectionExtract(ST_Buffer(g4326,0),3) AS g4326
       FROM clip WHERE g4326 IS NOT NULL AND NOT ST_IsEmpty(g4326)
     ),
     simp AS (
       SELECT ST_Buffer(
                ST_SimplifyPreserveTopology(ST_SnapToGrid(ST_Transform(g4326,5179),0.05), %s), 0
              ) AS g5179
       FROM fix
     ),
     u AS (SELECT ST_Buffer(ST_UnaryUnion(ST_Collect(g5179)),0) AS g5179 FROM simp)
SELECT ST_AsGeoJSON(ST_Transform(u.g5179,4326)) AS gj, (SELECT COUNT(*) FROM simp) AS cnt FROM u;
"""
            try:
                cur.execute(q, (minx, miny, maxx, maxy, simplify_tol_m))
            except Exception:
                conn.rollback()
                cur.execute("SET LOCAL statement_timeout='60s'; SET LOCAL jit=OFF; SET LOCAL work_mem='256MB';")
                q_fb = f"""
WITH bb AS (SELECT ST_MakeEnvelope(%s,%s,%s,%s,4326) AS env4326),
     src AS (SELECT ST_MakeValid(geometry) AS g4326 FROM {table}),
     clip AS (
       SELECT ST_Intersection(s.g4326, bb.env4326) AS g4326
       FROM src s, bb WHERE ST_Intersects(s.g4326, bb.env4326)
     ),
     fix AS (
       SELECT ST_CollectionExtract(ST_Buffer(g4326,0),3) AS g4326
       FROM clip WHERE g4326 IS NOT NULL AND NOT ST_IsEmpty(g4326)
     ),
     simp AS (
       SELECT ST_Buffer(
                ST_SimplifyPreserveTopology(ST_SnapToGrid(ST_Transform(g4326,5179),0.05), %s), 0
              ) AS g5179
       FROM fix
     )
SELECT ST_AsGeoJSON(ST_Transform(ST_Collect(g5179),4326)) AS gj, COUNT(*) AS cnt FROM simp;
"""
                cur.execute(q_fb, (minx, miny, maxx, maxy, simplify_tol_m))
        else:
            q = f"""
WITH src AS (SELECT ST_MakeValid(geometry) AS g4326 FROM {table}),
     fix AS (
       SELECT ST_CollectionExtract(ST_Buffer(g4326,0),3) AS g4326
       FROM src WHERE g4326 IS NOT NULL AND NOT ST_IsEmpty(g4326)
     ),
     simp AS (
       SELECT ST_Buffer(
                ST_SimplifyPreserveTopology(ST_SnapToGrid(ST_Transform(g4326,5179),0.05), %s), 0
              ) AS g5179
       FROM fix
     ),
     u AS (SELECT ST_Buffer(ST_UnaryUnion(ST_Collect(g5179)),0) AS g5179 FROM simp)
SELECT ST_AsGeoJSON(ST_Transform(u.g5179,4326)) AS gj, (SELECT COUNT(*) FROM simp) AS cnt;
"""
            try:
                cur.execute(q, (simplify_tol_m,))
            except Exception:
                conn.rollback()
                cur.execute("SET LOCAL statement_timeout='60s'; SET LOCAL jit=OFF; SET LOCAL work_mem='256MB';")
                q_fb = f"""
WITH src AS (SELECT ST_MakeValid(geometry) AS g4326 FROM {table}),
     fix AS (
       SELECT ST_CollectionExtract(ST_Buffer(g4326,0),3) AS g4326
       FROM src WHERE g4326 IS NOT NULL AND NOT ST_IsEmpty(g4326)
     ),
     simp AS (
       SELECT ST_Buffer(
                ST_SimplifyPreserveTopology(ST_SnapToGrid(ST_Transform(g4326,5179),0.05), %s), 0
              ) AS g5179
       FROM fix
     )
SELECT ST_AsGeoJSON(ST_Transform(ST_Collect(g5179),4326)) AS gj, COUNT(*) AS cnt FROM simp;
"""
                cur.execute(q_fb, (simplify_tol_m,))
        row = cur.fetchone()
        gj = json.loads(row["gj"]) if row and row["gj"] else None
        cnt = int(row["cnt"]) if row and row["cnt"] is not None else 0
        return {"gj": gj, "count": cnt, "table": table}


MAP_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>그늘길 라우팅 데모</title>
  <!-- Kakao SDK: services 라이브러리 포함 -->
  <script defer src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={{ kakao_js_key }}&autoload=false&libraries=services"></script>
  <style>
    html, body, #map { height: 100%; margin: 0; }
    #panel {
      position:absolute; top:10px; right:10px; z-index:1000;
      background:rgba(255,255,255,.96); padding:12px; border-radius:12px;
      box-shadow:0 2px 12px rgba(0,0,0,.15); width:360px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, 'Noto Sans KR', Arial, sans-serif;
    }
    #panel .row{display:flex; gap:6px; margin-top:4px; align-items:center}
    #panel input[type=text], #panel input[type=datetime-local] { width:100%; padding:6px 8px; margin:4px 0 8px; border:1px solid #ccc; border-radius:8px; }
    #panel button{flex:1; padding:8px; border:0; border-radius:8px; cursor:pointer}
    #panel .btn{background:#1BA952; color:#fff}
    #panel .ghost{background:#f1f3f5; color:#222}
    #shadow-stats {font-size:12px; color:#333; margin-top:6px; line-height:1.4}
    .chip{display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:middle}

    /* ==== 좌측하단 경로 패널 ==== */
    #routes-panel{
      position:absolute; left:14px; bottom:14px; z-index:1000;
      background:rgba(255,255,255,.96); padding:10px; border-radius:12px;
      box-shadow:0 2px 12px rgba(0,0,0,.15); width:340px;
      font-family: inherit;
    }
    #routes-title{ font-size:13px; font-weight:700; margin:2px 2px 8px; color:#111; }
    #routes{ display:flex; flex-direction:column; gap:8px; }

    .route-card {
      border:1px solid #e5e7eb; border-radius:10px; padding:10px;
      background:#fff; cursor:pointer; transition:box-shadow .15s, border-color .15s;
      display:flex; align-items:center; gap:12px;
    }
    .route-card:hover { box-shadow:0 2px 10px rgba(0,0,0,.08); }
    .route-card.active { border-color:#225ea8; box-shadow:0 0 0 2px rgba(34,94,168,.18) inset; }

    .route-time{
      font-weight:800; font-size:26px; color:#111; line-height:1; min-width:64px; text-align:center;
    }
    .route-time .unit{ font-size:13px; font-weight:600; color:#444; margin-left:2px; }

    .route-main { display:flex; flex-direction:column; gap:4px; flex:1; min-width:0; }
    .title-coolest  { font-weight:800; color:#46e583; font-size:14px; }
    .title-shortest { font-weight:800; color:#1BA952; font-size:14px; }
    .title-shelter  { font-weight:800; color:#03491e; font-size:14px; } /* [ADD] 쉼터우선 색상 */
    .route-meta  { display:flex; gap:10px; align-items:center; color:#444; font-size:12px; }
    .route-meta span { white-space:nowrap; }
    .muted { color:#6b7280 }
  </style>
</head>
<body>
<div id="map"><!-- Chat button & panel -->
<button id="chat-toggle" style="position:absolute; right:16px; bottom:16px; z-index:1100;
  border:0; background:none; padding:0; cursor:pointer;">
  <img src="/static/chatbot.png" alt="챗봇" style="width:120px; height:auto;">
</button>
<div id="chat-panel" style="position:absolute; right:16px; bottom:70px; z-index:1100; width:320px;
  background:rgba(255,255,255,.98); border:1px solid #e5e7eb; border-radius:12px; display:none;
  box-shadow:0 8px 24px rgba(0,0,0,.18); overflow:hidden;">
  <div style="padding:10px; font-weight:700; font-size:14px; border-bottom:1px solid #eee">SHADI 챗봇</div>
  <div id="chat-log" style="height:260px; overflow:auto; padding:10px; font-size:13px;"></div>
  <div style="display:flex; gap:6px; padding:10px; border-top:1px solid #eee">
    <input id="chat-input" type="text" placeholder="예) 출발 36.3617,127.3447 / 도착 36.3721,127.3452"
           style="flex:1; border:1px solid #ddd; border-radius:8px; padding:8px 10px;">
    <button id="chat-send" class="ghost" style="padding:8px 10px; border-radius:8px;">보내기</button>
  </div>
</div>
</div>

<!-- 우측 상단 입력 패널 -->
<div id="panel">
  <label>출발 (lat, lon)</label>
  <input id="src" type="text" placeholder="예: 36.361738, 127.344776">
  <label>도착 (lat, lon)</label>
  <input id="dst" type="text" placeholder="예: 36.372113, 127.345180">
  <label>분석시각</label>
  <input id="time" type="datetime-local" value="2025-08-18T14:00">

  <label>장소 검색</label>
  <div class="row">
    <input id="kw" type="text" placeholder="예: 충남대 정문, 스타벅스" style="flex:1">
    <button class="ghost" id="kw-btn" style="flex:0 0 auto">검색</button>
  </div>
  <ul id="kw-results" style="list-style:none; padding:0; margin:6px 0 8px; max-height:160px; overflow:auto; font-size:12px"></ul>

  <div class="row">
    <button class="ghost" id="pick-src">지도에서 출발 찍기</button>
    <button class="ghost" id="pick-dst">지도에서 도착 찍기</button>
  </div>
  <div class="row">
    <button class="btn" id="run">실행</button>
    <button class="ghost" id="clear">초기화</button>
  </div>

  <div style="font-size:13px; font-weight:600; margin:10px 0 4px">그림자 레이어</div>
  <div class="row" style="gap:10px; justify-content:space-between">
    <label style="display:flex; align-items:center; gap:6px; flex:1">
      <input type="checkbox" id="toggle-building" checked>
      <span><span class="chip" style="background:#28252c; border:1px solid #463f4f"></span>건물</span>
    </label>
    <label style="display:flex; align-items:center; gap:6px; flex:1">
      <input type="checkbox" id="toggle-tree" checked>
      <span><span class="chip" style="background:#7fc97f; border:1px solid #4daf4a"></span>가로수</span>
    </label>
    <label style="display:flex; align-items:center; gap:6px; flex:1">
      <input type="checkbox" id="toggle-shelter" checked>
      <span><span class="chip" style="background:#fdae61; border:1px solid #e66101"></span>쉼터</span>
    </label>
    <button class="ghost" id="refresh-shadow" style="flex:0 0 auto">갱신</button>
  </div>
  <div id="shadow-stats"></div>
  <small class="muted">Tip: 지도 클릭으로 출발/도착을 쉽게 찍을 수 있어요.</small>
</div>

<!-- 좌측 하단 경로 패널 -->
<div id="routes-panel">
  <div id="routes-title">경로 목록</div>
  <div id="routes"><div class="muted">경로를 실행하면 여기 표시됩니다.</div></div>
</div>

<script>
  window.addEventListener('DOMContentLoaded', () => {
    kakao.maps.load(() => {
      const $ = (id)=>document.getElementById(id);

      const WALK_KMH = 4.0;

      const map = new kakao.maps.Map(document.getElementById('map'), {
        center: new kakao.maps.LatLng(36.36917, 127.34515), level: 4
      });

      // ===== [ADD] 무더위쉼터 마커 =====
      const heatMarkers = [];
      function sunPinSVG(){
        return `
        <svg xmlns='http://www.w3.org/2000/svg' width='30' height='42' viewBox='0 0 30 42'>
          <path fill='#f39c12' d='M15 0C7.3 0 1 6.3 1 14c0 10.1 14 28 14 28s14-17.9 14-28C29 6.3 22.7 0 15 0z'/>
          <circle cx='15' cy='14' r='5.5' fill='#fff'/>
          <g transform='translate(15,14)' stroke='#fff' stroke-width='1.4'>
            <line x1='0' y1='-8.5' x2='0' y2='-5'/>
            <line x1='0' y1='8.5'  x2='0' y2='5'/>
            <line x1='8.5' y1='0'  x2='5' y2='0'/>
            <line x1='-8.5' y1='0' x2='-5' y2='0'/>
            <line x1='6' y1='6' x2='3.5' y2='3.5'/>
            <line x1='-6' y1='6' x2='-3.5' y2='3.5'/>
            <line x1='6' y1='-6' x2='3.5' y2='-3.5'/>
            <line x1='-6' y1='-6' x2='-3.5' y2='-3.5'/>
          </g>
        </svg>`;
      }
      const HEAT_PIN = new kakao.maps.MarkerImage(
        'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(sunPinSVG()),
        new kakao.maps.Size(30, 42),
        { offset: new kakao.maps.Point(15, 42) }
      );
      function escapeHtml(s){
        return String(s||"").replace(/[&<>"']/g, m=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
      }
      function addHeatMarker(lat, lng, name, addr, hours){
        const ll = new kakao.maps.LatLng(lat, lng);
        const mk = new kakao.maps.Marker({ position: ll, image: HEAT_PIN });
        mk.setMap(map); heatMarkers.push(mk);
        const html = `<div style="padding:8px 10px; font-size:12px; line-height:1.35; max-width:260px">
            <b>${escapeHtml(name)}</b><br/>${escapeHtml(addr)}<br/>운영 시간: ${escapeHtml(hours)}</div>`;
        const iw = new kakao.maps.InfoWindow({ content: html, removable: true });
        kakao.maps.event.addListener(mk, 'click', () => iw.open(map, mk));
      }
      async function loadHotShelters(){
        try{
          const res = await fetch('/hot-shelters');
          const list = await res.json();
          if(Array.isArray(list)){
            list.forEach(s => addHeatMarker(s.lat, s.lng, s.name || "무더위쉼터", s.addr || "", s.hours || ""));
          }
        }catch(e){ console.warn('hot-shelters load failed:', e); }
      }
      loadHotShelters();
      // ===== [ADD] 끝 =====

      // ====== 출발/도착 핀 ======
      function pinSVG(fill){
        return `<svg xmlns='http://www.w3.org/2000/svg' width='30' height='42' viewBox='0 0 30 42'>
          <path fill='${fill}' d='M15 0c-7.2 0-13 5.8-13 13 0 9.5 13 29 13 29s13-19.5 13-29C28 5.8 22.2 0 15 0z'/>
          <circle cx='15' cy='13' r='5' fill='#fff'/>
        </svg>`;
      }
      function makePin(color){
        return new kakao.maps.MarkerImage(
          'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(pinSVG(color)),
          new kakao.maps.Size(30, 42),
          { offset: new kakao.maps.Point(15, 42) }
        );
      }
      const SRC_PIN = makePin('#2ecc71');
      const DST_PIN = makePin('#e74c3c');

      // state
      let srcMarker=null, dstMarker=null, pickMode=null;
      let srcLL=null, dstLL=null;
      let shortestPolyline=null, coolestPolyline=null, shelterPolyline=null; // [ADD] shelter
      let searchMarkers=[];
      let selectedKind = null;
      const shadowBuilding=[], shadowTree=[], shadowShelter=[];
      const places = new kakao.maps.services.Places();

      // helpers
      function llstrLL(lat, lng){ return lat.toFixed(6)+", "+lng.toFixed(6); }
      function toLonLat(str){
        const m = /^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/.exec(str||"");
        if(!m) return null;
        const a=parseFloat(m[1]), b=parseFloat(m[2]);
        let lat,lon;
        if(Math.abs(a)<=90 && Math.abs(b)<=180 && Math.abs(a)<Math.abs(b)){ lat=a; lon=b; } else { lon=a; lat=b; }
        return [lon,lat];
      }
      function setSrcByLatLng(lat, lng){
        if(srcMarker) srcMarker.setMap(null);
        srcMarker = new kakao.maps.Marker({ position: new kakao.maps.LatLng(lat, lng), image: SRC_PIN });
        srcMarker.setMap(map);
        const el=$('src'); if(el) el.value = llstrLL(lat, lng);
        srcLL = {lat, lng};
      }
      function setDstByLatLng(lat, lng){
        if(dstMarker) dstMarker.setMap(null);
        dstMarker = new kakao.maps.Marker({ position: new kakao.maps.LatLng(lat, lng), image: DST_PIN });
        dstMarker.setMap(map);
        const el=$('dst'); if(el) el.value = llstrLL(lat, lng);
        dstLL = {lat, lng};
      }
      function clearPolygons(arr){ arr.forEach(p => p.setMap(null)); arr.length = 0; }
      function clearSearchResults(){
        searchMarkers.forEach(m => m.setMap(null));
        searchMarkers.length = 0;
        const ul=$('kw-results'); if(ul) ul.innerHTML="";
      }

      const fmtDist = (m)=> (m==null? '-' : (m>=1000? (m/1000).toFixed(2)+' km' : m.toFixed(0)+' m'));
      const fmtPct  = (p)=> (p==null? '-' : (p*100).toFixed(1)+'%');
      function walkMinutesFromMeters(m){ if(m==null) return null; const mpm=(WALK_KMH*1000)/60.0; return m/mpm; }
      const fmtMinInt  = (min)=> (min==null? '-' : Math.max(1, Math.round(min)));

      // ----- 버튼 바인딩 -----
      const pickSrcBtn = $('pick-src'); if(pickSrcBtn){ pickSrcBtn.onclick = function(){ pickMode='src'; this.style.opacity=1; const d=$('pick-dst'); if(d) d.style.opacity=.8; }; }
      const pickDstBtn = $('pick-dst'); if(pickDstBtn){ pickDstBtn.onclick = function(){ pickMode='dst'; this.style.opacity=1; const s=$('pick-src'); if(s) s.style.opacity=.8; }; }
      const clearBtn = $('clear');
      if(clearBtn){
        clearBtn.onclick = function(){
          if(srcMarker) srcMarker.setMap(null); if(dstMarker) dstMarker.setMap(null);
          srcMarker=dstMarker=null; srcLL=dstLL=null;
          if(shortestPolyline) shortestPolyline.setMap(null);
          if(coolestPolyline)  coolestPolyline.setMap(null);
          if(shelterPolyline)  shelterPolyline.setMap(null); // [ADD]
          shortestPolyline=coolestPolyline=shelterPolyline=null;
          selectedKind=null;
          clearPolygons(shadowBuilding); clearPolygons(shadowTree); clearPolygons(shadowShelter);
          clearSearchResults();
          const s=$('src'), d=$('dst'); if(s) s.value=""; if(d) d.value="";
          const ps=$('pick-src'), pd=$('pick-dst'); if(ps) ps.style.opacity=1; if(pd) pd.style.opacity=1;
          pickMode=null;
          const routes=$('routes'); if(routes) routes.innerHTML='<div class="muted">경로를 실행하면 여기 표시됩니다.</div>';
        };
      }

      kakao.maps.event.addListener(map, 'click', function(mouseEvent){
        if(!pickMode) return;
        const latlng = mouseEvent.latLng;
        if(pickMode==='src') setSrcByLatLng(latlng.getLat(), latlng.getLng());
        else setDstByLatLng(latlng.getLat(), latlng.getLng());
      });

      // --- 경로 실행 ---
      const runBtn = $('run');
      if(runBtn){
        runBtn.onclick = async function(){
          let s = srcLL ? [srcLL.lng, srcLL.lat] : toLonLat(($('src')||{}).value);
          let d = dstLL ? [dstLL.lng, dstLL.lat] : toLonLat(($('dst')||{}).value);
          const t = ($('time')||{}).value || "";
          if(!s || !d){ alert("좌표 형식이 올바르지 않습니다. 예: 36.361738, 127.344776"); return; }
          const qs = new URLSearchParams({ src: `${s[1]},${s[0]}`, dst: `${d[1]},${d[0]}`, time: t }).toString();
          runBtn.disabled = true; runBtn.innerText = "계산중...";
          try{
            const res = await fetch('/route?'+qs); const js = await res.json();
            if(!res.ok) throw new Error(js.error || "route API 실패");

            // 기존 선 지우기
            if(shortestPolyline) shortestPolyline.setMap(null);
            if(coolestPolyline)  coolestPolyline.setMap(null);
            if(shelterPolyline)  shelterPolyline.setMap(null);
            shortestPolyline=coolestPolyline=shelterPolyline=null;
            selectedKind=null;

            // 새 선 생성
            if(js.shortest && js.shortest.gj){
              shortestPolyline = polylineFromGeoJSON(js.shortest.gj, {strokeWeight:6, strokeColor:'#1BA952', strokeOpacity:0.95});
              shortestPolyline.setMap(map);
            }
            if(js.coolest && js.coolest.gj){
              coolestPolyline  = polylineFromGeoJSON(js.coolest.gj, {strokeWeight:6, strokeColor:'#46e583', strokeOpacity:0.95});
              coolestPolyline.setMap(map);
            }
            if(js.shelter && js.shelter.gj){
              shelterPolyline = polylineFromGeoJSON(js.shelter.gj, {strokeWeight:6, strokeColor:'#03491e', strokeOpacity:0.95}); // [ADD] 보라
              shelterPolyline.setMap(map);
            }

            // 지도 bounds
            const lines=[shortestPolyline, coolestPolyline, shelterPolyline].filter(Boolean);
            if(lines.length){
              const bounds = new kakao.maps.LatLngBounds();
              lines.forEach(pl => pl.getPath().forEach(ll => bounds.extend(ll)));
              map.setBounds(bounds, 30, 30, 30, 30);
            }

            // 경로 카드 렌더 (쉼터우선 → 시원 → 최단)
            renderRouteCards(js);
          }catch(e){ alert("오류: "+e.message); }
          finally{ runBtn.disabled=false; runBtn.innerText="실행"; }
        };
      }

      // 경로 카드 렌더링 + 동작
      function renderRouteCards(data){
        const routesEl = $('routes'); if(!routesEl) return;
        const items = [];
        function build(kind, titleClass, titleText, obj){
          if(!obj || !obj.gj || obj.total_m==null) return;
          const m = obj.total_m;
          const min = walkMinutesFromMeters(m);
          const minInt = fmtMinInt(min);
          const html = `
            <div class="route-card" data-kind="${kind}">
              <div class="route-time">${minInt}<span class="unit">분</span></div>
              <div class="route-main">
                <div class="${titleClass}">${titleText}</div>
                <div class="route-meta">
                  <span>거리 ${fmtDist(m)}</span>
                  <span>/ shade ${fmtPct(obj.avg_shade_ratio)}</span>
                </div>
              </div>
            </div>`;
          items.push(html);
        }
        build('coolest',  'title-coolest',  '시원한길', data.coolest);
        build('shortest', 'title-shortest', '최단거리', data.shortest);
        build('shelter',  'title-shelter',  '쉼터우선', data.shelter);  // [ADD]

        routesEl.innerHTML = items.join('') || '<div class="muted">경로가 없습니다.</div>';
        routesEl.onclick = (e)=>{
          const card = e.target.closest('.route-card');
          if(!card) return;
          selectRoute(card.getAttribute('data-kind'));
        };
      }

      function setLineColor(line, color){ if(line) line.setOptions({ strokeColor: color }); }
      function selectRoute(kind){
        selectedKind = kind;
        if(kind === 'shortest'){
          setLineColor(shortestPolyline, '#1BA952');
          setLineColor(coolestPolyline,  '#46e583');
          setLineColor(shelterPolyline,  '#03491e');
          if(shortestPolyline) shortestPolyline.setMap(map);
          if(coolestPolyline)  coolestPolyline.setMap(null);
          if(shelterPolyline)  shelterPolyline.setMap(null);
        }else if(kind === 'coolest'){
          setLineColor(coolestPolyline,  '#46e583');
          setLineColor(shortestPolyline, '#1BA952');
          setLineColor(shelterPolyline,  '#03491e');
          if(coolestPolyline)  coolestPolyline.setMap(map);
          if(shortestPolyline) shortestPolyline.setMap(null);
          if(shelterPolyline)  shelterPolyline.setMap(null);
        }else if(kind === 'shelter'){
          setLineColor(shelterPolyline,  '#03491e');
          setLineColor(shortestPolyline, '#1BA952');
          setLineColor(coolestPolyline,  '#46e583');
          if(shelterPolyline)  shelterPolyline.setMap(map);
          if(shortestPolyline) shortestPolyline.setMap(null);
          if(coolestPolyline)  coolestPolyline.setMap(null);
        }
        const routesEl = $('routes'); if(!routesEl) return;
        routesEl.querySelectorAll('.route-card').forEach(card=>{
          card.classList.toggle('active', card.getAttribute('data-kind')===kind);
        });
      }

      // ---- Shadow helpers ----
      function mapBbox(){
        const b = map.getBounds(); const sw=b.getSouthWest(), ne=b.getNorthEast();
        return [sw.getLng(), sw.getLat(), ne.getLng(), ne.getLat()];
      }
      async function loadShadow(kind){
        const t = (($('time')||{}).value) || "";
        const bbox = mapBbox();
        const qs = new URLSearchParams({bbox: bbox.join(','), time: t, kind}).toString();
        const res = await fetch('/shadow?'+qs); const js = await res.json();
        if(!res.ok) throw new Error(js.error || 'shadow API 실패');
        return js;
      }
      async function refreshShadows(){
        const btn=$('refresh-shadow'); if(btn){ btn.disabled=true; btn.innerText="불러오는 중..."; }
        try{
          let stats = [];
          if(($('toggle-building')||{}).checked){
            const b = await loadShadow('building');
            clearPolygons(shadowBuilding);
            if(b.gj){ shadowBuilding.push(...polygonsFromGeoJSON(b.gj, {strokeWeight:0.5, strokeColor:'#463f4f', fillColor:'#28252c', fillOpacity:0.35, strokeOpacity:1})); shadowBuilding.forEach(p=>p.setMap(map)); }
            stats.push("건물: "+b.count);
          } else { clearPolygons(shadowBuilding); }
          if(($('toggle-tree')||{}).checked){
            const t = await loadShadow('tree');
            clearPolygons(shadowTree);
            if(t.gj){ shadowTree.push(...polygonsFromGeoJSON(t.gj, {strokeWeight:0.3, strokeColor:'#4daf4a', fillColor:'#7fc97f', fillOpacity:0.6, strokeOpacity:1})); shadowTree.forEach(p=>p.setMap(map)); }
            stats.push("가로수: "+t.count);
          } else { clearPolygons(shadowTree); }
          if(($('toggle-shelter')||{}).checked){
            const s = await loadShadow('shelter');
            clearPolygons(shadowShelter);
            if(s.gj){ shadowShelter.push(...polygonsFromGeoJSON(s.gj, {strokeWeight:0.3, strokeColor:'#e66101', fillColor:'#fdae61', fillOpacity:0.6, strokeOpacity:1})); shadowShelter.forEach(p=>p.setMap(map)); }
            stats.push("쉼터: "+s.count);
          } else { clearPolygons(shadowShelter); }
          const ss=$('shadow-stats'); if(ss) ss.innerHTML = stats.join("  /  ");
        }catch(e){ alert("오류: "+e.message); }
        finally{ const btn=$('refresh-shadow'); if(btn){ btn.disabled=false; btn.innerText="갱신"; } }
      }
      const btnRefresh = $('refresh-shadow'); if(btnRefresh) btnRefresh.onclick = refreshShadows;
      const tb = $('toggle-building'); if(tb) tb.onchange = refreshShadows;
      const tt = $('toggle-tree');     if(tt) tt.onchange = refreshShadows;
      const ts = $('toggle-shelter');  if(ts) ts.onchange = refreshShadows;
      kakao.maps.event.addListener(map, 'idle', refreshShadows);

      // --- Kakao Places 검색 ---
      function doSearch(){
        const q = (($('kw')||{}).value || "").trim();
        if(!q){ alert("검색어를 입력하세요."); return; }
        places.keywordSearch(q, (data, status) => {
          clearSearchResults();
          if(status !== kakao.maps.services.Status.OK){ alert("검색 결과가 없습니다."); return; }
          const results = data.slice(0, 10);
          const bounds = new kakao.maps.LatLngBounds();
          const ul = $('kw-results'); if(!ul) return;

          results.forEach(p => {
            const lat = parseFloat(p.y), lng = parseFloat(p.x);
            const ll  = new kakao.maps.LatLng(lat, lng);
            const mk = new kakao.maps.Marker({ position: ll }); mk.setMap(map); searchMarkers.push(mk);
            bounds.extend(ll);
            const kakaoLink = `https://map.kakao.com/link/map/${encodeURIComponent(p.place_name)},${lat},${lng}`;
            const li = document.createElement('li');
            li.style.padding="6px 4px"; li.style.borderBottom="1px solid #eee";
            li.innerHTML = `
              <div style="display:flex; justify-content:space-between; gap:6px; align-items:center">
                <div style="flex:1; min-width:0">
                  <div style="font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">${p.place_name}</div>
                  <div style="color:#666; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">${p.road_address_name || p.address_name || ""}</div>
                  <a href="${kakaoLink}" target="_blank" style="color:#225ea8; text-decoration:none; font-size:11px">카카오맵</a>
                </div>
                <div style="flex:0 0 auto; display:flex; gap:4px">
                  <button class="ghost" style="padding:4px 6px" data-lat="${lat}" data-lng="${lng}" data-kind="src">출발</button>
                  <button class="ghost" style="padding:4px 6px" data-lat="${lat}" data-lng="${lng}" data-kind="dst">도착</button>
                </div>
              </div>`;
            ul.appendChild(li);
          });
          if(!bounds.isEmpty()) map.setBounds(bounds, 20, 20, 20, 20);
        }, { size: 15 });
      }
      const kwBtn=$('kw-btn'); if(kwBtn) kwBtn.onclick = doSearch;
      const kwIn =$ ('kw');   if(kwIn)  kwIn.addEventListener('keydown', e => { if(e.key==='Enter') doSearch(); });
      const ulRes=$('kw-results');
      if(ulRes){
        ulRes.addEventListener('click', (e) => {
          const btn = e.target.closest('button[data-kind]'); if(!btn) return;
          const lat = parseFloat(btn.getAttribute('data-lat'));
          const lng = parseFloat(btn.getAttribute('data-lng'));
          if(btn.getAttribute('data-kind') === 'src') setSrcByLatLng(lat, lng);
          else setDstByLatLng(lat, lng);
          map.setLevel(3); map.panTo(new kakao.maps.LatLng(lat, lng));
        });
      }

      // 첫 로딩 시 그림자 불러오기
      refreshShadows();

      // ---- GeoJSON helpers ----
      function polygonsFromGeoJSON(gj, style){
        const out = [];
        function latlng(lat, lng){ return new kakao.maps.LatLng(lat, lng); }
        function makePolygon(rings){
          const paths = rings.map(ring => ring.map(([lng,lat]) => latlng(lat,lng)));
          const poly = new kakao.maps.Polygon({
            map:null, path: paths,
            strokeWeight: style.strokeWeight ?? 1,
            strokeColor:  style.strokeColor  ?? '#000',
            strokeOpacity:style.strokeOpacity?? 1,
            strokeStyle:  style.strokeStyle  ?? 'solid',
            fillColor:    style.fillColor    ?? '#000',
            fillOpacity:  style.fillOpacity  ?? 0.5
          });
          out.push(poly);
        }
        function walk(geom){
          if(!geom) return;
          const t = geom.type, c = geom.coordinates;
          if(t==='Polygon') makePolygon(c);
          else if(t==='MultiPolygon') c.forEach(rings=>makePolygon(rings));
          else if(t==='GeometryCollection') (geom.geometries||[]).forEach(g=>walk(g));
        }
        if(gj.type==='FeatureCollection') (gj.features||[]).forEach(f=>walk(f.geometry));
        else if(gj.type==='Feature') walk(gj.geometry);
        else walk(gj);
        return out;
      }
      function polylineFromGeoJSON(gj, style){
        const pts = [];
        function latlng(lat, lng){ return new kakao.maps.LatLng(lat, lng); }
        function addLine(line){ line.forEach(([lng,lat])=>pts.push(latlng(lat,lng))); }
        function walk(geom){
          if(!geom) return;
          const t = geom.type, c = geom.coordinates;
          if(t==='LineString') addLine(c);
          else if(t==='MultiLineString') c.forEach(ls=>addLine(ls));
          else if(t==='GeometryCollection') (geom.geometries||[]).forEach(g=>walk(g));
        }
        if(gj.type==='FeatureCollection') (gj.features||[]).forEach(f=>walk(f.geometry));
        else if(gj.type==='Feature') walk(gj.geometry);
        else walk(gj);
        return new kakao.maps.Polyline({
          path: pts,
          strokeWeight: style.strokeWeight ?? 5,
          strokeColor:  style.strokeColor  ?? '#000',
          strokeOpacity:style.strokeOpacity?? 1,
          strokeStyle:  style.strokeStyle  ?? 'solid'
        });
      }
      // === Chatbot wiring ===
      const chatBtn = document.getElementById('chat-toggle');
      const chatBox = document.getElementById('chat-panel');
      const chatLog = document.getElementById('chat-log');
      const chatIn  = document.getElementById('chat-input');
      const chatSend= document.getElementById('chat-send');

      function pushMsg(who, text){
        const div = document.createElement('div');
        div.style.margin = '6px 0';
        div.innerHTML = `<div style="font-weight:700;color:${who==='me'?'#225ea8':'#111'}">${who==='me'?'나':'SHADI'}</div>
                        <div style="white-space:pre-wrap; line-height:1.35">${text}</div>`;
        chatLog.appendChild(div);
        chatLog.scrollTop = chatLog.scrollHeight;
      }

      async function askChat(t){
        pushMsg('me', t);
        chatIn.value = '';
        const res = await fetch('/chat/ask', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ message: t })
        });
        const js = await res.json();

        if(js.type === 'route' && js.src && js.dst){
          // 지도 입력창/마커에 주입 후 기존 "실행" 로직 그대로 활용
          setSrcByLatLng(js.src.lat, js.src.lon);
          setDstByLatLng(js.dst.lat, js.dst.lon);
          pushMsg('bot', js.text || '좌표를 인식했어요. 경로를 계산합니다.');
          document.getElementById('run')?.click();
        }else{
          pushMsg('bot', js.text || '도움이 필요하신가요? 출발/도착 좌표 두 쌍을 알려주시면 경로를 그려드릴게요.');
        }
      }

      chatBtn?.addEventListener('click', ()=> {
        chatBox.style.display = (chatBox.style.display==='none' || !chatBox.style.display)? 'block':'none';
        if(chatBox.style.display==='block') chatIn.focus();
      });
      chatSend?.addEventListener('click', ()=> { const v=(chatIn.value||'').trim(); if(v) askChat(v); });
      chatIn?.addEventListener('keydown', (e)=> { if(e.key==='Enter'){ const v=(chatIn.value||'').trim(); if(v) askChat(v); }});

      // 시작 안내 한 줄
      pushMsg('bot', '안녕하세요! 기능 설명을 물어보거나, 출발/도착 좌표 두 쌍을 주시면 바로 경로를 계산해 드려요.\n예) 출발 36.3617,127.3447 / 도착 36.3721,127.3452');

    }); // kakao.maps.load
  });   // DOMContentLoaded
</script>
</body>
</html>"""

@app.get("/")
def index():
    if not KAKAO_JS_KEY:
        return "환경변수 KAKAO_JS_KEY 가 설정되어야 지도가 표시됩니다.", 500
    return render_template_string(MAP_HTML, kakao_js_key=KAKAO_JS_KEY)

@app.get("/route")
def route():
    src_s = request.args.get("src","").strip()
    dst_s = request.args.get("dst","").strip()
    time_s= request.args.get("time","").strip()
    weight= request.args.get("weight","").strip()
    src = _parse_coord_pair(src_s); dst = _parse_coord_pair(dst_s)
    if not src or not dst:
        return jsonify(error="Invalid src/dst. Use 'lat,lon' or 'lon,lat'."), 400
    stamp = _stamp_from_time(time_s); union_table = f"shadow_union_{stamp}"
    try: cool_weight = float(weight) if weight else DEFAULT_COOL_WEIGHT
    except Exception: cool_weight = DEFAULT_COOL_WEIGHT
    try:
        out = _fetch_routes(PG_URL, src, dst, union_table, cool_weight)  # shortest, coolest
        # [ADD] 쉼터 우선 경로 추가
        try:
            via_shelter = _fetch_route_via_shelter(PG_URL, src, dst, union_table)
        except Exception as _e:
            via_shelter = None
        out["shelter"] = via_shelter
        return jsonify(out)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.get("/shadow")
def shadow():
    kind   = request.args.get("kind","union").strip().lower()
    bbox_s = request.args.get("bbox","").strip()
    time_s = request.args.get("time","").strip()
    tol_s  = request.args.get("tol","").strip()
    if kind not in {"union","building","tree","shelter"}: kind = "union"
    bbox = None
    if bbox_s:
        try:
            parts = [float(x) for x in bbox_s.split(",")]
            if len(parts)==4: bbox = tuple(parts)
        except Exception:
            return jsonify(error="Invalid bbox"), 400
    try: tol = float(tol_s) if tol_s else 0.7
    except Exception: tol = 0.7
    stamp = _stamp_from_time(time_s)
    prefix = {"union":"shadow_union","building":"shadow_building","tree":"shadow_tree","shelter":"shadow_shelter"}[kind]
    try:
        out = _fetch_shadow_any(PG_URL, prefix, stamp, bbox, simplify_tol_m=tol)
        return jsonify(out)
    except Exception as e:
        return jsonify(error=str(e)), 500

# [ADD] 무더위쉼터 목록 API
@app.get("/hot-shelters")
def hot_shelters():
    rows = []
    if not os.path.exists(HOT_SHELTER_CSV):
        return jsonify([])
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(HOT_SHELTER_CSV, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        lat = float(str(r.get("위도", "")).strip())
                        lng = float(str(r.get("경도", "")).strip())
                    except Exception:
                        continue
                    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                        continue
                    rows.append({
                        "name":  str(r.get("시설명", "무더위쉼터")).strip(),
                        "addr":  str(r.get("도로명 주소", "")).strip(),
                        "hours": str(r.get("운영 시간", "")).strip(),
                        "lat": lat, "lng": lng
                    })
            break
        except Exception:
            continue
    return jsonify(rows)

if __name__ == "__main__":
    print(f"Serving on http://{APP_HOST}:{APP_PORT}  (PG_URL={PG_URL})")
    app.run(host=APP_HOST, port=APP_PORT, debug=True)

# shadow_map_debug.py
import time, math, datetime, pytz, re, warnings
import pandas as pd, geopandas as gpd, folium, osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.affinity import translate
from shapely.validation import make_valid
from shapely.errors import GEOSException
from shapely.ops import unary_union, transform
from pysolar.solar import get_altitude, get_azimuth
from pyproj import Transformer
import numpy as np
from shapely.affinity import scale, rotate

# ────────────────────────────── 기본 설정 ──────────────────────────────
tz   = pytz.timezone("Asia/Seoul")
now  = tz.localize(datetime.datetime(2024, 7, 31, 18, 0, 0))   # 분석 시각
WIDTH_RATIO_TREE = 1.5                                         # 나무 그림자 폭 = 높이×1.5
proj = Transformer.from_crs(4326, 5179, always_xy=True)        # 면적(m²) 계산용

warnings.filterwarnings("ignore", message="I don't know about leap seconds")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

# ────────────────────────────── 유틸 함수 ──────────────────────────────
def shadow_len(h, alt):                        # 그림자 길이(m)
    return 0 if alt <= 0 else h / math.tan(math.radians(alt))

def offset_latlon(lat, lon, dist_m, brg):
    dlat =  dist_m*math.cos(math.radians(brg)) / 111_320
    dlon = (dist_m*math.sin(math.radians(brg))
            / (40075_000*math.cos(math.radians(lat))/360))
    return lat+dlat, lon+dlon

def tree_shadow_polygon(lat, lon, h, alt, azi):
    L = shadow_len(h, alt)
    half = max(1, h*WIDTH_RATIO_TREE)/2
    p1 = offset_latlon(lat, lon, half, (azi+90)%360)
    p2 = offset_latlon(lat, lon, half, (azi-90)%360)
    end_lat, end_lon = offset_latlon(lat, lon, L, azi)
    p3 = offset_latlon(end_lat, end_lon, half, (azi-90)%360)
    p4 = offset_latlon(end_lat, end_lon, half, (azi+90)%360)
    return Polygon([p1, p2, p3, p4])

def tree_shadow_ellipse(lat, lon, r_m, alt, azi):
    """수관 반경 r_m → 타원형 그림자 Polygon 반환"""
    if alt <= 0:
        return Polygon()            # 밤이면 그림자 X

    # ─ 1) 좌표체계: WGS84 → EPSG:5179 ─
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    to4326 = Transformer.from_crs(5179, 4326, always_xy=True)
    x0, y0 = to5179.transform(lon, lat)

    circle = Point(x0, y0).buffer(r_m)       # 반경 r_m짜리 원

    # ─ 2) 원 → 타원(늘리기) ─
    stretch = 1 / math.tan(math.radians(alt))     # 고도 낮을수록 길어짐
    ellip   = scale(circle, 1, stretch, origin=(x0, y0))

    # ─ 3) 태양과 직각이 되도록 회전 ─
    ellip   = rotate(ellip, (azi + 90) % 360, origin=(x0, y0))

    # ─ 4) 트렁크와 타원 앞머리가 맞게 L/2 만큼 뒤로 이동 ─
    L   = shadow_len(r_m, alt)                   # 캐노피 기준 그림자 길이
    dx  =  (L/2) * math.sin(math.radians(azi))
    dy  =  (L/2) * math.cos(math.radians(azi))
    ellip = translate(ellip, xoff=dx, yoff=dy)

    # ─ 5) EPSG:4326 로 환원 ─
    shadow = transform(lambda x, y, z=None: to4326.transform(x, y), ellip)
    return make_valid(shadow)

# ────────── 0. 충남대 50 m 버퍼 ──────────
CENTER_CNU = (36.36917, 127.34515)          # 충남대 정문 좌표 (대략)
DIST_M     = 150                            
deg = DIST_M / 111_320                      # 위도 1° ≈ 111,320 m
buffer_50m_poly = Polygon([
    (CENTER_CNU[1]-deg, CENTER_CNU[0]-deg),
    (CENTER_CNU[1]+deg, CENTER_CNU[0]-deg),
    (CENTER_CNU[1]+deg, CENTER_CNU[0]+deg),
    (CENTER_CNU[1]-deg, CENTER_CNU[0]+deg)
])

CENTER = CENTER_CNU                         # folium 지도 중심

WIDTH_RATIO_TREE = 3.0                      # ★ 더 넓게

# ────────── building_shadow_polygon 재정의 ──────────
def building_shadow_polygon(poly, h, alt, azi):
    if alt <= 0:
        return Polygon()

    L  = shadow_len(h, alt)
    dy =  L*math.cos(math.radians(azi)) / 111_320
    dx = (L*math.sin(math.radians(azi))
          / (40075_000*math.cos(math.radians(poly.centroid.y))/360))

    def _shadow(p):
        src  = list(p.exterior.coords)
        dest = [(x+dx, y+dy) for x, y in src]
        quads = [Polygon([src[i], src[i+1], dest[i+1], dest[i]])
                 for i in range(len(src)-1)]
        return unary_union([p, Polygon(dest), *quads])

    if isinstance(poly, Polygon):
        return make_valid(_shadow(poly))
    else:                                   # MultiPolygon
        return make_valid(unary_union(_shadow(g) for g in poly.geoms))


def geom_area_m2(geom):                        # Polygon·MultiPolygon 모두 OK
    proj_fn = lambda x, y, z=None: proj.transform(x, y)
    return transform(proj_fn, geom).area

def to_float_or_none(val):
    """문자열에서 숫자·소수점만 남겨 float 변환, 없으면 None 반환"""
    num = re.sub(r"[^0-9.]", "", str(val))
    return float(num) if num else None


# ────────────────────────────── 시작 로그 ──────────────────────────────
t0 = time.time()
print("▶ [START] shadow_map_debug.py 실행")

# ───────────────────── 0. 유성구 행정경계 폴리곤 ──────────────────────
admin_gdf  = ox.geocode_to_gdf("Yuseong-gu, Daejeon, South Korea")
admin_poly = make_valid(admin_gdf.loc[0, "geometry"].buffer(0))
CENTER     = (admin_poly.centroid.y, admin_poly.centroid.x)

# ───────────────────── 1. 가로수 CSV → 그림자 ────────────────────────
print("  • 가로수 CSV 로드 중 …")
trees = pd.read_csv("data/대전광역시_가로수 현황_20221201.csv", encoding="euc-kr")
trees_gdf = gpd.GeoDataFrame(
    trees.dropna(subset=["경도","위도"]),
    geometry=[Point(xy) for xy in zip(trees["경도"], trees["위도"])],
    crs="EPSG:4326",
)
trees_gdf = gpd.clip(trees_gdf, buffer_50m_poly)
print(f"    → 가로수 {len(trees_gdf):,} 개 (유성구)")

tree_layers = []
for _, r in trees_gdf.iterrows():
    lat, lon  = r["위도"], r["경도"]
    h         = float(r.get("수고", 4))          # 수고 없으면 4 m
    crown_r   = h * 0.25                        # 수관 반경 ≈ 높이 1/4
    alt, azi  = get_altitude(lat, lon, now), get_azimuth(lat, lon, now)

    poly = tree_shadow_ellipse(lat, lon, crown_r, alt, azi)  # ★ 새 함수
    if poly.is_empty: continue

    area  = geom_area_m2(poly)
    tip   = f"Tree shadow<br>H={h:.1f} m / crown≈{crown_r:.1f} m<br>{area:,.1f} ㎡"
    tree_layers.append((poly, tip))
print(f"    → 그림자 폴리곤 {len(tree_layers):,} 개 생성")

# ───────────────────── 2. 건물 OSM → 그림자 ──────────────────────────
try:
    # ① 먼저 작은 버퍼(150 m)로 시도
    build = ox.features_from_polygon(buffer_50m_poly, tags={"building": True})
except ox._errors.InsufficientResponseError:
    # ② 결과가 없으면 'point + dist' 방식으로 재시도
    build = ox.features_from_point(CENTER_CNU, dist=DIST_M,
                                   tags={"building": True})
# 공통 후처리
build = build.to_crs(epsg=4326).loc[~build.geometry.is_empty]
print(f"    → 건물 footprint {len(build):,} 개 (버퍼 {DIST_M} m)")

bld_layers = []
for _, row in build.iterrows():
    poly = make_valid(row.geometry)
    if poly.is_empty or not poly.is_valid or not isinstance(poly, (Polygon, MultiPolygon)):
        continue
    # 높이 추정
    h = to_float_or_none(row.get("height"))
    if h is None:
        lv = to_float_or_none(row.get("building:levels"))
        h = lv*3 if lv is not None else 10.0

    # 그림자
    alt, azi = get_altitude(poly.centroid.y, poly.centroid.x, now), get_azimuth(poly.centroid.y, poly.centroid.x, now)
    try:
        s_poly = make_valid(building_shadow_polygon(poly, h, alt, azi))
    except GEOSException:
        continue
    if s_poly.is_empty or not s_poly.is_valid:
        continue
    area = geom_area_m2(s_poly)
    cent = s_poly.centroid
    tooltip = (f"Building shadow<br>"
               f"Lat, Lon: {cent.y:.6f}, {cent.x:.6f}<br>"
               f"Height: {h:.1f} m<br>"
               f"Area: {area:,.1f} ㎡")
    bld_layers.append((s_poly, tooltip))
print(f"    → 그림자 폴리곤 {len(bld_layers):,} 개 생성")

# ───────────────────── 3. Folium 시각화 ──────────────────────────────
print("  • Folium 지도 생성 중 …")
m = folium.Map(location=CENTER, zoom_start=15)

# 건물 그림자 (빨간색 + 중심 마커)
for poly, tip in bld_layers:
    folium.GeoJson(gpd.GeoSeries([poly]).__geo_interface__,
                   style_function=lambda x: {"fillColor": "#ff5555",
                                             "color": "#ff5555",
                                             "weight": 0.6,
                                             "fillOpacity": 0.55},
                   tooltip=tip).add_to(m)
    c = poly.centroid
    folium.CircleMarker((c.y, c.x), radius=4,
                        color="red", fill=True, fill_opacity=1).add_to(m)

# 나무 그림자 (파란색)
for poly, tip in tree_layers:
    folium.GeoJson(gpd.GeoSeries([poly]).__geo_interface__,
                   style_function=lambda x: {"fillColor": "#004fb7",
                                             "color": "#004fb7",
                                             "weight": 0.4,
                                             "fillOpacity": 0.60},
                   tooltip=tip).add_to(m)

# 나무 위치 점
folium.GeoJson(trees_gdf[["geometry"]].__geo_interface__,
               marker=folium.CircleMarker(radius=2,
                                          color="green", fill=True)).add_to(m)

m.save("shadow_map.html")
print(f"✅ shadow_map.html 저장 완료  (총 {time.time()-t0:,.1f} 초)")
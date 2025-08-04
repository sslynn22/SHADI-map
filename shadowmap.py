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
now  = tz.localize(datetime.datetime(2024, 7, 31, 15, 0, 0))   # 분석 시각
WIDTH_RATIO_TREE = 7                                         # 나무 그림자 폭 = 높이×1.5
proj = Transformer.from_crs(4326, 5179, always_xy=True)        # 면적(m²) 계산용
SHELTER_SCALE = 3                                            # 쉼터 그림자 폭

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

    # ─ 3) 태양과 직각이 되도록 회전 =-0987654321
    ellip   = rotate(ellip, (azi + 90) % 360, origin=(x0, y0))

    # ─ 4) 트렁크와 타원 앞머리가 맞게 L/2 만큼 뒤로 이동 ─
    L   = shadow_len(r_m, alt)                   # 캐노피 기준 그림자 길이
    dx  =  (L/2) * math.sin(math.radians(azi))
    dy  =  (L/2) * math.cos(math.radians(azi))
    ellip = translate(ellip, xoff=dx, yoff=dy)

    # ─ 5) EPSG:4326 로 환원 ─
    shadow = transform(lambda x, y, z=None: to4326.transform(x, y), ellip)
    return make_valid(shadow)

def shelter_shadow_octagon(lat, lon, diameter_m, alt, azi):
    """
    • diameter_m : 파라솔 지름 (m)
    • alt, azi   : 고도각·방위각
    반환값        : shapely Polygon (팔각형 그림자)
    """
    if alt <= 0:
        return Polygon()              # 태양 고도 0 이하면 그림자 X

    r = diameter_m / 2
    # ─ A. 정팔각형 꼭짓점 (XY 평면) 생성 ─
    angles = [math.radians(22.5 + 45*i) for i in range(8)]   # 0° 대신 22.5° 돌려 중심 정렬
    base_pts = [( r*math.cos(th), r*math.sin(th)) for th in angles]
    base = Polygon(base_pts)

    # ─ B. 투영(늘리기) : 그림자 길이 계수 = 1/tan(alt) ─
    stretch = 1 / math.tan(math.radians(alt))
    shadow = scale(base, 1, stretch, origin=(0, 0))

    # ─ C. 방위각 +90° 로 회전 (태양과 직각) ─
    shadow = rotate(shadow, (azi + 90) % 360, origin=(0, 0))

    # ─ D. L/2 뒤로 평행이동해서 기둥과 연결 ─
    L = shadow_len(r, alt)
    dx, dy = (L/2)*math.sin(math.radians(azi)), (L/2)*math.cos(math.radians(azi))
    shadow = translate(shadow, xoff=dx, yoff=dy)

    # ─ E. WGS84 좌표로 위치시키기 ─
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    to4326 = Transformer.from_crs(5179, 4326, always_xy=True)
    cx, cy = to5179.transform(lon, lat)          # 쉼터 위치를 원점으로
    shadow = translate(shadow, xoff=cx, yoff=cy)
    shadow = transform(lambda x, y, z=None: to4326.transform(x, y), shadow)
    return make_valid(shadow)


# ────────── 0. 충남대 50 m 버퍼 ──────────
CENTER_CNU = (36.36917, 127.34515)          # 충남대 정문 좌표 (대략)
DIST_M     = 1000                            
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
        return make_valid(unary_union([_shadow(g) for g in poly.geoms]))


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

# ───────────────────── 1-B. 그늘막 쉼터 CSV → 그림자 ─────────────────────
print("  • 그늘막 쉼터 CSV 로드 중 …")
shel = pd.read_csv("data/대전광역시 유성구_그늘막쉼터_20240920.csv", encoding="euc-kr")

shel_gdf = gpd.GeoDataFrame(
    shel.dropna(subset=["위도", "경도"]),
    geometry=[Point(xy) for xy in zip(shel["위도"], shel["경도"])],
    crs="EPSG:4326"
)
shel_gdf = gpd.clip(shel_gdf, buffer_50m_poly)   # 버퍼 범위로 자르기
print(f"    → 쉼터 {len(shel_gdf):,} 개 (버퍼 범위)")


# ─────────────────── 2-A. Shapefile 건물 → 보라색 그림자 ───────────────────
print("  • Shapefile 건물 로드 중 …")
shp_gdf = (gpd.read_file("CH_D010_00_20250731.shp", encoding="euc-kr")
             .to_crs(epsg=4326))
shp_gdf = shp_gdf[shp_gdf["A4"].str.contains("대전광역시", na=False)]
shp_gdf = gpd.clip(shp_gdf, buffer_50m_poly)

shp_layers = []
for _, r in shp_gdf.iterrows():
    poly = make_valid(r.geometry)
    if poly.is_empty: continue
    floors = pd.to_numeric(r.get("A25"), errors="coerce")
    h      = floors*3 if not pd.isna(floors) else 10.0
    alt, azi = get_altitude(poly.centroid.y, poly.centroid.x, now), get_azimuth(poly.centroid.y, poly.centroid.x, now)
    s_poly  = building_shadow_polygon(poly, h, alt, azi)
    if not s_poly.is_valid or s_poly.is_empty: continue
    shp_layers.append((s_poly, f"Shapefile<br>높이≈{h:.1f} m"))

print(f"    → Shapefile 그림자 {len(shp_layers):,} 개")

# 셰이프 건물 합집합(중복 제거용)
shp_union = unary_union([g for g, _ in shp_layers])

shel_layers = []
for _, r in shel_gdf.iterrows():
    lat, lon = r["경도"], r["위도"]
    dia = to_float_or_none(r.get("그늘막지름")) or 3.0   # 원본 지름(m)
    dia *= SHELTER_SCALE          # ★ 여기서 지름을 1.6배 확대

    alt, azi = get_altitude(lat, lon, now), get_azimuth(lat, lon, now)
    poly = shelter_shadow_octagon(lat, lon, dia, alt, azi)  # 커진 지름 사용

    if poly.is_empty: continue

    area = geom_area_m2(poly)
    tip  = (f"쉼터 팔각 그림자<br>지름≈{dia} m<br>{area:,.1f} ㎡")
    shel_layers.append((poly, tip))

# ─────────────────── 2-B. OSM 건물 → 빨간색 그림자 ───────────────────
print("  • OSM 건물 로드 중 …")
try:
    osm = ox.features_from_polygon(buffer_50m_poly, tags={"building": True})
except ox._errors.InsufficientResponseError:
    osm = ox.features_from_point(CENTER_CNU, dist=DIST_M, tags={"building": True})
osm = osm.to_crs(epsg=4326)
print(f"    → OSM 건물 {len(osm):,} 개")

osm_layers = []
for _, row in osm.iterrows():
    poly = make_valid(row.geometry)
    if poly.is_empty or poly.intersects(shp_union):     # Shapefile과 겹치면 skip
        continue

    NAME_FIX = {
        "공과대학 2호관": 5,   # 층수 직접 지정
        "공과대학 4호관": 3,
    }
    bname = row.get("name:ko") or row.get("name")

    if bname in NAME_FIX:                # ① 수동 지정 우선
        h = NAME_FIX[bname] * 3
    else:                                # ② OSM 태그 → ③ 기본값
        h = to_float_or_none(row.get("height"))
        if h is None:
            lv = to_float_or_none(row.get("building:levels"))
            h  = lv*3 if lv else 10.0     # 기본 10 m

    alt, azi = get_altitude(poly.centroid.y, poly.centroid.x, now), get_azimuth(poly.centroid.y, poly.centroid.x, now)
    
    # 1) 유효한 폴리곤만 남기기
    if (poly.is_empty or poly.intersects(shp_union) or
        not isinstance(poly, (Polygon, MultiPolygon))):
        continue

    # 2) 이제 안전하게 그림자 생성
    s_poly = building_shadow_polygon(poly, h, alt, azi)
    if s_poly.is_empty or not s_poly.is_valid:
        continue
    tooltip = f"OSM<br>높이≈{h:.1f} m"
    osm_layers.append((s_poly, tooltip))
    
print(f"    → OSM 그림자 {len(osm_layers):,} 개")


# ───────────────────── 3. Folium 시각화 ──────────────────────────────
print("  • Folium 지도 생성 중 …")
m = folium.Map(location=CENTER, zoom_start=15)

# 3-A. Shapefile 건물 그림자(보라)
for poly, tip in shp_layers:
    folium.GeoJson(gpd.GeoSeries([poly]).__geo_interface__,
                   style_function=lambda x: {"fillColor": "#7e3ff2",
                                             "color": "#7e3ff2",
                                             "weight": 0.6,
                                             "fillOpacity": 0.55},
                   tooltip=tip).add_to(m)

# 3-B. OSM 건물 그림자(빨강)
for poly, tip in osm_layers:
    folium.GeoJson(gpd.GeoSeries([poly]).__geo_interface__,
                   style_function=lambda x: {"fillColor": "#ff5555",
                                             "color": "#ff5555",
                                             "weight": 0.6,
                                             "fillOpacity": 0.55},
                   tooltip=tip).add_to(m)

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

# 그늘막 쉼터 그림자
for poly, tip in shel_layers:
    folium.GeoJson(gpd.GeoSeries([poly]).__geo_interface__,
                   style_function=lambda x: {"fillColor": "#f39c12",
                                             "color": "#f39c12",
                                             "weight": 0.4,
                                             "fillOpacity": 0.60},
                   tooltip=tip).add_to(m)


m.save("shadow_map.html")
print(f"✅ shadow_map.html 저장 완료  (총 {time.time()-t0:,.1f} 초)")
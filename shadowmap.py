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
WIDTH_RATIO_TREE = 7                                         # 나무 그림자 폭 = 높이×1.5
proj = Transformer.from_crs(4326, 5179, always_xy=True)        # 면적(m²) 계산용
SHELTER_SCALE = 3                                            # 쉼터 그림자 폭

warnings.filterwarnings("ignore", message="I don't know about leap seconds")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

SHELTER_CSV = "data/대전광역시 유성구_그늘막쉼터_20240920.csv"

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

    # ─ 3) 그림자 방향 
    b = (azi + 180) % 360

    # 타원의 장축을 '그림자 방향'으로 정렬 (Shapely는 수학각도, CCW 기준 → 음수 회전)
    ellip = rotate(ellip, -b, origin=(x0, y0))

    # 트렁크와 앞머리 맞추기: L/2만큼 '그림자 방향'으로 평행이동
    dx = (L/2.0) * math.sin(math.radians(b))
    dy = (L/2.0) * math.cos(math.radians(b))

    # ─ 4) 트렁크와 타원 앞머리가 맞게 L/2 만큼 뒤로 이동 ─
    L   = shadow_len(r_m, alt)                   # 캐노피 기준 그림자 길이
    ellip = translate(ellip, xoff=dx, yoff=dy)

    # ─ 5) EPSG:4326 로 환원 ─
    shadow = transform(lambda x, y, z=None: to4326.transform(x, y), ellip)
    return make_valid(shadow)

def tree_shadow_circle(lat, lon, radius_m=3.0):
    """수관폭 6m(반지름 3m) 등 고정 반경으로 원형 그림자 생성"""
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    to4326 = Transformer.from_crs(5179, 4326, always_xy=True)
    x0, y0 = to5179.transform(lon, lat)
    circle = Point(x0, y0).buffer(radius_m)               # meters in EPSG:5179
    shadow = transform(lambda x, y, z=None: to4326.transform(x, y), circle)
    return make_valid(shadow)

def tree_shadow_dynamic(lat, lon, crown_d_m=6.0, height_m=10.0, alt=None, azi=None, when=None):
    """
    시간별 태양 고도/방위각을 반영해 수관 원(지름 crown_d_m)을
    태양 방향으로 (기본지름 + L) 길이의 타원으로 변환한 그림자 Polygon.
    L은 나무 높이 height_m로 계산.
    """
    # 고도/방위각 확보
    if alt is None or azi is None:
        if when is None:
            raise ValueError("alt/azi 또는 when(분석 시각)을 제공해야 합니다.")
        alt = get_altitude(lat, lon, when)
        azi = get_azimuth(lat, lon, when)

    if alt <= 0:
        return Polygon()  # 해 지면 그림자 없음

    # 좌표계를 미터 단위(5179)로
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    to4326 = Transformer.from_crs(5179, 4326, always_xy=True)
    x0, y0 = to5179.transform(lon, lat)

    r = crown_d_m / 2.0
    base_circle = Point(x0, y0).buffer(r)

    # 나무 높이로 그림자 길이 산정
    L = shadow_len(height_m, alt)  # = height / tan(alt)

    # 길이 (기본지름 + L) 이 되도록 한 축만 스케일
    # 기본 타원 장축 = 2*r*scale_y  →  2*r*scale_y = 2*r + L
    scale_y = (2*r + L) / (2*r)

    ellip = scale(base_circle, 1.0, scale_y, origin=(x0, y0))

    # 태양과 직각이 되도록 회전(그림자 장축이 태양방향을 가리키게)
    ellip = rotate(ellip, (azi + 90) % 360, origin=(x0, y0))

    # 줄기와 그림자 앞머리가 맞도록 L/2 만큼 태양방향으로 평행이동
    dx = (L/2.0) * math.sin(math.radians(azi))
    dy = (L/2.0) * math.cos(math.radians(azi))
    ellip = translate(ellip, xoff=dx, yoff=dy)

    # WGS84로 복귀
    shadow = transform(lambda x, y, z=None: to4326.transform(x, y), ellip)
    return make_valid(shadow)



def shelter_shadow_octagon(lat, lon, diameter_m, height_m, alt, azi):
    if alt <= 0 or diameter_m is None or height_m is None:
        return Polygon()
    if diameter_m <= 0 or height_m <= 0:
        return Polygon()

    # 반지름
    r = diameter_m / 2
    # 1) 원점(0,0)에 반지름 r짜리 팔각형 생성
    angles = [math.radians(22.5 + 45*i) for i in range(8)]
    base_pts = [(r*math.cos(th), r*math.sin(th)) for th in angles]
    base = Polygon(base_pts)

    # 2) 그림자 길이 (height_m 기준)
    L = shadow_len(height_m, alt)
    # 3) 늘리기 배율 = L / r
    stretch = L / r
    shadow = scale(base, 1, stretch, origin=(0, 0))

    # 4) 태양과 직각으로 회전
    b = (azi + 180) % 360  # 그림자 방향
    shadow = rotate(shadow, -b, origin=(0, 0))
    
    # 5) 기둥(쉼터)과 그림자 이어붙이기 (반만 평행이동)
    dx, dy = (L/2)*math.sin(math.radians(b)), (L/2)*math.cos(math.radians(b))
    shadow = translate(shadow, xoff=dx, yoff=dy)

    # 6) WGS84 좌표로 이동
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    to4326 = Transformer.from_crs(5179, 4326, always_xy=True)
    cx, cy = to5179.transform(lon, lat)
    shadow = translate(shadow, xoff=cx, yoff=cy)
    return make_valid(transform(lambda x, y, z=None: to4326.transform(x, y), shadow))


# ────────── 0. 충남대 50 m 버퍼 ──────────
CENTER_CNU = (36.36917, 127.34515)          # 충남대 정문 좌표 (대략)
# CENTER_CNU = (36.386793, 127.406985) # 쓰레기 집하·전환 시설
DIST_M     = 1000                            
deg = DIST_M / 111_320                      # 위도 1° ≈ 111,320 m
buffer_50m_poly = Polygon([
    (CENTER_CNU[1]-deg, CENTER_CNU[0]-deg),
    (CENTER_CNU[1]+deg, CENTER_CNU[0]-deg),
    (CENTER_CNU[1]+deg, CENTER_CNU[0]+deg),
    (CENTER_CNU[1]-deg, CENTER_CNU[0]+deg)
])

CENTER = CENTER_CNU                         # folium 지도 중심          

# ────────── building_shadow_polygon 재정의 ──────────
def building_shadow_polygon(poly, h, alt, azi):
    if alt <= 0:
        return Polygon()

    L  = shadow_len(h, alt)
    b  = (azi + 180) % 360  # 그림자 방향
    dy =  L*math.cos(math.radians(b)) / 111_320
    dx = (L*math.sin(math.radians(b))
        / (40075_000*math.cos(math.radians(poly.centroid.y))/360))

    def _shadow(p):
        src  = list(p.exterior.coords)
        dest = [(x+dx, y+dy) for x, y in src]
        quads = [Polygon([src[i], src[i+1], dest[i+1], dest[i]])
                 for i in range(len(src)-1)]
        return unary_union([Polygon(dest), *quads])

    # 1) 그림자 다각형 생성
    if isinstance(poly, Polygon):
        shadow = _shadow(poly)
    else:  # MultiPolygon
        shadow = unary_union([_shadow(g) for g in poly.geoms])

    # 2) 원래 건물 footprint 부분 제거 → 순수 그림자만 남김
    shadow = shadow.difference(poly)

    return make_valid(shadow)


def geom_area_m2(geom):               
    proj_fn = lambda x, y, z=None: proj.transform(x, y)
    return transform(proj_fn, geom).area

def to_float_or_none(val):
    """문자열에서 숫자·소수점만 남겨 float 변환, 없으면 None 반환"""
    num = re.sub(r"[^0-9.]", "", str(val))
    return float(num) if num else None

def fix_lonlat(df, lat_col="위도", lon_col="경도"):
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    # 한국 대략 범위: 위도 33~39, 경도 124~132
    lat_looks_like_lon = df[lat_col].between(124, 132).mean()
    lon_looks_like_lat = df[lon_col].between(33, 39).mean()
    if lat_looks_like_lon > 0.5 and lon_looks_like_lat > 0.5:
        df[[lat_col, lon_col]] = df[[lon_col, lat_col]]
    return df

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
    lat, lon = r["위도"], r["경도"]

    poly = tree_shadow_dynamic(
    lat, lon,
    crown_d_m=6.0, height_m=10.0,
    when=now
    ) 
    if poly.is_empty:
        continue

    area = geom_area_m2(poly)
    tip  = f"Tree shadow (원형)<br>H≈10 m / crown≈6 m<br>{area:,.1f} ㎡"
    tree_layers.append((poly, tip))
print(f"    → 그림자 폴리곤 {len(tree_layers):,} 개 생성")

# ───────────────────── 1-B. 그늘막 쉼터 CSV → 그림자 ─────────────────────
print("  • 그늘막 쉼터 CSV 로드 중 …")
shel = pd.read_csv(SHELTER_CSV, encoding="euc-kr")
shel_clean = (fix_lonlat(shel, "위도", "경도")
              .dropna(subset=["위도","경도"])
              .reset_index(drop=True))

shel_gdf = gpd.GeoDataFrame(
    shel_clean,
    geometry=gpd.points_from_xy(shel_clean["경도"], shel_clean["위도"]), # x=경도, y=위도
    crs="EPSG:4326"
)

shel_gdf = gpd.clip(shel_gdf, buffer_50m_poly)
print(f"    → 그늘막 쉼터 {len(shel_gdf):,} 개 (버퍼 범위)")


# ───────────────────── 1-C. 무더위 쉼터 CSV → 그림자 ─────────────────────

# 3-D. 무더위쉼터 레이어 (보라 + 아이콘)
print("  • 무더위쉼터 CSV 로드 중 …")
heat_csv = "data/유성구_무더위쉼터_20250817.csv"

# 인코딩 유연 처리
try:
    heat = pd.read_csv(heat_csv, encoding="utf-8-sig")
except UnicodeDecodeError:
    heat = pd.read_csv(heat_csv, encoding="cp949")

# 필수 좌표 결측 제거
heat = heat.dropna(subset=["위도", "경도"])

# ⚠️ 좌표는 (경도, 위도) 순서로 Point 생성해야 지도에 올바르게 찍힘
heat_gdf = gpd.GeoDataFrame(
    heat,
    geometry=[Point(xy) for xy in zip(heat["경도"], heat["위도"])],
    crs="EPSG:4326"
)

# 유성구 경계 내만 사용하고 싶으면 아래 한 줄 활성화 (원하면 유지, 아니면 주석)
heat_gdf = gpd.clip(heat_gdf, admin_poly)

print(f"    → 무더위쉼터 {len(heat_gdf):,} 개")

# 아이콘 마커 레이어
heat_fg = folium.FeatureGroup(name="🥵 무더위쉼터", show=True)
for _, r in heat_gdf.iterrows():
    name  = str(r.get("시설명", "무더위쉼터")).strip()
    addr  = str(r.get("도로명 주소", "")).strip()
    hours = str(r.get("운영 시간", "")).strip()
    lat, lon = float(r["위도"]), float(r["경도"])

    folium.Marker(
        location=[lat, lon],
        tooltip=name,  # 짧고 가벼운 툴팁
        popup=folium.Popup(
            f"<b>{name}</b><br>{addr}<br>운영 시간: {hours}",
            max_width=300
        ),
        icon=folium.Icon(color="orange", icon="sun", prefix="fa")
    ).add_to(heat_fg)

# ─────────────────── 2-A. Shapefile 건물 → 보라색 그림자 ───────────────────
print("  • Shapefile 건물 로드 중 …")
shp_gdf = (gpd.read_file("data/CH_D010_00_20250731.shp", encoding="euc-kr")
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
    lat, lon = float(r["위도"]), float(r["경도"])
    shelter_h = to_float_or_none(r.get("전체높이"))
    canopy_d  = to_float_or_none(r.get("펼침지름"))

    # None, 0, 음수 모두 기본값으로 보정
    if not shelter_h or shelter_h <= 0: shelter_h = 2.5
    if not canopy_d  or canopy_d  <= 0: canopy_d  = 2.0

    alt, azi  = get_altitude(lat, lon, now), get_azimuth(lat, lon, now)
    poly = shelter_shadow_octagon(lat, lon, canopy_d, shelter_h, alt, azi)

    if poly.is_empty: continue

    area = geom_area_m2(poly)
    tip  = (f"쉼터 팔각 그림자<br>지름≈{canopy_d:.1f} m<br>{area:,.1f} ㎡")
    shel_layers.append((poly, tip))
print(f"    → 쉼터 그림자 폴리곤 {len(shel_layers):,} 개 생성")

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
    else:                                # OSM 태그 → ③ 기본값
        # height 태그가 있으면 그대로 float, 없으면 None
        h_height = to_float_or_none(row.get("height"))
        # building:levels 가 "5;4" 같이 여러 개일 때 최대값만 골라 3m/층 으로 환산
        raw_lv = row.get("building:levels")
        lv_list = [int(x) for x in re.findall(r'\d+', str(raw_lv) if raw_lv else "")]
        if lv_list:
            h_levels = max(lv_list) * 3
        else:
            h_levels = None

        # 후보들 중 존재하는 값만 골라 최대값 → 없으면 10 m
        candidates = [h for h in (h_height, h_levels) if h is not None]
        h = max(candidates) if candidates else 10.0

    alt, azi = get_altitude(poly.centroid.y, poly.centroid.x, now), get_azimuth(poly.centroid.y, poly.centroid.x, now)
    
    if (poly.is_empty or poly.intersects(shp_union) or
        not isinstance(poly, (Polygon, MultiPolygon))):
        continue

    s_poly = building_shadow_polygon(poly, h, alt, azi)
    if s_poly.is_empty or not s_poly.is_valid:
        continue
    tooltip = f"OSM<br>높이≈{h:.1f} m"
    osm_layers.append((s_poly, tooltip))
    
print(f"    → OSM 그림자 {len(osm_layers):,} 개")


# ───────────────────── 3. Folium 시각화 (토글 완비) ─────────────────────
print("  • Folium 지도 생성 중 …")

# 1) 맵은 '딱 한 번' 생성하고, 기본/라이트 타일을 먼저 등록
m = folium.Map(location=CENTER, zoom_start=15, tiles=None)
folium.TileLayer("OpenStreetMap",       name="기본 지도").add_to(m)
folium.TileLayer("CartoDB positron",    name="밝은 지도").add_to(m)
# 필요하면 다크 테마도 추가 가능:
# folium.TileLayer("CartoDB dark_matter", name="다크 지도").add_to(m)

# 2) 오버레이 레이어들을 FeatureGroup으로 만들어 채운다
# 2-1) 건물 그림자
bld_fg = folium.FeatureGroup(name="🏢 건물 그림자", show=False)
for poly, tip in (shp_layers + osm_layers):
    folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda _:{
            "fillColor":"#28252c","color":"#463f4f",
            "weight":0.5,"fillOpacity":0.5
        },
        tooltip=tip
    ).add_to(bld_fg)
m.add_child(bld_fg)

# 2-2) 가로수 그림자 (6m 수관, 10m 고정 가정이면 그대로)
tree_fg = folium.FeatureGroup(name="🌳 가로수 그림자", show=True)
for _, r in trees_gdf.iterrows():
    lat, lon = float(r["위도"]), float(r["경도"])

    poly = tree_shadow_dynamic(
    lat, lon,
    crown_d_m=6.0, height_m=10.0,
    when=now
    )
    if poly.is_empty:
        continue

    folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda _:{
            "fillColor":"#7fc97f","color":"#4daf4a",
            "weight":0.5,"fillOpacity":0.5
        }
    ).add_to(tree_fg)
m.add_child(tree_fg)


# 2-3) 그늘막 쉼터 그림자
shelter_fg = folium.FeatureGroup(name="⛱️ 그늘막 쉼터 그림자", show=True)
for _, r in shel_gdf.iterrows():
    lat = r.geometry.y  # ← 항상 geometry에서 읽어 안전
    lon = r.geometry.x
    shelter_h = to_float_or_none(r.get("전체높이"))
    canopy_d  = to_float_or_none(r.get("펼침지름"))
    if not shelter_h or shelter_h <= 0: shelter_h = 2.5
    if not canopy_d  or canopy_d  <= 0: canopy_d  = 2.0

    alt, azi = get_altitude(lat, lon, now), get_azimuth(lat, lon, now)
    poly = shelter_shadow_octagon(lat, lon, canopy_d, shelter_h, alt, azi)
    if poly.is_empty:
        continue

    folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda _:{ "fillColor":"#fdae61","color":"#e66101",
                                  "weight":0.3,"fillOpacity":0.6 },
        tooltip=f"그늘막 쉼터 그림자<br>높이≈{shelter_h} m / 지름≈{canopy_d} m"
    ).add_to(shelter_fg)
m.add_child(shelter_fg)

# 2-4) 혐오시설
bad_fg = folium.FeatureGroup(name="🚮 혐오시설", show=False)

# bad_osm이 없다면 '빈 GeoDataFrame'으로 초기화해 NameError 방지
if 'bad_osm' not in locals():
    bad_osm = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

# 데이터가 있을 때만 실제 피처를 채움
if not bad_osm.empty:
    for _, row in bad_osm.iterrows():
        geom = make_valid(row.geometry)
        if geom.is_empty:
            continue
        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda _:{ "fillColor":"#ff6961","color":"#c23b22",
                                      "weight":0.5,"fillOpacity":0.5 },
            tooltip="<br>".join([f"{k}={row.get(k)}" for k in
                                 ["name","amenity","landuse","man_made","industrial","waste"]
                                 if row.get(k)])
        ).add_to(bad_fg)
m.add_child(bad_fg)


# 2-5) 무더위쉼터 
heat_fg = folium.FeatureGroup(name="🥵 무더위쉼터", show=True)
for _, r in heat_gdf.iterrows():
    name  = str(r.get("시설명","무더위쉼터")).strip()
    addr  = str(r.get("도로명 주소","")).strip()
    hours = str(r.get("운영 시간","")).strip()
    lat, lon = float(r["위도"]), float(r["경도"])
    folium.Marker(
        location=[lat, lon],
        tooltip=name,
        popup=folium.Popup(f"<b>{name}</b><br>{addr}<br>운영 시간: {hours}", max_width=320),
        icon=folium.Icon(color="orange", icon="sun", prefix="fa")
    ).add_to(heat_fg)
m.add_child(heat_fg)

# 3) (선택) 데이터가 있으면 자동으로 보기 영역 맞추기
try:
    candidates = []
    if shp_layers or osm_layers:
        candidates += [g for g,_ in (shp_layers + osm_layers)]
    if 'bad_osm' in locals() and not bad_osm.empty:
        candidates += list(bad_osm.geometry.values)
    if len(heat_gdf) > 0:
        candidates += list(heat_gdf.geometry.values)
    if candidates:
        tb = gpd.GeoSeries(candidates, crs="EPSG:4326").total_bounds
        m.fit_bounds([[tb[1], tb[0]],[tb[3], tb[2]]])
except Exception as e:
    print(f"(참고) 자동 줌 실패: {e}")

# 4) 마지막에 딱 한 번 LayerControl 추가 (여기서 해야 토글에 전부 보임)
folium.LayerControl(collapsed=False, position="topleft").add_to(m)

# 5) 저장
m.save("shadow_map_pretty.html")
print("shadow_map_pretty.html 저장 완료")

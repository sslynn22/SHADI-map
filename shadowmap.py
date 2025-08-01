# shadow_map_debug.py
import time, math, datetime, pytz
import pandas as pd, geopandas as gpd, folium, osmnx as ox
from shapely.geometry import Point, Polygon
from shapely.affinity  import translate
from shapely.validation import make_valid
from shapely.errors import GEOSException
from pysolar.solar     import get_altitude, get_azimuth
from pyproj            import Transformer
import warnings
from math import isnan

t0 = time.time()
print("▶ [START] shadow_map_debug.py 실행")

# ── 0. 글로벌 설정 ──────────────────────────────────────
tz   = pytz.timezone("Asia/Seoul")
now  = tz.localize(datetime.datetime(2024, 7, 31, 18, 0, 0))
CENTER = (36.355, 127.31)     # 대전 유성구
BBOX_EXTENT      = 0.01       # 약 ±1 km
WIDTH_RATIO_TREE = 0.3        # 나무 그림자 너비 = 높이×0.3
proj = Transformer.from_crs(4326, 5179, always_xy=True)

def shadow_len(h, alt): return 0 if alt <= 0 else h / math.tan(math.radians(alt))

def offset_latlon(lat, lon, dist_m, bearing_deg):
    dlat =  dist_m*math.cos(math.radians(bearing_deg)) / 111_320
    dlon = (dist_m*math.sin(math.radians(bearing_deg))
            / (40075_000*math.cos(math.radians(lat))/360))
    return lat+dlat, lon+dlon

def tree_shadow_polygon(lat, lon, h, alt, azi):
    L, W, half = shadow_len(h, alt), max(1, h*WIDTH_RATIO_TREE), h*WIDTH_RATIO_TREE/2
    p1 = offset_latlon(lat, lon, half, (azi+90)%360)
    p2 = offset_latlon(lat, lon, half, (azi-90)%360)
    end_lat, end_lon = offset_latlon(lat, lon, L, azi)
    p3 = offset_latlon(end_lat, end_lon, half, (azi-90)%360)
    p4 = offset_latlon(end_lat, end_lon, half, (azi+90)%360)
    return Polygon([p1, p2, p3, p4])

def building_shadow_polygon(poly, h, alt, azi):
    L = shadow_len(h, alt)
    dy =  L*math.cos(math.radians(azi)) / 111_320
    dx = (L*math.sin(math.radians(azi))
          / (40075_000*math.cos(math.radians(poly.centroid.y))/360))
    return translate(poly, xoff=dx, yoff=dy)

# ── (파일 맨 위) 경고 필터링 ─────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="I don't know about leap seconds after 2023"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="shapely"
)

# ── 1. 가로수 CSV 로드 ──────────────────────────────────
print("  • 가로수 CSV 로드 중 …")
trees = pd.read_csv("data/대전광역시_가로수 현황_20221201.csv", encoding="euc-kr")
trees = trees.dropna(subset=["경도", "위도"])
trees_gdf = gpd.GeoDataFrame(
    trees,
    geometry=[Point(xy) for xy in zip(trees["경도"], trees["위도"])],
    crs="EPSG:4326",
)
print(f"    → 가로수 {len(trees_gdf):,} 개 로드 완료")

# ── 1-B. 가로수 그림자 생성 ─────────────────────────────
print("  • 가로수 그림자 계산 중 …")
tree_layers = []
for _, r in trees_gdf.iterrows():
    lat, lon, h = r["위도"], r["경도"], float(r.get("수고", 4))
    alt, azi    = get_altitude(lat, lon, now), get_azimuth(lat, lon, now)
    poly        = make_valid(tree_shadow_polygon(lat, lon, h, alt, azi))
    if poly.is_empty: continue
    area = Polygon([proj.transform(*pt) for pt in poly.exterior.coords]).area
    tree_layers.append( (poly, f"Tree shadow<br>H={h} m<br>{area:,.1f} ㎡") )
print(f"    → 그림자 폴리곤 {len(tree_layers):,} 개 생성")

# ── 2. OSM 건물 취득 ────────────────────────────────────
print("  • OSM 건물 다운로드 중 … (최초 10–40 초 소요)")
# n, s = CENTER[0]+BBOX_EXTENT, CENTER[0]-BBOX_EXTENT
# e, w = CENTER[1]+BBOX_EXTENT, CENTER[1]-BBOX_EXTENT
# bbox = (n, s, e, w)  # (north, south, east, west)
# build = ox.features_from_bbox(bbox, {"building": True})
# print(f"    → 건물 footprint {len(build):,} 개 수신")

# ① 반경(m) → bbox 대신
dist_m = 1000            # ≒ BBOX_EXTENT 0.01°(약 1 km)
build = ox.features_from_point(CENTER, dist=dist_m,
                               tags={"building": True})
build = build.to_crs(epsg=4326).loc[~build.geometry.is_empty]
print(f"    → 건물 footprint {len(build):,} 개 수신")

# ── 2-B. 건물 그림자 생성 ───────────────────────────────
print("  • 건물 그림자 계산 중 …")
bld_layers = []
for _, row in build.iterrows():
    poly = make_valid(row.geometry)
    if poly.is_empty or not poly.is_valid or not isinstance(poly, Polygon):
        continue

    # 높이 추정
    if row.get("height"):
        h = float(str(row["height"]).replace("m", ""))
    elif row.get("building:levels"):
        h = float(row["building:levels"])*3
    else:
        h = 10.0        # 기본값


    alt, azi = get_altitude(poly.centroid.y, poly.centroid.x, now), get_azimuth(poly.centroid.y, poly.centroid.x, now)
    try:
        s_poly = make_valid(building_shadow_polygon(poly, h, alt, azi))
    except GEOSException:
        continue
    if s_poly.is_empty or not s_poly.is_valid:
        continue

    area = Polygon([proj.transform(*pt) for pt in s_poly.exterior.coords]).area
    cent = s_poly.centroid
    lat, lon = round(cent.y, 6), round(cent.x, 6)

    tooltip_txt = (
        f"Building shadow<br>"
        f"Lat, Lon: {lat}, {lon}<br>"
        f"Height: {h} m<br>"
        f"Area: {area:,.1f} ㎡"
    )
    bld_layers.append((s_poly, tooltip_txt))
    
    area = Polygon([proj.transform(*pt) for pt in s_poly.exterior.coords]).area

print(f"    → 그림자 폴리곤 {len(bld_layers):,} 개 생성")

# ── 3. Folium 시각화 ───────────────────────────────────
print("  • Folium 지도 생성 중 …")
m = folium.Map(location=CENTER, zoom_start=16)

# 가로수 그림자
for poly, tip in tree_layers:
    folium.GeoJson(gpd.GeoSeries([poly]).__geo_interface__,
                   style_function=lambda x: {"fillColor":"#000","color":"#000",
                                             "weight":0.4,"fillOpacity":0.35},
                   tooltip=tip).add_to(m)

# 건물 그림자
for poly, tip in bld_layers:
    folium.GeoJson(
        gpd.GeoSeries([poly]).__geo_interface__,
        style_function=lambda x: {
            "fillColor": "#555", "color": "#555",
            "weight": 0.3, "fillOpacity": 0.45},
        tooltip=tip
    ).add_to(m)

# 나무 위치
folium.GeoJson(trees_gdf[["geometry"]].__geo_interface__,
               marker=folium.CircleMarker(radius=2,color="green",fill=True)).add_to(m)

m.save("shadow_map.html")
print(f"✅ shadow_map.html 저장 완료  (총 {time.time()-t0:,.1f} 초)")

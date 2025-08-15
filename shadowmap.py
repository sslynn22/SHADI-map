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

def shelter_shadow_octagon(lat, lon, diameter_m, height_m, alt, azi):
    if alt <= 0:
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
    shadow = rotate(shadow, (azi + 90) % 360, origin=(0, 0))

    # 5) 기둥(쉼터)과 그림자 이어붙이기 (반만 평행이동)
    dx, dy = (L/2)*math.sin(math.radians(azi)), (L/2)*math.cos(math.radians(azi))
    shadow = translate(shadow, xoff=dx, yoff=dy)

    # 6) WGS84 좌표로 이동
    to5179 = Transformer.from_crs(4326, 5179, always_xy=True)
    to4326 = Transformer.from_crs(5179, 4326, always_xy=True)
    cx, cy = to5179.transform(lon, lat)
    shadow = translate(shadow, xoff=cx, yoff=cy)
    return make_valid(transform(lambda x, y, z=None: to4326.transform(x, y), shadow))


# ────────── 0. 충남대 50 m 버퍼 ──────────
# CENTER_CNU = (36.36917, 127.34515)          # 충남대 정문 좌표 (대략)
CENTER_CNU = (36.386793, 127.406985) # 쓰레기 집하·전환 시설
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
        return unary_union([Polygon(dest), *quads])

    # 1) 그림자 다각형 생성
    if isinstance(poly, Polygon):
        shadow = _shadow(poly)
    else:  # MultiPolygon
        shadow = unary_union([_shadow(g) for g in poly.geoms])

    # 2) 원래 건물 footprint 부분 제거 → 순수 그림자만 남김
    shadow = shadow.difference(poly)

    return make_valid(shadow)


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
    lat, lon = r["위도"], r["경도"]

    # CSV에 있는 실제 값을 읽어와서 사용
    shelter_h = to_float_or_none(r.get("전체높이"))    # 전체높이(m)
    canopy_d  = to_float_or_none(r.get("펼침지름"))     # 펼침지름(m)
    if shelter_h is None: shelter_h = 3.0
    if canopy_d  is None: canopy_d  = 3.0

    alt, azi = get_altitude(lat, lon, now), get_azimuth(lat, lon, now)
    # diameter_m=canopy_d, height_m=shelter_h 순으로 인자 전달
    poly = shelter_shadow_octagon(lat, lon, canopy_d, shelter_h, alt, azi)

    if poly.is_empty: continue

    area = geom_area_m2(poly)
    tip  = (f"쉼터 팔각 그림자<br>지름≈{canopy_d:.1f} m<br>{area:,.1f} ㎡")
    shel_layers.append((poly, tip))
    print(f"    → 쉼터 그림자 폴리곤 {len(tree_layers):,} 개 생성")

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
    else:                                # ② OSM 태그 → ③ 기본값
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


# ───────────────────── 3. Folium 시각화 (Pretty 버전) ─────────────────────
print("  • Folium 지도 생성 중 …")

# 최종 맵 생성: 타일 + 레이어 토글 한 번만
m = folium.Map(location=CENTER, zoom_start=15, tiles=None)
folium.TileLayer("OpenStreetMap", name="Default").add_to(m)
folium.TileLayer("CartoDB positron", name="Light").add_to(m)

# 3-A. 건물 그림자 레이어 (보라)
bld_fg = folium.FeatureGroup(name="🏢 건물 그림자", show=False)
for poly, tip in shp_layers + osm_layers:
    folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda x: {
            "fillColor": "#28252c",
            "color": "#463f4f",
            "weight": 0.5,
            "fillOpacity": 0.5
        },
        tooltip=tip
    ).add_to(bld_fg)

# 3-B. 나무 그림자 레이어 (연녹색)
tree_fg = folium.FeatureGroup(name="🌳 나무 그림자", show=True)
for _, r in trees_gdf.iterrows():
    lat, lon = r["위도"], r["경도"]
    # 고정 높이 10m, 수관폭 6m 적용
    alt, azi = get_altitude(lat, lon, now), get_azimuth(lat, lon, now)
    poly = tree_shadow_ellipse(lat, lon, 6/2, alt, azi)
    if poly.is_empty: continue
    folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#7fc97f",
            "color": "#4daf4a",
            "weight": 0.5,
            "fillOpacity": 0.5
        }
    ).add_to(tree_fg)
m.add_child(tree_fg)

# 3-C. 쉼터 그림자 레이어 (주황)
shelter_fg = folium.FeatureGroup(name="⛱️ 쉼터 그림자", show=True)
for _, r in shel_gdf.iterrows():
    # geometry.x/y 에 실제 (경도, 위도)가 들어 있으므로 이걸 바로 꺼냅니다
    lon = r.geometry.x
    lat = r.geometry.y

    shelter_h = to_float_or_none(r.get("전체높이")) or 2.5
    canopy_d  = to_float_or_none(r.get("펼침지름")) or 2.0

    # 위도(lat), 경도(lon)를 올바르게 넘겨 줍니다
    alt = get_altitude(lat, lon, now)
    azi = get_azimuth(lat, lon, now)
    poly = shelter_shadow_octagon(lat, lon, canopy_d, shelter_h, alt, azi)
    if poly.is_empty:
        continue

    folium.GeoJson(
        poly.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#fdae61",
            "color": "#e66101",
            "weight": 0.3,
            "fillOpacity": 0.6
        },
        tooltip=f"쉼터 그림자<br>높이≈{shelter_h} m / 지름≈{canopy_d} m"
    ).add_to(shelter_fg)

# ─────────────────── 3-D. OSM 혐오시설 레이어 (빨강) ───────────────────
print("  • OSM 혐오시설 로드 중 …")
bad_tags = {
    "landuse": ["landfill"],
    "amenity": [
        "waste_transfer_station",
        "waste_disposal",
        "incinerator",
        "crematorium",
        "recycling"
    ],
    "man_made": ["wastewater_plant", "composting_plant"]
}

def fetch_bad_features(poly, center, dist_m, tags):
    """poly에서 먼저 찾고 없으면 center+radius로 폴백."""
    try:
        gdf = ox.features_from_polygon(poly, tags=tags)
        print(f"    → polygon hit: {len(gdf):,} 개")
    except ox._errors.InsufficientResponseError:
        print("    → polygon 결과 없음 → point/radius 폴백 시도 …")
        try:
            gdf = ox.features_from_point(center, dist=dist_m*3, tags=tags)  # 반경 3배로 확장
            print(f"    → point/radius hit: {len(gdf):,} 개")
        except ox._errors.InsufficientResponseError:
            gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
            print("    → 여전히 결과 없음")
            return gdf

    # 좌표계 정리
    if getattr(gdf, "crs", None) is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

bad_osm = fetch_bad_features(buffer_50m_poly, CENTER_CNU, DIST_M, bad_tags)

if not bad_osm.empty:
    bad_fg = folium.FeatureGroup(name="🚮 혐오시설", show=True)
    for _, row in bad_osm.iterrows():
        geom = make_valid(row.geometry)
        if geom.is_empty:
            continue
        tooltip = "<br>".join(
            f"{k} = {v}" for k, v in row.items()
            if k in bad_tags and isinstance(v, str)
        )
        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#ff6961",
                "color":      "#c23b22",
                "weight":     0.5,
                "fillOpacity":0.5
            },
            tooltip=tooltip or "혐오시설"
        ).add_to(bad_fg)
    m.add_child(bad_fg)
else:
    print("    → 혐오시설 결과가 없어 레이어 생성을 건너뜁니다.")


# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# 생성해둔 FeatureGroup을 최종 맵에 붙이기 (순서: 나무 → 쉼터 → 혐오시설 → 건물)
m.add_child(tree_fg)
m.add_child(shelter_fg)
if 'bad_fg' in locals():
    m.add_child(bad_fg)
m.add_child(bld_fg)  # ← 건물을 맨 마지막에 올려서 다른 레이어 위에 보이게

# 보기 영역 자동 맞춤: 그림자/혐오시설이 있는 구역으로 줌
try:
    candidates = []
    if shp_layers or osm_layers:
        candidates += [g for g, _ in (shp_layers + osm_layers)]
    if 'bad_osm' in locals() and not bad_osm.empty:
        candidates += list(bad_osm.geometry.values)

    if candidates:
        tb = gpd.GeoSeries(candidates, crs="EPSG:4326").total_bounds  # (minx, miny, maxx, maxy)
        m.fit_bounds([[tb[1], tb[0]], [tb[3], tb[2]]])
    else:
        # 후보가 없으면 기존 CENTER 사용
        pass
except Exception as e:
    print(f"(참고) 자동 줌 실패: {e}")

folium.LayerControl(collapsed=False).add_to(m)
m.save("shadow_map_pretty.html")
print("✅ shadow_map_pretty.html 저장 완료")

# daejeon_osm_nimby.py
import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

# 1) 대상 지역: 대전광역시 경계
def get_daejeon_polygon():
    try:
        gdf = ox.geocode_to_gdf("대전광역시, 대한민국")
    except Exception:
        gdf = ox.geocode_to_gdf("Daejeon, South Korea")
    poly = gdf.loc[0, "geometry"]
    if poly.is_empty:
        raise RuntimeError("대전 광역시 경계를 찾지 못했습니다.")
    return poly

# 2) 혐오시설로 볼 수 있는 OSM 태그 묶음(필요시 추가/수정)
BAD_TAGS = {
    "landuse": ["landfill"],  # 매립장
    "amenity": [
        "waste_transfer_station",  # 쓰레기 집하장
        "waste_disposal",          # 폐기물 처리
        "recycling",               # 재활용 센터/포인트
        "crematorium"              # 화장시설
    ],
    "man_made": [
        "wastewater_plant",        # 하수처리장
        "composting_plant",        # 퇴비화 시설
        "incinerator"              # 소각장(일부 지역)
    ],
}

def fetch_features(poly):
    try:
        gdf = ox.features_from_polygon(poly, tags=BAD_TAGS)
    except ox._errors.InsufficientResponseError:
        # 결과가 없으면 빈 GeoDataFrame 반환
        gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    # CRS 보정
    if getattr(gdf, "crs", None) is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def representative_point(geom):
    if geom.is_empty:
        return None
    if isinstance(geom, Point):
        return geom
    try:
        return geom.representative_point()  # 폴리곤 내부 임의 대표점
    except Exception:
        return geom.centroid

def extract_records(gdf):
    rows = []
    for _, r in gdf.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        rp = representative_point(geom)
        if rp is None:
            continue
        lat, lon = rp.y, rp.x
        tag_key, tag_val = None, None
        for k in ["landuse", "amenity", "man_made"]:
            v = r.get(k)
            if isinstance(v, str):
                tag_key, tag_val = k, v
                break
        name = r.get("name") if isinstance(r.get("name"), str) else ""
        osm_id = r.get("osmid") if r.get("osmid") is not None else ""
        rows.append({
            "tag_key": tag_key or "",
            "tag_val": tag_val or "",
            "name": name,
            "lat": lat,
            "lon": lon,
            "osmid": osm_id,
            "geom_type": geom.geom_type
        })
    return pd.DataFrame(rows)

def print_summary(df):
    print("\n==== 대전 OSM 혐오시설 요약 ====")
    print(f"총 개수: {len(df):,}개")
    if df.empty:
        return
    print("\n좌표 목록 (lat, lon) — 태그 / 이름:")
    for i, row in enumerate(df.itertuples(index=False), start=1):
        tag = f"{row.tag_key}={row.tag_val}".strip("=")
        label = (row.name or "").strip()
        print(f"[{i:02d}] {row.lat:.6f}, {row.lon:.6f}  —  {tag}  {label}")

def save_outputs(df, bounds, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "daejeon_osm_nimby.csv")
    html_path = os.path.join(out_dir, "daejeon_osm_nimby_map.html")

    # CSV 저장(엑셀 호환을 위해 utf-8-sig)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {csv_path}")

    # 간단 Folium 맵(원클릭 확인용)
    try:
        import folium
        # 맵 중심은 대전 시청 근처로 대략
        m = folium.Map(location=[36.3504, 127.3845], zoom_start=12, tiles="CartoDB positron")
        # 피처 폴리곤도 GeoJson으로 추가
        # (필요 시 전체 gdf를 넘기면 되지만, 여기서는 대표점 마커 위주)
        for row in df.itertuples(index=False):
            folium.CircleMarker(
                location=(row.lat, row.lon),
                radius=4,
                tooltip=f"{row.tag_key}={row.tag_val} | {row.name}",
            ).add_to(m)
        # 피처 전체가 보이도록
        if bounds is not None:
            minx, miny, maxx, maxy = bounds  # lon/lat
            m.fit_bounds([[miny, minx], [maxy, maxx]])
        m.save(html_path)
        print(f"맵 저장 완료: {html_path}")
    except Exception as e:
        print(f"(참고) 맵 저장을 건너뜀: {e}")

def main():
    poly = get_daejeon_polygon()
    gdf = fetch_features(poly)

    if gdf.empty:
        print("\n대전 행정경계 내에서 지정한 태그에 해당하는 OSM 피처가 없습니다.")
        return

    df = extract_records(gdf)
    print_summary(df)

    # 결과 저장 + 지도를 전체 피처가 보이게 설정
    save_outputs(df, gdf.total_bounds)

if __name__ == "__main__":
    main()

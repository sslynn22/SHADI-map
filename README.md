# SHADI-map

# Conda 환경 생성 및 활성화
``` py
# 1) Python 3.9 버전 환경 생성
conda create -n shadow_map python=3.9 -y

# 2) 환경 활성화
conda activate shadow_map

# 3) 의존 패키지 설치
conda install -c conda-forge pandas geopandas folium osmnx shapely pyproj numpy pytz pysolar rtree fiona cairo matplotlib -y
```
# ai_chat/blueprint.py
import os, re, json, logging
from dataclasses import dataclass
from typing import Optional, Tuple
from flask import Blueprint, request, jsonify

# --- (선택) OpenAI: 설명/잡담용. 없거나 모델 오류면 우회 ---
_USE_AI = True
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
    _openai_model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
except Exception:
    _openai_client = None
    _USE_AI = False

# --- Kakao Local REST Key ---
KAKAO_REST_KEY = os.getenv("KAKAO_REST_KEY", "").strip()

# --- HTTP (requests) ---
import requests
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SHADI/route-bot"})
TIMEOUT = (3, 5)  # (connect, read) seconds

log = logging.getLogger(__name__)

# 좌표 정규식(숫자, 소수점, 부호)
COORD_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)')

# 출발/도착 분리 패턴들
_PATTERNS = [
    # 1) "출발 A / 도착 B"
    re.compile(r'(출발)\s*([^\n/]+?)\s*/\s*(도착)\s*([^\n]+)$'),
    # 2) "A에서 B까지"
    re.compile(r'(.+?)\s*에서\s*(.+?)\s*까지$'),
    # 3) "from A to B"
    re.compile(r'(?:from)\s+(.+?)\s+(?:to)\s+(.+)$', re.IGNORECASE),
    # 4) "출발 A 도착 B" (슬래시 없이)
    re.compile(r'출발\s*(.+?)\s*도착\s*(.+)$'),
]

@dataclass
class LL:
    lat: float
    lon: float

def _norm_pair(a: float, b: float) -> Optional[LL]:
    a = float(a); b = float(b)
    # "lat, lon" 우선. 경위도 값 범위 검사.
    if abs(a) <= 90 and abs(b) <= 180 and abs(a) < abs(b):
        return LL(lat=a, lon=b)
    # "lon, lat"도 허용
    if abs(b) <= 90 and abs(a) <= 180:
        return LL(lat=b, lon=a)
    return None

def _extract_two_coords(text: str):
    pairs = []
    for m in COORD_RE.finditer(text or ""):
        ll = _norm_pair(m.group(1), m.group(2))
        if ll:
            pairs.append(ll)
            if len(pairs) == 2:
                break
    return pairs if len(pairs) == 2 else None

def _split_start_end(text: str) -> Optional[Tuple[str, str]]:
    t = (text or "").strip()
    # 안전장치: 슬래시 구분이 가장 명확 -> 우선 처리
    if " / " in t:
        a, b = t.split(" / ", 1)
        a = re.sub(r'^\s*출발\s*', '', a).strip()
        b = re.sub(r'^\s*도착\s*', '', b).strip()
        if a and b:
            return (a, b)

    # 패턴 순회
    for pat in _PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        # 패턴별 그룹 수가 다름: 마지막 두 그룹을 출발/도착으로 간주
        g = [x for x in m.groups() if x is not None]
        if len(g) >= 2:
            start = g[-2].strip()
            end   = g[-1].strip()
            if start and end:
                return (start, end)

    # "출발 A" 단독, "도착 B" 단독이 있을 때
    m1 = re.search(r'출발\s*([^\n]+)', t)
    m2 = re.search(r'도착\s*([^\n]+)', t)
    if m1 and m2:
        a = m1.group(1).strip()
        b = m2.group(1).strip()
        if a and b:
            return (a, b)
    return None

def _kakao_keyword(query: str) -> Optional[LL]:
    """카카오 장소 키워드 검색 → 첫 후보 좌표 반환"""
    if not KAKAO_REST_KEY:
        return None
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {"query": query, "size": 3}
    try:
        r = SESSION.get(url, headers=headers, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        js = r.json()
        docs = js.get("documents") or []
        if not docs:
            return None
        # 1순위 후보
        d0 = docs[0]
        x = float(d0["x"]); y = float(d0["y"])
        return LL(lat=y, lon=x)
    except Exception as e:
        log.warning("kakao keyword error: %s", e)
        return None

def _kakao_address(query: str) -> Optional[LL]:
    """카카오 주소 검색(지번/도로명) → 좌표 반환 (키워드 실패시 보강)"""
    if not KAKAO_REST_KEY:
        return None
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {"query": query}
    try:
        r = SESSION.get(url, headers=headers, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        js = r.json()
        docs = js.get("documents") or []
        if not docs:
            return None
        d0 = docs[0]
        # 도로명/지번 케이스 모두 x,y 보장
        x = float(d0["x"]); y = float(d0["y"])
        return LL(lat=y, lon=x)
    except Exception as e:
        log.warning("kakao address error: %s", e)
        return None

def _geocode_place(q: str) -> Optional[LL]:
    """장소명/주소를 좌표로. 키워드→주소 순으로 시도."""
    q = (q or "").strip()
    if not q:
        return None
    p = _kakao_keyword(q)
    if p:
        return p
    return _kakao_address(q)

def _resolve_two_places(text: str) -> Optional[Tuple[LL, LL]]:
    """자연어에서 출발/도착 문자열을 분리해 각각 지오코딩."""
    se = _split_start_end(text)
    if not se:
        return None
    s_txt, d_txt = se
    s_ll = _geocode_place(s_txt)
    d_ll = _geocode_place(d_txt)
    if s_ll and d_ll:
        return (s_ll, d_ll)
    return None

def _ai_help(user_text: str) -> str:
    """모델이 있으면 간단 설명, 없으면 기본 안내 텍스트."""
    base_help = (
        "출발/도착을 장소명으로 입력하면 카카오 지오코딩으로 좌표를 찾고,\n"
        "그늘(건물/가로수/쉼터) 정보를 이용해 시원한 길을 계산해 드립니다.\n"
        "예) 출발 충남대 정문 / 도착 유성온천역\n"
        "예) from 대전역 to KAIST 정문\n"
        "TIP: '출발 … / 도착 …' 형태가 가장 정확합니다."
    )
    if not (_USE_AI and _openai_client and os.getenv("OPENAI_API_KEY")):
        return base_help
    try:
        resp = _openai_client.chat.completions.create(
            model=_openai_model,
            temperature=0.2,
            messages=[
                {"role":"system","content":"너는 SHADI(그늘길 라우팅) 안내 챗봇. 4문장 이내 한국어로 간단히 설명."},
                {"role":"user","content":f"사용자 입력: {user_text}\nSHADI의 핵심 기능을 간단히 안내해줘."}
            ]
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt or base_help
    except Exception as e:
        log.warning("OpenAI help fallback: %s", e)
        return base_help

def make_chat_blueprint():
    bp = Blueprint("ai_chat", __name__)

    @bp.post("/ask")
    def ask():
        data = request.get_json(force=True, silent=True) or {}
        text = str(data.get("message", "")).strip()
        if not text:
            return jsonify(error="message is required"), 400

        # 1) 좌표가 직접 들어온 경우 우선 처리
        pairs = _extract_two_coords(text)
        if pairs:
            return jsonify({
                "type": "route",
                "text": "좌표를 인식했어요. 경로를 계산할게요.",
                "src": {"lat": pairs[0].lat, "lon": pairs[0].lon},
                "dst": {"lat": pairs[1].lat, "lon": pairs[1].lon},
            })

        # 2) 장소명 → 카카오 지오코딩 (REST KEY 필요)
        if not KAKAO_REST_KEY:
            msg = (
                "장소명을 좌표로 바꾸려면 서버 환경변수 KAKAO_REST_KEY(카카오 REST API 키)가 필요합니다. "
                "관리자에게 REST 키 설정을 요청하거나, 임시로 좌표를 직접 입력해주세요.\n"
                "예) 출발 36.3617,127.3447 / 도착 36.3721,127.3452"
            )
            return jsonify({"type":"answer","text": msg})

        resolved = _resolve_two_places(text)
        if resolved:
            s_ll, d_ll = resolved
            return jsonify({
                "type": "route",
                "text": f"'{text}'에서 출발/도착을 해석했어요. 경로를 계산합니다.",
                "src": {"lat": s_ll.lat, "lon": s_ll.lon},
                "dst": {"lat": d_ll.lat, "lon": d_ll.lon},
            })

        # 3) 그래도 못 찾으면 간단 도움말
        help_txt = _ai_help(text)
        return jsonify({"type":"answer","text": help_txt})

    return bp

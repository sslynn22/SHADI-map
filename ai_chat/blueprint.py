import os, re, json, logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from flask import Blueprint, request, jsonify

# ----------------------------
# 로깅 (DEBUG 레벨)
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("SHADI.ai_chat")

# ----------------------------
# (선택) OpenAI: 설명/잡담용
# ----------------------------
_USE_AI = True
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
    _openai_model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
except Exception as e:
    log.warning("OpenAI import 실패: %s", e)
    _openai_client = None
    _USE_AI = False

# ----------------------------
# Kakao Local REST Key
# ----------------------------
KAKAO_REST_KEY = os.getenv("KAKAO_REST_KEY", "").strip()

# ----------------------------
# HTTP (requests)
# ----------------------------
import requests
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SHADI/route-bot"})
TIMEOUT = (3, 5)  # (connect, read)

# ----------------------------
# 정규식 및 패턴
# ----------------------------
COORD_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)')
_PATTERNS = [
    re.compile(r'(출발)\s*([^\n/]+?)\s*/\s*(도착)\s*([^\n]+)$'),
    re.compile(r'(.+?)\s*에서\s*(.+?)\s*까지'),
    re.compile(r'(?:from)\s+(.+?)\s+(?:to)\s+(.+)$', re.IGNORECASE),
    re.compile(r'출발\s*(.+?)\s*도착\s*(.+)$'),
]

HELP_KWS = re.compile(
    r"(기능\s*설명|설명|사용법|사용\s*방법|도움|도움말|헬프|help|what\s+can\s+you\s+do)",
    re.I)
COOL_ONLY_KWS = re.compile(r"(시원한\s*(길|경로).*(안내|추천|라우팅)|cool(est)?\s*route)", re.I)

# ----------------------------
# 데이터 클래스
# ----------------------------
@dataclass
class LL:
    lat: float
    lon: float

# ----------------------------
# 좌표 파싱
# ----------------------------
def _norm_pair(a: str, b: str) -> Optional[LL]:
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return None
    if abs(fa) <= 90 and abs(fb) <= 180 and abs(fa) < abs(fb):
        return LL(lat=fa, lon=fb)
    if abs(fb) <= 90 and abs(fa) <= 180:
        return LL(lat=fb, lon=fa)
    return None

def _extract_two_coords(text: str) -> Optional[List[LL]]:
    log.debug("_extract_two_coords: text='%s'", text)
    pairs: List[LL] = []
    for m in COORD_RE.finditer(text or ""):
        ll = _norm_pair(m.group(1), m.group(2))
        log.debug("  좌표 매치 groups=%s -> %s", m.groups(), ll)
        if ll:
            pairs.append(ll)
            if len(pairs) == 2:
                break
    log.debug("  좌표 추출 결과: %s", pairs)
    return pairs if len(pairs) == 2 else None

# ----------------------------
# 장소 텍스트 정리/분리
# ----------------------------
def _clean_place(s: str) -> str:
    orig = s
    s = s.strip()
    s = re.sub(r'^(출발|도착)\s*', '', s)      # 앞 키워드 제거
    s = re.sub(r'(까지|행)\s*$', '', s)        # 뒤 꼬리 제거 ('로'는 보존)
    log.debug("_clean_place: '%s' -> '%s'", orig, s)
    return s

def _split_start_end(text: str) -> Optional[Tuple[str, str]]:
    log.debug("_split_start_end 입력: '%s'", text)
    t = (text or "").strip()
    # 문장 꼬리(요청형 표현) 제거
    t = re.sub(r'(시원한\s*(길|거리).*|최단\s*(길|거리).*|경로.*|안내.*|보여.*|띄워.*)$', '', t).strip()

    # 1) 슬래시 구분
    if " / " in t:
        a, b = t.split(" / ", 1)
        a, b = _clean_place(a), _clean_place(b)
        log.debug("  슬래시 분리: a='%s', b='%s'", a, b)
        return (a, b) if a and b else None

    # 2) 패턴들
    for pat in _PATTERNS:
        m = pat.search(t)
        if m:
            g = [x for x in m.groups() if x]
            log.debug("  패턴 매치: %s -> groups=%s", pat.pattern, g)
            if len(g) >= 2:
                start, end = _clean_place(g[-2]), _clean_place(g[-1])
                return (start, end) if start and end else None

    # 3) 출발/도착 키워드 분리
    m1 = re.search(r'출발\s*([^\n]+)', t)
    m2 = re.search(r'도착\s*([^\n]+)', t)
    if m1 and m2:
        start, end = _clean_place(m1.group(1)), _clean_place(m2.group(1))
        log.debug("  키워드 분리: start='%s', end='%s'", start, end)
        return (start, end)
    log.debug("  출발/도착 분리 실패")
    return None

# ----------------------------
# Kakao 지오코딩
# ----------------------------
def _kakao_keyword(query: str) -> Optional[LL]:
    if not KAKAO_REST_KEY:
        log.debug("_kakao_keyword: KAKAO_REST_KEY 미설정")
        return None
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {"query": query, "size": 3}
    try:
        log.debug("Kakao keyword 호출: %s params=%s", url, params)
        r = SESSION.get(url, headers=headers, params=params, timeout=TIMEOUT)
        log.debug("  응답 status=%s", r.status_code)
        r.raise_for_status()
        js = r.json()
        docs = (js.get("documents") or [])
        log.debug("  결과 건수=%d (top=%s)", len(docs), (docs[0]["place_name"] if docs else None))
        if not docs:
            return None
        x, y = float(docs[0]["x"]), float(docs[0]["y"])
        return LL(lat=y, lon=x)
    except Exception as e:
        try:
            snippet = r.text[:200] if 'r' in locals() and hasattr(r, "text") else ""
        except Exception:
            snippet = ""
        log.warning("kakao keyword error: %s (body~200='%s')", e, snippet)
        return None

def _kakao_address(query: str) -> Optional[LL]:
    if not KAKAO_REST_KEY:
        log.debug("_kakao_address: KAKAO_REST_KEY 미설정")
        return None
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {"query": query}
    try:
        log.debug("Kakao address 호출: %s params=%s", url, params)
        r = SESSION.get(url, headers=headers, params=params, timeout=TIMEOUT)
        log.debug("  응답 status=%s", r.status_code)
        r.raise_for_status()
        js = r.json()
        docs = (js.get("documents") or [])
        log.debug("  결과 건수=%d", len(docs))
        if not docs:
            return None
        x, y = float(docs[0]["x"]), float(docs[0]["y"])
        return LL(lat=y, lon=x)
    except Exception as e:
        try:
            snippet = r.text[:200] if 'r' in locals() and hasattr(r, "text") else ""
        except Exception:
            snippet = ""
        log.warning("kakao address error: %s (body~200='%s')", e, snippet)
        return None

def _geocode_place(q: str) -> Optional[LL]:
    q = (q or "").strip()
    if not q:
        return None
    log.debug("_geocode_place: '%s'", q)
    p = _kakao_keyword(q)
    if p:
        return p
    return _kakao_address(q)

# ----------------------------
# 장소 2개 해석(+실패 사유)
# ----------------------------
def _resolve_two_places_with_reason(text: str):
    log.debug("_resolve_two_places_with_reason 입력: '%s'", text)
    se = _split_start_end(text)
    if not se:
        return None, "출발/도착 구문을 인식하지 못했어요. '출발 A / 도착 B' 형식으로 입력해 보세요."
    s_txt, d_txt = se
    log.debug("  파싱: 출발='%s', 도착='%s'", s_txt, d_txt)
    s_ll = _geocode_place(s_txt)
    d_ll = _geocode_place(d_txt)
    log.debug("  지오코딩: 출발=%s, 도착=%s", s_ll, d_ll)
    if not s_ll and not d_ll:
        return None, f"'{s_txt}'와(과) '{d_txt}'를 찾지 못했어요. 다른 이름이나 좌표를 입력해 주세요."
    if not s_ll:
        return None, f"출발지 '{s_txt}'를 찾지 못했어요. 다른 이름이나 좌표를 입력해 주세요."
    if not d_ll:
        return None, f"도착지 '{d_txt}'를 찾지 못했어요. 다른 이름이나 좌표를 입력해 주세요."
    return (s_ll, d_ll), None

# (구형 헬퍼: 호환용)
def _resolve_two_places(text: str) -> Optional[Tuple[LL, LL]]:
    se = _split_start_end(text)
    if not se:
        return None
    s_txt, d_txt = se
    s_ll = _geocode_place(s_txt)
    d_ll = _geocode_place(d_txt)
    if s_ll and d_ll:
        return (s_ll, d_ll)
    return None

# ----------------------------
# OpenAI 안내(토큰 절약)
# ----------------------------
def _ai_help(user_text: str) -> str:
        # 🔒 도움말은 외부 생성 사용 금지 — SHADI 고정 가이드 반환
    guide = (
        "첫째, 건물·가로수·그늘막 그림자를 지도에 보여드려요.\n"
        "둘째, 출발지랑 목적지를 말해주시면, 그늘 많은 길을 찾아드려요.\n"
        "셋째, 선택한 경로로 길안내를 해드리고요.\n"
        "마지막으로, 안내가 끝난 뒤 만족도를 물어보고 그 피드백으로 더 똑똑해집니다.\n\n"

        "예를 들어 '출발 충남대 정문 / 도착 유성온천역' 이렇게 말씀하시면 가장 정확해요."
    )
    return guide

# ----------------------------
# Flask Blueprint
# ----------------------------
def make_chat_blueprint():
    log.debug("make_chat_blueprint: 생성 시작")
    bp = Blueprint("ai_chat", __name__)

    @bp.post("/ask")
    def ask():
        data = request.get_json(force=True, silent=True) or {}
        text = str(data.get("message", "")).strip()

        # --- 0) 기능 설명/사용법 요청: 규칙 기반 즉시 응답 (고정 가이드) ---
        if HELP_KWS.search(text):
            return jsonify({"type": "answer", "text": _ai_help(text)})

        # --- 0-1) 시원한 경로만 요청인데 좌표/장소가 없는 경우: 사용법 안내 ---
        if COOL_ONLY_KWS.search(text) and not _extract_two_coords(text) and not _split_start_end(text):
            return jsonify({"type": "answer",
                            "text": "시원한 길만 안내할게요! 먼저 출발/도착을 알려주세요.\n예) 출발 충남대 정문 / 도착 유성온천역"})


        if re.search(r'(기능\s*설명|도움말|사용법|사용\s*방법|help|헬프)', text, re.IGNORECASE):
            return jsonify({"type": "answer", "text": _ai_help(text)})

        log.debug("/ask 수신: message='%s'", text)

        if not text:
            log.debug("  빈 메시지")
            return jsonify(error="message is required"), 400

        # --- 경로 선호도 키워드 파악 ---
        prefer_kind = None
        tlow = text.lower()
        if "쉼터" in text or "shelter" in tlow:
            prefer_kind = "shelter"
        elif "시원" in text or "cool" in tlow:
            prefer_kind = "coolest"
        elif "최단" in text or "short" in tlow:
            prefer_kind = "shortest"

        # ✅ 선호가 없으면 세 가지 모두
        prefer_modes = [prefer_kind] if prefer_kind else ["shortest", "coolest", "shelter"]

        # 1) 좌표 직접 입력
        # (장소명 지오코딩 성공) 분기
        pairs = _extract_two_coords(text)
        if pairs:
            s_ll, d_ll = pairs
            log.debug("  좌표 입력 인식 -> route 반환: %s -> %s", s_ll, d_ll)
            return jsonify({
                "type": "route",
                "text": f"'{text}'에서 출발/도착을 해석했어요. 경로를 계산합니다.",
                "src": {"lat": s_ll.lat, "lon": s_ll.lon},
                "dst": {"lat": d_ll.lat, "lon": d_ll.lon},
                # prefer: 단일 선호어가 있을 때만 지정, 없으면 'all'로 프론트가 판단
                "prefer": (prefer_modes[0] if len(prefer_modes) == 1 else "all"),
                "modes": prefer_modes
            })

        # 2) 장소명 지오코딩 (키 없으면 즉시 안내)
        if not KAKAO_REST_KEY:
            log.debug("  KAKAO_REST_KEY 미설정 -> 안내 반환")
            msg = (
                "장소명을 좌표로 바꾸려면 서버 환경변수 KAKAO_REST_KEY(카카오 REST API 키)가 필요합니다.\n"
                "관리자에게 REST 키 설정을 요청하거나, 임시로 좌표를 직접 입력해주세요.\n"
                "예) 출발 36.3617,127.3447 / 도착 36.3721,127.3452"
            )
            return jsonify({"type": "answer", "text": msg})

        # 3) 장소명 해석 + 실패 사유 피드백
        resolved, err = _resolve_two_places_with_reason(text)
        if resolved:
            s_ll, d_ll = resolved
            log.debug("  해석 성공 -> route 반환: src=%s dst=%s", s_ll, d_ll)
            return jsonify({
                "type": "route",
                "text": f"'{text}'에서 출발/도착을 해석했어요. 경로를 계산합니다.",
                "src": {"lat": s_ll.lat, "lon": s_ll.lon},
                "dst": {"lat": d_ll.lat, "lon": d_ll.lon},
                "prefer": (prefer_modes[0] if len(prefer_modes) == 1 else "all"),  # ⬅️ PATCH
                "modes": prefer_modes
            })

        # 4) 실패: 구체 사유 반환
        log.debug("  해석 실패 -> err='%s'", err)
        return jsonify({"type": "answer", "text": err + "\n\n" + _ai_help(text)})

    log.debug("make_chat_blueprint: 생성 완료 -> Blueprint 반환")
    return bp

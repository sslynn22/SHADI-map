# ai_bandit.py
import json, math, datetime, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

# ---------- Feature 설계 ----------
# 각 후보(kind: shortest/coolest/shelter)의 메타(거리, 그늘비율) + 컨텍스트(시간)
# x = [bias, norm_dist, avg_shade, hour_sin, hour_cos]
def _parse_stamp(stamp: str) -> datetime.datetime:
    # 'YYYYMMDD_HHMM'만 들어온다고 가정 (main에서 보정)
    return datetime.datetime.strptime(stamp, "%Y%m%d_%H%M")

def _hour_feats(dt: datetime.datetime) -> Tuple[float, float]:
    h = dt.hour + dt.minute/60.0
    a = 2*math.pi*h/24.0
    return math.sin(a), math.cos(a)

def _safe(v, default=None):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

@dataclass
class Candidate:
    kind: str                 # 'shortest' | 'coolest' | 'shelter'
    total_m: Optional[float]  # 총 거리(m)
    avg_shade: Optional[float]# 0~1

class ContextBandit:
    """
    Linear UCB per-arm
      A_a (dxd), b_a (d), theta_a = A^-1 b
      p_a = x^T theta_a + alpha * sqrt(x^T A^-1 x)
    """
    def __init__(self, pg_url: str, alpha: float = 0.6, ridge: float = 1.0):
        self.pg_url = pg_url
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.d = 5  # feature dimension
        self._ensure_schema()

    # ---------- PG 스키마 ----------
    def _ensure_schema(self):
        with psycopg2.connect(self.pg_url) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bandit_models (
                    arm TEXT PRIMARY KEY,                      -- shortest/coolest/shelter
                    A   DOUBLE PRECISION[][] NOT NULL,         -- dxd
                    b   DOUBLE PRECISION[]  NOT NULL,          -- d
                    updated_at TIMESTAMPTZ DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bandit_events (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT now(),
                    stamp TEXT,                 -- YYYYMMDD_HHMM
                    arm TEXT,                   -- chosen kind
                    reward DOUBLE PRECISION,    -- 0~1
                    src_lat DOUBLE PRECISION, src_lon DOUBLE PRECISION,
                    dst_lat DOUBLE PRECISION, dst_lon DOUBLE PRECISION,
                    x DOUBLE PRECISION[],       -- feature used
                    meta JSONB                  -- raw meta (dist/shade of arms etc.)
                );
            """)
            # 초기 모델(arm 3개) 없으면 생성
            for arm in ("shortest","coolest","shelter"):
                cur.execute("SELECT 1 FROM bandit_models WHERE arm=%s;", (arm,))
                if cur.fetchone() is None:
                    A = self._eye(self.d, self.ridge)
                    b = [0.0]*self.d
                    cur.execute("INSERT INTO bandit_models(arm, A, b) VALUES(%s,%s,%s);", (arm, A, b))

    # ---------- 선형대수 보조 ----------
    def _eye(self, d: int, c: float) -> List[List[float]]:
        M = [[0.0]*d for _ in range(d)]
        for i in range(d):
            M[i][i] = c
        return M

    def _matvec(self, M: List[List[float]], v: List[float]) -> List[float]:
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    def _add_outer(self, A: List[List[float]], x: List[float]):
        for i in range(len(x)):
            xi = x[i]
            for j in range(len(x)):
                A[i][j] += xi*x[j]

    def _add_vec(self, b: List[float], x: List[float], s: float):
        for i in range(len(x)):
            b[i] += s*x[i]

    def _inv(self, M: List[List[float]]) -> List[List[float]]:
        # 단순 가우스-조던 (d=5 고정 소규모라 충분)
        d = len(M)
        A = [row[:] for row in M]
        I = self._eye(d, 1.0)
        for col in range(d):
            # pivot
            piv = col
            for r in range(col, d):
                if abs(A[r][col]) > abs(A[piv][col]): piv = r
            if abs(A[piv][col]) < 1e-12:
                # 아주 드물게 수치불안정 → ridge 더
                for k in range(d): A[k][k] += 1e-6
                piv = col
            if piv != col:
                A[col], A[piv] = A[piv], A[col]
                I[col], I[piv] = I[piv], I[col]
            # normalize
            denom = A[col][col] if abs(A[col][col])>1e-12 else 1e-12
            f = 1.0/denom
            A[col] = [x*f for x in A[col]]
            I[col] = [x*f for x in I[col]]
            # eliminate
            for r in range(d):
                if r == col: continue
                factor = A[r][col]
                if abs(factor) < 1e-15: continue
                A[r] = [A[r][c] - factor*A[col][c] for c in range(d)]
                I[r] = [I[r][c] - factor*I[col][c] for c in range(d)]
        return I

    def _dot(self, u: List[float], v: List[float]) -> float:
        return sum((u[i]*v[i]) for i in range(len(u)))

    # ---------- Feature 만들기 ----------
    def make_features(self, stamp: str,
                      candidate: Candidate,
                      ref_dist_m: Optional[float]) -> List[float]:
        """
        ref_dist_m: 기준 거리(세 후보 중 최소 거리)로 정규화에 사용
        """
        dt = _parse_stamp(stamp)
        hsin, hcos = _hour_feats(dt)
        dist = _safe(candidate.total_m, None)
        shade = _safe(candidate.avg_shade, None)

        # 거리 정규화: 없는 경우 0, 있는 경우 minDist 기준 0~?
        if dist is None or not ref_dist_m or ref_dist_m <= 0:
            nd = 0.0
        else:
            nd = min(3.0, dist / ref_dist_m)  # 매우 먼 값은 3으로 캡
        if shade is None: shade = 0.0
        # x = [bias, norm_dist, avg_shade, hour_sin, hour_cos]
        return [1.0, nd, shade, hsin, hcos]

    # ---------- 예측(정렬) ----------
    def rank(self, stamp: str,
             src: Tuple[float,float], dst: Tuple[float,float],
             candidates: Dict[str, Candidate]) -> Tuple[List[str], Dict[str, float], Dict[str, List[float]]]:
        """
        return: (ordered_kinds, scores, x_by_kind)
        """
        with psycopg2.connect(self.pg_url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 모델 로드
            cur.execute("SELECT arm, A, b FROM bandit_models;")
            models = {r["arm"]: (r["A"], r["b"]) for r in cur.fetchall()}

        # 기준 거리(세 후보 최소값)
        dists = [c.total_m for c in candidates.values() if c.total_m is not None]
        ref = min(dists) if dists else None

        scores, xmap = {}, {}
        for kind, cand in candidates.items():
            x = self.make_features(stamp, cand, ref)
            xmap[kind] = x
            A, b = models.get(kind, (self._eye(self.d, self.ridge), [0.0]*self.d))
            Ainv = self._inv(A)
            theta = self._matvec(Ainv, b)
            exploit = self._dot(theta, x)
            # ucb term
            var = self._dot(x, self._matvec(Ainv, x))
            explore = self.alpha * math.sqrt(max(0.0, var))
            scores[kind] = exploit + explore

        ordered = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return ordered, scores, xmap

    # ---------- 업데이트 ----------
    def update(self, stamp: str,
               chosen_kind: str,
               reward_0_1: float,
               src: Tuple[float,float], dst: Tuple[float,float],
               x_used: List[float],
               meta: Dict):
        reward = max(0.0, min(1.0, float(reward_0_1)))
        with psycopg2.connect(self.pg_url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 이벤트 저장
            cur.execute("""
                INSERT INTO bandit_events(stamp, arm, reward, src_lat, src_lon, dst_lat, dst_lon, x, meta)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id;
            """, (stamp, chosen_kind, reward, src[1], src[0], dst[1], dst[0], x_used, json.dumps(meta)))
            # 모델 로드
            cur.execute("SELECT A, b FROM bandit_models WHERE arm=%s FOR UPDATE;", (chosen_kind,))
            row = cur.fetchone()
            if row is None:
                A = self._eye(self.d, self.ridge); b = [0.0]*self.d
                cur.execute("INSERT INTO bandit_models(arm, A, b) VALUES(%s,%s,%s);",
                            (chosen_kind, A, b))
                cur.execute("SELECT A, b FROM bandit_models WHERE arm=%s FOR UPDATE;", (chosen_kind,))
                row = cur.fetchone()

            A = row.get("a") or row.get("A")
            b = row.get("b") or row.get("B")


            # A += x x^T, b += reward * x
            self._add_outer(A, x_used)
            self._add_vec(b, x_used, reward)

            cur.execute("UPDATE bandit_models SET A=%s, b=%s, updated_at=now() WHERE arm=%s;",
                        (A, b, chosen_kind))

# ---- 편의 싱글턴 생성기 ----
_bandit_singletons: Dict[str, ContextBandit] = {}
def get_bandit(pg_url: str, alpha: float = 0.6) -> ContextBandit:
    key = f"{pg_url}|{alpha}"
    if key not in _bandit_singletons:
        _bandit_singletons[key] = ContextBandit(pg_url, alpha=alpha)
    return _bandit_singletons[key]

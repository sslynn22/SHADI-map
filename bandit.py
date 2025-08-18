# bandit.py
import json, os, threading, math, time
from typing import Dict, List
import numpy as np

ACTIONS = ["shortest", "coolest", "shelter"]
STATE_PATH = os.getenv("BANDIT_STATE_PATH", "bandit_state.json")

class LinUCB:
    def __init__(self, d: int, actions: List[str], alpha: float = 1.0):
        self.d = d
        self.actions = actions
        self.alpha = alpha
        self.A = {a: np.eye(d) for a in actions}
        self.b = {a: np.zeros((d, 1)) for a in actions}

    def _theta_Ainv(self, a):
        Ainv = np.linalg.inv(self.A[a])
        theta = Ainv @ self.b[a]
        return theta, Ainv

    def score(self, x: np.ndarray) -> Dict[str, float]:
        # x: (d,1)
        out = {}
        for a in self.actions:
            theta, Ainv = self._theta_Ainv(a)
            mean = float((theta.T @ x)[0, 0])
            unc  = float((x.T @ Ainv @ x)[0, 0]) ** 0.5
            out[a] = mean + self.alpha * unc
        return out

    def update(self, a: str, x: np.ndarray, r: float):
        self.A[a] += x @ x.T
        self.b[a] += r * x

    def dumps(self) -> str:
        return json.dumps({
            "d": self.d, "actions": self.actions, "alpha": self.alpha,
            "A": {k: self.A[k].tolist() for k in self.actions},
            "b": {k: self.b[k].tolist() for k in self.actions},
        })

    @staticmethod
    def loads(s: str):
        js = json.loads(s)
        obj = LinUCB(js["d"], js["actions"], js["alpha"])
        obj.A = {k: np.array(js["A"][k]) for k in obj.actions}
        obj.b = {k: np.array(js["b"][k]) for k in obj.actions}
        return obj

# ---- 단일톤 매니저 (파일에 상태 저장) ----
class BanditManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self._load()

    def _load(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    self.model = LinUCB.loads(f.read())
            except Exception:
                self.model = None
        if self.model is None:
            # 피처 차원 d 정의 (아래 make_x와 반드시 일치)
            self.model = LinUCB(d=10, actions=ACTIONS, alpha=1.0)

    def _save(self):
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self.model.dumps())
        os.replace(tmp, STATE_PATH)

    # ---------- 컨텍스트 → 피처벡터 ----------
    def make_x(self, ctx: dict) -> np.ndarray:
        """
        ctx 예시:
          hour:int, temp_c:float (없으면 생략가능)
          shortest_total_m, coolest_total_m, shelter_total_m
          shade_short, shade_cool, shade_shel
          prefer: 'shortest'|'coolest'|'shelter'|None
        """
        hour = int(ctx.get("hour", 12))
        # 시간 원형 특성
        hs = math.sin(2*math.pi*hour/24.0)
        hc = math.cos(2*math.pi*hour/24.0)

        temp = float(ctx.get("temp_c", 30.0))

        s_m = float(ctx.get("shortest_total_m", 0) or 0.0)
        c_m = float(ctx.get("coolest_total_m", 0) or 0.0)
        h_m = float(ctx.get("shelter_total_m", 0) or 0.0)

        # 상대 우회율(없으면 0)
        def over(x): 
            return (x / s_m - 1.0) if (s_m and x) else 0.0

        # 평균 그늘 비율 (0~1)
        shade_short = float(ctx.get("shade_short", 0.0) or 0.0)
        shade_cool  = float(ctx.get("shade_cool",  0.0) or 0.0)
        shade_shel  = float(ctx.get("shade_shel",  0.0) or 0.0)

        pref = str(ctx.get("prefer") or "").lower()
        p_short = 1.0 if pref == "shortest" else 0.0
        p_cool  = 1.0 if pref == "coolest"  else 0.0
        p_shel  = 1.0 if pref == "shelter"  else 0.0

        vec = np.array([
            hs, hc,                # 0,1
            temp,                  # 2
            over(c_m), over(h_m),  # 3,4  (coolest/shelter의 우회율)
            shade_short,           # 5
            shade_cool,            # 6
            shade_shel,            # 7
            p_cool, p_shel         # 8,9  (선호 힌트; shortest는 기본0, 필요시 추가)
        ], dtype=float).reshape(-1,1)

        # d=10 여야 합니다 (LinUCB 초기화와 동일)
        return vec

    def recommend(self, candidates: List[str], ctx: dict) -> dict:
        with self.lock:
            x = self.make_x(ctx)
            scores = self.model.score(x)
            # 후보만 필터
            scores = {k: scores[k] for k in candidates if k in scores}
            picked = max(scores, key=scores.get)
            return {"picked": picked, "scores": scores}

    def learn(self, action: str, ctx: dict, rating: int):
        # 별점 1~5 → [0,1] 정규화
        r = max(1, min(5, int(rating)))
        reward = (r - 1) / 4.0
        with self.lock:
            x = self.make_x(ctx)
            self.model.update(action, x, reward)
            self._save()

bandit_mgr = BanditManager()

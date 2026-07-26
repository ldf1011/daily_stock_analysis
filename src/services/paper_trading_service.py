"""A-share paper-trading loop. It never sends orders to a broker."""
from __future__ import annotations
import json, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from src.config import get_config
from src.services.alphasift_service import AlphaSiftService, get_dsa_realtime_quote

class PaperTradingService:
    _lock = threading.RLock(); _path = Path("data") / "paper_trading.json"
    def status(self):
        with self._lock:
            state = self._load(); self._mark(state); self._save(state); return self._view(state)
    def set_enabled(self, enabled: bool):
        with self._lock:
            state=self._load(); state["enabled"]=bool(enabled); state["updated_at"]=self._now(); self._save(state); return self._view(state)
    def run_cycle(self):
        with self._lock:
            state=self._load(); self._mark(state); actions=self._sell_risk(state)
            if not state["enabled"]: actions.append({"action":"paused"})
            elif self._equity(state) < state["peak_equity"]*.92: state["enabled"]=False; actions.append({"action":"paused","reason":"max_drawdown"})
            else: actions += self._buy_confirmed(state)
            state["peak_equity"]=max(state["peak_equity"],self._equity(state)); state["last_cycle"]={"at":self._now(),"actions":actions}; self._save(state); return self._view(state)
    def _buy_confirmed(self,state):
        if len(state["positions"])>=5:return [{"action":"skipped","reason":"max_positions"}]
        try: candidates=AlphaSiftService(get_config()).screen(strategy="trend_pullback_confirmed",market="cn",max_results=5).get("candidates") or []
        except Exception as exc:return [{"action":"skipped","reason":f"screen_failed:{type(exc).__name__}"}]
        actions=[]; held={p["code"] for p in state["positions"]}
        for item in candidates:
            if len(state["positions"])>=5: break
            code=str(item.get("code") or "")
            plan=(item.get("raw") or {}).get("trade_plan") or {}; price=self._number(plan.get("entry_confirmation") or item.get("price")); stop=self._number(plan.get("initial_stop_loss")); target=self._number(plan.get("two_r_observation"))
            if not code or code in held or not price or not stop or not target or stop>=price: continue
            equity=self._equity(state); room=equity*.6-self._exposure(state); shares=int(min(equity*.1,room,state["cash"])//price//100)*100
            if shares<100: continue
            state["cash"]=round(state["cash"]-shares*price,2); pos={"code":code,"name":item.get("name") or code,"shares":shares,"entry_price":price,"last_price":price,"stop_loss":stop,"target_price":target,"strategy":"trend_pullback_confirmed","ai_source":"alphasift_llm","opened_at":self._now()}; state["positions"].append(pos); state["orders"].append({"at":self._now(),"side":"buy","code":code,"shares":shares,"price":price,"reason":"ai_confirmed","simulated":True}); actions.append({"action":"simulated_buy","code":code,"shares":shares,"price":price})
        return actions or [{"action":"skipped","reason":"no_confirmed_candidates"}]
    def _sell_risk(self,state):
        actions=[]; keep=[]
        for p in state["positions"]:
            price=p.get("last_price") or p["entry_price"]; reason="stop_loss" if price<=p["stop_loss"] else "target_reached" if price>=p["target_price"] else ""
            if not reason: keep.append(p); continue
            state["cash"]=round(state["cash"]+p["shares"]*price,2); state["orders"].append({"at":self._now(),"side":"sell","code":p["code"],"shares":p["shares"],"price":price,"reason":reason,"simulated":True}); actions.append({"action":"simulated_sell","code":p["code"],"reason":reason})
        state["positions"]=keep; return actions
    def _mark(self,state):
        for p in state["positions"]:
            q=get_dsa_realtime_quote(p["code"]); price=self._number(q.get("price") or q.get("last_price") or q.get("close")); p["last_price"]=price or p["last_price"]
    def _exposure(self,state): return sum(p["shares"]*p.get("last_price",p["entry_price"]) for p in state["positions"])
    def _equity(self,state): return state["cash"]+self._exposure(state)
    def _view(self,state): return {**state,"equity":round(self._equity(state),2),"mode":"paper","broker_orders_enabled":False,"risk":{"initial_cash":300000,"max_position_pct":10,"max_exposure_pct":60,"max_positions":5,"max_drawdown_pct":8}}
    def _load(self):
        try:return json.loads(self._path.read_text())
        except Exception:return {"enabled":False,"cash":300000.0,"peak_equity":300000.0,"positions":[],"orders":[],"last_cycle":None}
    def _save(self,state): self._path.parent.mkdir(parents=True,exist_ok=True); self._path.write_text(json.dumps(state,ensure_ascii=False))
    @staticmethod
    def _number(value:Any):
        try:return float(value) if value is not None and float(value)>0 else None
        except (TypeError,ValueError):return None
    @staticmethod
    def _now():return datetime.now(timezone.utc).isoformat(timespec="seconds")
_service=PaperTradingService()
def get_paper_trading_service():return _service

import os, time, json, re, hashlib, threading, queue, datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

# 可选：docx 解析
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

app = Flask(__name__)

# =============== 环境配置 ===============
LLM_API_BASE = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
LLM_API_KEY  = os.getenv("LLM_API_KEY")  or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or "deepseek-reasoner"

MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS", "8000"))
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT", "1200"))  # API 总超时
SECTION_TIMEOUT     = int(os.getenv("SECTION_TIMEOUT", "600"))   # 单段超时
MAX_TEXT_CHARS      = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS        = int(os.getenv("MAX_JD_CHARS", "10000"))

BRAND_NAME = "Alsos AI Resume"

# =============== 简易 LRU 缓存 ===============
class LRUCache:
    def __init__(self, capacity=200):
        self.cap = capacity
        self.lock = threading.Lock()
        self.data = OrderedDict()
    def get(self, k):
        with self.lock:
            if k in self.data:
                self.data.move_to_end(k)
                return self.data[k]
            return None
    def set(self, k, v):
        with self.lock:
            self.data[k] = v
            self.data.move_to_end(k)
            if len(self.data) > self.cap:
                self.data.popitem(last=False)

cache = LRUCache(200)

# =============== 工具函数 ===============
def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def truncate(s: str, limit: int) -> str:
    return (s or "")[:limit]

def compress_context(s: str, hard_limit: int) -> str:
    if not s: return ""
    return clean_text(s)[:hard_limit]

def is_text_too_short(s: str) -> bool:
    if not s: return True
    en_words = len(re.findall(r"[A-Za-z]+", s))
    return not (len(s) >= 500 or en_words >= 300)

def read_docx(file_storage) -> str:
    if not HAS_DOCX: return ""
    try:
        doc = Document(file_storage)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def _extract_json(text: str):
    if not text:
        raise ValueError("空响应")
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("未找到有效 JSON")
    return json.loads(m.group(0))

def call_llm(payload, json_mode=True):
    if not payload.get("model"):
        raise RuntimeError("缺少模型参数")
    url = LLM_API_BASE.rstrip("/") + "/v1/chat/completions"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API 错误：{r.status_code} {r.text[:250]}")
    return r.json()["choices"][0]["message"]["content"]

def make_payload(messages, model, temperature=0.25, max_tokens=MAX_TOKENS_DEFAULT):
    return {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

# =============== Level 锚点（预分析） ===============
LEVEL_ORDER = {"Junior":0, "Middle":1, "Senior":2, "Executive":3}

SENIOR_KEYWORDS = [
    "副总裁","VP","Vice President","总监","Director","执行总监","Executive Director",
    "负责人","Head","HRD","Chief","首席","CXO","合伙人","Partner","总经理","GM","HRVP","HR Head"
]
MIDDLE_KEYWORDS = ["经理","Manager","资深","Senior","主管","Lead","Leader","负责人"]

def quick_pre_analyze(text: str):
    now = datetime.datetime.utcnow().year
    years = re.findall(r"(19|20)\d{2}", text)
    years = [int(y if len(y)==4 else "2000") for y in years]
    years = [y for y in years if 1980 <= y <= now]
    span = 0
    if years:
        span = max(years) - min(years) + 1
        span = max(1, min(span, 40))

    lower_txt = text.lower()

    if any(kw.lower() in lower_txt for kw in ["vp","vice president","chief","首席","cxo","hrvp","合伙人","partner"]):
        anchor = "Executive"
        reason = "检测到 VP/Chief/合伙人等高阶头衔"
    elif any(kw.lower() in lower_txt for kw in ["总监","director","hrd","head","负责人","executive director"]) or span >= 12:
        anchor = "Senior"
        reason = "总监/HRD/Head 等头衔或年限≥12"
    elif any(kw.lower() in lower_txt for kw in ["经理","manager","资深","senior","主管","lead"]) or span >= 6:
        anchor = "Middle"
        reason = "经理/资深/主管等头衔或年限≥6"
    elif span >= 3:
        anchor = "Middle"
        reason = "年限≥3，保底不低于 Middle"
    else:
        anchor = "Junior"
        reason = "年限较短且未检测到中高阶头衔"

    return {
        "years_span_estimate": span,
        "anchor_min_level": anchor,
        "anchor_reason": reason
    }

# =============== 降级/补位 ===============
def _fallback_from_raw(raw: str, section: str):
    return {"_raw_preview": (raw or "")[:1200], "_note": "解析失败，展示原文片段供参考。"}

def _ensure_nonempty(section: str, obj: dict):
    if section == "summary_highlights":
        obj.setdefault("summary", "（待补充）")
        obj.setdefault("highlights", ["（待补充）", "建议继续完善可量化成果（金额/规模/增速/覆盖度等）"])
    if section == "improvements":
        arr = obj.setdefault("resume_improvements", [])
        if not arr:
            arr.append({"issue":"（待补充）","fix":"补充可执行的改进动作","why":"说明原因与预期影响"})
    if section == "career_diagnosis":
        arr = obj.setdefault("career_diagnosis", [])
        if not arr:
            arr.append({"issue":"（暂未检出明显风险）","risk":"","advice":"保持稳定产出，丰富外显成果"})
    if section == "career_level":
        obj.setdefault("career_level_analysis", {"level":"-","reason":"（待补充）","path":[],"interview_focus":{}})
    if section == "personalized_strategy":
        obj.setdefault("strategy", {"assumptions":"（待补充）","short_term":[],"mid_term":[],"long_term":[]})
    if section == "interview":
        obj.setdefault("interview_handbook", {"answer_logic":[],"interviewer_focus":{},"star_sets":[],"risk_mitigation":[]})
    if section == "ats":
        obj.setdefault("ats", {"enabled": False})
    if section == "salary":
        obj.setdefault("salary_insights", {"title":"","city":"","currency":"CNY","low":0,"mid":0,"high":0,"factors":[],"notes":"模型估算，供参考"})
    return obj

# =============== 页面 ===============
@app.route("/")
def index():
    return render_template("index.html", brand=BRAND_NAME)

@app.route("/extract-text", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "未收到文件"}), 400
    f = request.files["file"]
    name = (f.filename or "").lower()
    if name.endswith(".txt"):
        text = f.read().decode("utf-8", errors="ignore")
    elif name.endswith(".docx") and HAS_DOCX:
        text = read_docx(f)
    else:
        return jsonify({"ok": False, "error": "仅支持 .txt / .docx"}), 400
    return jsonify({"ok": True, "text": clean_text(text)})

# =============== 模型路由（平衡模式） ===============
def model_for(section: str, mode: str):
    # 极速：全 chat
    if mode in ("speed", "fast"):
        return "deepseek-chat", 2800
    # 平衡：只对“需要推理”的块用 reasoner
    if mode == "balanced":
        if section in ["personalized_strategy", "interview", "career_diagnosis", "career_level"]:
            return "deepseek-reasoner", 6000
        return "deepseek-chat", 2800
    # 深度：全 reasoner
    return "deepseek-reasoner", 8000

# =============== 主流程（SSE） ===============
@app.route("/optimize_stream", methods=["POST"])
def optimize_stream():
    t0 = time.time()
    data = request.get_json(force=True) or {}
    resume_text     = compress_context(truncate(data.get("resume_text",""), MAX_TEXT_CHARS), 9000)
    job_description = compress_context(truncate(data.get("job_description",""), MAX_JD_CHARS), 6000)
    target_title    = clean_text(data.get("target_title",""))
    target_location = clean_text(data.get("target_location",""))
    target_industry = clean_text(data.get("target_industry",""))

    mode = (data.get("model") or "").strip().lower()  # depth / balanced / speed

    # 加速：平衡/极速模式时做轻压缩
    if mode in ("speed","fast","balanced"):
        resume_text = compress_context(resume_text, 6500)
        job_description = compress_context(job_description, 4500)

    if not LLM_API_KEY:
        return jsonify({"ok": False, "error": "未配置 LLM API key"}), 500
    if not resume_text:
        return jsonify({"ok": False, "error": "请粘贴简历文本或上传文件"}), 400
    if is_text_too_short(resume_text):
        return jsonify({"ok": False, "error": "简历文本过短（≥500 中文字符或 ≥300 英文词）"}), 400

    pre = quick_pre_analyze(resume_text)

    base_user = {
        "resume_text": resume_text,
        "job_description": job_description,
        "target_title": target_title,
        "target_location": target_location,
        "target_industry": target_industry,
        "pre_analysis": pre
    }
    has_jd = bool(job_description)

    # ---------- Prompts ----------
    prompts = {
        "summary_highlights": f"""你是"{BRAND_NAME}"的资深猎头。仅输出 JSON（中文）：
{{"summary":"<160-220字职业概要>","highlights":["…"]}}
- 必须中文、温和、专业；
- highlights ≥ 8 条；每条 20-60 字；包含场景/动作/结果（数字/规模/指标）；拒绝空话。""",

        "improvements": """仅输出 JSON（中文）：
{"resume_improvements":[{"issue":"","fix":"","why":""}, ...]}
- 至少 10 条；必须“问题点→改进方案→原因解释”三段齐全、具体、可执行；禁止空话。""",

        "career_diagnosis": """仅输出 JSON（中文）：
{"career_diagnosis":[{"issue":"","risk":"","advice":""}, ...]}
- 6–10 条；围绕：跳槽频繁、履历断层、教育一般、单一文化/行业、职责堆砌、管理跨度不足、成果对外可见度低、平台选择偏差；
- “advice”必须是具体动作（如“稳定 2-3 年并拿到 X 类量化成果；对外发布作品集”）。""",

        "career_level": """仅输出 JSON（中文）：
{"career_level_analysis":{
  "level":"Junior|Middle|Senior|Executive",
  "reason":"≤60字",
  "path":["…","…","…"],
  "interview_focus":{"junior":["…"],"middle":["…"],"senior":["…"],"executive":["…"]}
}}
- 你会收到 pre_analysis.anchor_min_level（Level 下限），判定不得低于该下限；
- 若文本出现冲突，请解释原因，但仍不低于该下限；
- 若年限≥12 或出现“总监/HRD/Head/Director/VP/Chief/合伙人”等信号，最低为 Senior；出现 VP/Chief/合伙人且负战略职责，可判 Executive；""",

        "personalized_strategy": """基于简历、JD（如有）、【职业轨迹诊断】、【Level 判定】与【pre_analysis】做“个性化处方”。仅输出 JSON（中文）：
{"strategy":{
  "assumptions":"≤80字",
  "short_term":[
    "现状评估（点名关键风险点）","技能与经验补齐（点名技能/证书/项目/作品集）",
    "风险规避（如跳槽频繁→建议稳定 2-3 年并拿到量化成果；若断层→统一解释模板+复位计划）",
    "小目标设计（3-6 个月，含定量指标）"
  ],
  "mid_term":["目标角色与行业匹配","路径拆解（职责→交付物→量化目标）","关键拐点（带团队/跨部门/对外成果/跨文化）"],
  "long_term":["角色演进的里程碑","平台选择取舍逻辑","风险与弹性（行业周期/转型预案）"],
  "if_job_hopping":["若‘跳槽频繁’成立：给出统一叙事模板（承认→内/外因→复盘→稳定计划）与下一份工作沉淀清单"]
}}
- 条目必须具体可执行；短期应服务于中长期目标；禁止空话。""",

        "interview": """仅输出 JSON（中文）：
{"interview_handbook":{
  "level":"Junior|Middle|Senior|Executive",
  "answer_logic":["…","…","…","…","…"],
  "interviewer_focus":{"HR":["…"],"hiring_manager":["…"],"executive":["…"]},
  "star_sets":[
    {"project_title":"","question":"","how_to_answer":["…","…","…"]},
    {"project_title":"","question":"","how_to_answer":["…","…","…"]},
    {"project_title":"","question":"","how_to_answer":["…","…","…"]}
  ],
  "risk_mitigation":[
    "跳槽频繁：统一解释模板……",
    "履历断层：正面叙述……+ 复位动作……",
    "教育一般：以作品集/成果弥补……"
  ]
}}
- 仅生成候选人对应 level 的手册；各列表至少 5/5/3。""",

        "ats": """仅输出 JSON（中文）：
{"ats":{"enabled":true,"total_score":0,"sub_scores":{"skills":0,"experience":0,"education":0,"keywords":0},
 "reasons":{"skills":["…"],"experience":["…"],"education":["…"],"keywords":["…"]},
 "gap_keywords":["…"],"improvement_advice":["…"]}}
- reasons 各 3–5 条；gap_keywords ≥ 10；improvement_advice ≥ 6，尽量贴 JD。""",

        "salary": """仅输出 JSON（中文）：
{"salary_insights":{"title":"","city":"","currency":"CNY","low":0,"mid":0,"high":0,"factors":["…"],"notes":"模型估算，供参考"}}
- low < mid < high；给 5 个影响因子（体量/区域/热度/作品集/是否管理岗等）。"""
    }

    phase1 = ["summary_highlights","improvements","career_diagnosis","career_level"]
    phase2 = ["personalized_strategy","interview","salary"]
    if has_jd: phase2.append("ats")

    qout = queue.Queue()
    phase1_results = {}

    def run_section(section, extra_ctx=None):
        ck_raw = {"base": base_user, "sec": section, "mode": mode, "extra": extra_ctx}
        ck = hashlib.sha256(json.dumps(ck_raw, ensure_ascii=False).encode()).hexdigest()
        cached = cache.get(ck)
        if cached is not None:
            qout.put({"section": section, "data": cached}); return

        user_payload = dict(base_user)
        if extra_ctx: user_payload["prior_findings"] = extra_ctx

        messages=[{"role":"system","content":prompts[section]},
                  {"role":"user","content":json.dumps(user_payload,ensure_ascii=False)}]

        sec_model, sec_tokens = model_for(section, mode)
        payload = make_payload(messages, model=sec_model, max_tokens=sec_tokens)

        try:
            with ThreadPoolExecutor(max_workers=1) as inner:
                fut = inner.submit(lambda: call_llm(payload, json_mode=True))
                raw = fut.result(timeout=SECTION_TIMEOUT)

            try:
                obj = _extract_json(raw)
            except Exception:
                obj = _fallback_from_raw(raw, section)

            obj = _ensure_nonempty(section, obj)

            # 低于锚点时自动纠偏
            if section == "career_level" and isinstance(obj.get("career_level_analysis"), dict):
                level = obj["career_level_analysis"].get("level","-")
                a_min = base_user["pre_analysis"]["anchor_min_level"]
                order = LEVEL_ORDER
                if level in order and order.get(level,0) < order.get(a_min,0):
                    obj["career_level_analysis"]["reason"] = (obj["career_level_analysis"].get("reason","") +
                        f"（系统基于年限/头衔纠偏：最低不低于 {a_min}）")
                    obj["career_level_analysis"]["level"] = a_min

            cache.set(ck, obj)
            qout.put({"section": section, "data": obj})
        except Exception as e:
            qout.put({"section": section, "error": str(e)})

    def streamer():
        yield "retry: 1500\n"
        yield 'data: {"section":"boot","data":{"msg":"引擎已启动，正在读取与拆解你的简历…"}}\n\n'

        # Phase 1
        with ThreadPoolExecutor(max_workers=min(4,len(phase1))) as ex1:
            for sec in phase1: ex1.submit(run_section, sec)

        need1=set(phase1); last_beat=time.time()
        while need1:
            if time.time()-last_beat>10: yield ": keep-alive\n\n"; last_beat=time.time()
            item=qout.get()
            yield f"data: {json.dumps(item,ensure_ascii=False)}\n\n"
            need1.discard(item["section"])
            if "error" not in item and item["section"] in ("career_diagnosis","career_level"):
                phase1_results[item["section"]] = item["data"]

        # Phase 2
        extra_ctx={"diagnosis":phase1_results.get("career_diagnosis",{}),
                   "level":phase1_results.get("career_level",{})}
        with ThreadPoolExecutor(max_workers=min(4,len(phase2))) as ex2:
            for sec in phase2: ex2.submit(run_section, sec, extra_ctx)

        need2=set(phase2)
        while need2:
            if time.time()-last_beat>10: yield ": keep-alive\n\n"; last_beat=time.time()
            item=qout.get()
            yield f"data: {json.dumps(item,ensure_ascii=False)}\n\n"
            need2.discard(item["section"])

        meta={"elapsed_ms":int((time.time()-t0)*1000),"mode":mode,"has_jd":has_jd,"pre_analysis":pre}
        yield f"data: {json.dumps({'section':'done','data':{'meta':meta}},ensure_ascii=False)}\n\n"

    headers={"Content-Type":"text/event-stream; charset=utf-8",
             "Cache-Control":"no-cache","X-Accel-Buffering":"no"}
    return Response(stream_with_context(streamer()), headers=headers)

@app.route("/healthz")
def healthz(): return "ok",200

if __name__=="__main__":
    port=int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0",port=port,debug=False)

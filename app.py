import os, time, json, re, hashlib, threading, queue, datetime, uuid
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

app = Flask(__name__)

# ------------ 环境 ------------
LLM_API_BASE = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
LLM_API_KEY  = os.getenv("LLM_API_KEY")  or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or "deepseek-reasoner"

MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS", "8000"))
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT", "1200"))
SECTION_TIMEOUT     = int(os.getenv("SECTION_TIMEOUT", "600"))
MAX_TEXT_CHARS      = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS        = int(os.getenv("MAX_JD_CHARS", "10000"))

BRAND_NAME = "Alsos AI Resume"

# ------------ 简易缓存 ------------
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

# ------------ 工具 ------------
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

# ------------ 预分析：年限/跳槽/教育/管理/ JD 诉求 ------------
LEVEL_ORDER = {"Junior":0, "Middle":1, "Senior":2, "Executive":3}

def _guess_year_span(text: str):
    now = datetime.datetime.utcnow().year
    yrs = [int(y) for y in re.findall(r"(19|20)\d{2}", text)]
    yrs = [y if y>1900 else 2000 for y in yrs]
    yrs = [y for y in yrs if 1980 <= y <= now]
    if not yrs: return 0
    return max(1, min(40, max(yrs)-min(yrs)+1))

def _hop_signal(text: str):
    # 粗略：统计形如 2018-2020 / 2018.09-2020.06 / 2018/2020 这种区段数
    ranges = re.findall(r"(20\d{2})(?:[./-]\d{1,2})?\s*[-–~]\s*(20\d{2})", text)
    total = len(ranges)
    # 近5年切片（粗略把包含近5年的区段计数）
    now = datetime.datetime.utcnow().year
    recent5 = sum(1 for a,b in ranges if int(b) >= now-5)
    longest = 0
    for a,b in ranges:
        try:
            longest = max(longest, int(b)-int(a)+1)
        except: pass
    # 判定：近5年>=3 次 → 可疑；最长任期>=3 年 & 近5年<=1 → 稳定加分
    hop_suspect = recent5 >= 3
    stability_plus = (longest >= 3 and recent5 <= 1) or (total <= 2 and _guess_year_span(text)>=6)
    return {"total_ranges": total, "recent5": recent5, "longest_span": longest,
            "hop_suspect": hop_suspect, "stability_plus": stability_plus}

def _edu_signal(text: str):
    t = text.lower()
    elite = any(k in t for k in ["985","211","双一流","qs","top 100","top100","top 200","top200"])
    edu_level = "bachelor"
    if re.search(r"博士|phd", text, re.I): edu_level="phd"
    elif re.search(r"硕士|master", text, re.I): edu_level="master"
    return {"elite": elite, "edu_level": edu_level}

def _mgmt_signal(text: str):
    t = text.lower()
    num = re.search(r"(?:带领|管理|负责人|lead|led|managed|manager of|direct reports)\D*(\d{1,3})", text, re.I)
    mgmt = bool(re.search(r"管理|带领|团队|leader|lead|managed|head of|负责人", text, re.I))
    span = int(num.group(1)) if num else 0
    return {"mgmt": mgmt, "mgmt_span": span, "mgmt_suspect": (not mgmt or span==0)}

def _jd_require(jd: str):
    if not jd: return {"need_phd": False, "need_master": False, "need_elite": False, "elite_floor": None}
    need_phd   = bool(re.search(r"博士|phd", jd, re.I))
    need_master= bool(re.search(r"硕士|master", jd, re.I))
    elite = bool(re.search(r"985|211|双一流|qs ?(top)? ?(100|200)|top ?(100|200)", jd, re.I))
    floor = None
    m = re.search(r"qs ?top ?(100|200)", jd, re.I)
    if m: floor = int(m.group(1))
    return {"need_phd":need_phd, "need_master":need_master, "need_elite":elite, "elite_floor":floor}

def quick_pre_analyze(resume_text: str, jd_text: str):
    span = _guess_year_span(resume_text)
    hop = _hop_signal(resume_text)
    edu = _edu_signal(resume_text)
    mg  = _mgmt_signal(resume_text)
    jd  = _jd_require(jd_text)
    # level anchor（更稳健）
    if edu["edu_level"]=="phd" or re.search(r"vp|chief|合伙人|hrvp|总监|director|head|hrd", resume_text, re.I):
        anchor="Senior"
    elif span>=6:
        anchor="Middle"
    elif span>=3:
        anchor="Middle"
    else:
        anchor="Junior"
    # 若出现 VP / Chief / 合伙人，直接 Executive 候选
    if re.search(r"vp|vice president|chief|cxo|合伙人|partner", resume_text, re.I):
        anchor="Executive"
    return {
        "years_span_estimate": span,
        "anchor_min_level": anchor,
        "anchor_reason": "基于年限/头衔快速锚定",
        "hop": hop,
        "edu": edu,
        "mgmt": mg,
        "jd": jd
    }

# ------------ 降级 ------------
def _fallback_from_raw(raw: str, section: str):
    return {"_raw_preview": (raw or "")[:1000], "_note": "解析失败，展示原文片段供参考。"}

def _ensure_nonempty(section: str, obj: dict):
    if section == "summary_highlights":
        obj.setdefault("summary", "（待补充）")
        obj.setdefault("highlights", ["（待补充）"])
    if section == "improvements":
        arr = obj.setdefault("resume_improvements", [])
        if not arr: arr.append({"issue":"（待补充）","fix":"补充可执行动作","why":"补充原因"})
    if section == "career_diagnosis":
        obj.setdefault("career_diagnosis", [])
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

# ------------ 页面 ------------
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

# ------------ 模型路由（平衡提速） ------------
def model_for(section: str, mode: str):
    if mode in ("speed","fast"): return "deepseek-chat", 2600
    if mode == "balanced":
        if section in ["personalized_strategy","interview","career_diagnosis","career_level"]:
            return "deepseek-reasoner", 5200
        return "deepseek-chat", 2400
    return "deepseek-reasoner", 7200

# ------------ 主流程（SSE 流式） ------------
@app.route("/optimize_stream", methods=["POST"])
def optimize_stream():
    t0 = time.time()
    data = request.get_json(force=True) or {}
    resume_text     = compress_context(truncate(data.get("resume_text",""), MAX_TEXT_CHARS), 9000)
    job_description = compress_context(truncate(data.get("job_description",""), MAX_JD_CHARS), 6000)
    target_title    = clean_text(data.get("target_title",""))
    target_location = clean_text(data.get("target_location",""))
    target_industry = clean_text(data.get("target_industry",""))
    mode = (data.get("model") or "").strip().lower()

    if mode in ("speed","fast","balanced"):
        resume_text = compress_context(resume_text, 6500)
        job_description = compress_context(job_description, 4500)

    if not LLM_API_KEY:
        return jsonify({"ok": False, "error": "未配置 LLM API key"}), 500
    if not resume_text:
        return jsonify({"ok": False, "error": "请粘贴简历文本或上传文件"}), 400
    if is_text_too_short(resume_text):
        return jsonify({"ok": False, "error": "简历文本过短（≥500 中文字符或 ≥300 英文词）"}), 400

    pre = quick_pre_analyze(resume_text, job_description)
    has_jd = bool(job_description)

    base_user = {
        "resume_text": resume_text,
        "job_description": job_description,
        "target_title": target_title,
        "target_location": target_location,
        "target_industry": target_industry,
        "pre_analysis": pre
    }

    # ------------ 更克制的 Prompts（限长 + 条件输出） ------------
    prompts = {
        "summary_highlights": f"""你是"{BRAND_NAME}"的资深猎头。输出紧凑 JSON（中文）：
{{"summary":"<120-160字职业概要>","highlights":["…"]}}
- 语气温和专业，禁止空话；
- highlights ≤ 6 条，每条 16-36 字，含“动作+结果(数值/规模)”；
- 若 pre_analysis.hop.stability_plus 为真，summary 中明确“稳定性强”；""",

        "improvements": """输出紧凑 JSON（中文）：
{"resume_improvements":[{"issue":"","fix":"","why":""}, ...]}
- 仅给“最重要的 Top6”改进点；每项为何≤30字，fix 可执行；""",

        "career_diagnosis": """仅在证据成立时输出。紧凑 JSON（中文）：
{"career_diagnosis":[{"issue":"","risk":"","advice":""}, ...]}
规则：
- “跳槽频繁”仅当 pre_analysis.hop.hop_suspect 为真（近5年≥3次）才出现；建议先稳定 2-3 年并拿到量化成果；
- 若 pre_analysis.hop.stability_plus 为真，加入“稳定性强”的正向项；
- 教育：若 JD 要求博士/QS/985/211 且候选达不到→提示“可能存在差距”；无 JD 要求时，若非名校仅表述“可能竞争力一般/不足”；若名校/研究生→表述“教育背景具备竞争力”；
- 管理跨度：仅当 pre_analysis.mgmt.mgmt_suspect 为真时出现；否则不提；
- 每条 ≤ 40 字，总数 3-6 条；""",

        "career_level": """输出 JSON（中文）：
{"career_level_analysis":{
  "level":"Junior|Middle|Senior|Executive",
  "reason":"≤40字",
  "path":["…","…","…"],
  "interview_focus":{"junior":["…"],"middle":["…"],"senior":["…"],"executive":["…"]}
}}
- Level 不得低于 pre_analysis.anchor_min_level；
- 若稳定性强或管理跨度小，reason 简短点出；
- path ≤3 条、每条 ≤ 20 字；interview_focus 各 ≤4 条；""",

        "personalized_strategy": """输出 JSON（中文）：
{"strategy":{
  "assumptions":"≤60字",
  "short_term":["现状评估(命中项)","技能/证书补齐","风险规避(若跳槽频繁给统一叙述模板)","3-6个月小目标(量化)"],
  "mid_term":["目标角色/行业匹配","路径拆解(职责→交付物→指标)","关键拐点(带团队/跨部门/作品集)"],
  "long_term":["角色演进里程碑","平台选择逻辑","风险与弹性(行业周期/转型预案)"],
  "if_job_hopping":["仅当 pre_analysis.hop.hop_suspect 为真时输出：统一叙述模板+复位计划"]
}}
- 每列 3-5 条，禁止空话，条目≤24字；""",

        "interview": """输出 JSON（中文）：
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
    "若跳槽频繁：统一解释模板……",
    "若履历断层：正面叙述……+ 复位动作……",
    "若教育不达标：以成果/作品集弥补……"
  ]
}}
- 仅生成对应 level；每条尽量 ≤ 26 字；""",

        "ats": """输出 JSON（中文）：
{"ats":{"enabled":true,"total_score":0,"sub_scores":{"skills":0,"experience":0,"education":0,"keywords":0},
 "reasons":{"skills":["…"],"experience":["…"],"education":["…"],"keywords":["…"]},
 "gap_keywords":["…"],"improvement_advice":["…"]}}
- reasons 各 ≤3 条；gap_keywords ≤10；advice ≤6 条；""",

        "salary": """输出 JSON（中文）：
{"salary_insights":{"title":"","city":"","currency":"CNY","low":0,"mid":0,"high":0,"factors":["…"],"notes":"模型估算，供参考"}}
- low < mid < high；factors ≤5 条；"""
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

            # 低于锚点纠偏
            if section == "career_level" and isinstance(obj.get("career_level_analysis"), dict):
                level = obj["career_level_analysis"].get("level","-")
                a_min = base_user["pre_analysis"]["anchor_min_level"]
                order = LEVEL_ORDER
                if level in order and order.get(level,0) < order.get(a_min,0):
                    obj["career_level_analysis"]["reason"] = (obj["career_level_analysis"].get("reason","") +
                        f"（锚点下限：{a_min}）")
                    obj["career_level_analysis"]["level"] = a_min

            cache.set(ck, obj)
            qout.put({"section": section, "data": obj})
        except Exception as e:
            qout.put({"section": section, "error": str(e)})

    def streamer():
        yield "retry: 1500\n"
        yield 'data: {"section":"boot","data":{"msg":"引擎已启动，正在读取与你的简历做对齐…"}}\n\n'

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

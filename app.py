import os, time, json, re, hashlib, threading, queue
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

# ======= 可选：.docx 文本抽取 =======
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

app = Flask(__name__)

# ======= 环境变量（兼容多命名） =======
LLM_API_BASE = (
    os.getenv("LLM_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.deepseek.com"
)
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
# ✅ 默认思考模式
DEFAULT_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("MODEL_NAME")
    or "deepseek-reasoner"
)

# ======= 性能参数 =======
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS", "10000"))
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT", "300"))
MAX_TEXT_CHARS      = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS        = int(os.getenv("MAX_JD_CHARS", "10000"))

BRAND_NAME = "Alsos AI Resume"

# ======= 简易 LRU 缓存（单实例） =======
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

cache = LRUCache(capacity=200)

# ======= 工具 =======
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
    s = clean_text(s)
    return s[:hard_limit]

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
    # 去掉 ```json 围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("未找到有效 JSON")
    return json.loads(m.group(0))

def call_llm(payload, json_mode=True):
    """同步调用；由线程池并发调度"""
    model = payload.pop("model")
    url = LLM_API_BASE.rstrip("/") + "/v1/chat/completions"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API 错误：{r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]

def make_payload(messages, model, temperature=0.25, max_tokens=MAX_TOKENS_DEFAULT):
    return {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

# ======= 页面 =======
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

# ======= 并发流式接口 =======
@app.route("/optimize_stream", methods=["POST"])
def optimize_stream():
    t0 = time.time()
    data = request.get_json(force=True) or {}
    resume_text     = compress_context(truncate(data.get("resume_text",""), MAX_TEXT_CHARS), 9000)
    target_title    = clean_text(data.get("target_title",""))
    target_location = clean_text(data.get("target_location",""))
    target_industry = clean_text(data.get("target_industry",""))
    job_description = compress_context(truncate(data.get("job_description",""), MAX_JD_CHARS), 6000)

    # 仍保留可选开关：speed=chat / depth=reasoner；未传则默认 reasoner
    model_choice = (data.get("model") or "").strip().lower()
    if model_choice in ("speed","fast"):
        model = "deepseek-chat"; per_call_tokens = 3000
    elif model_choice in ("depth","reasoner"):
        model = "deepseek-reasoner"; per_call_tokens = 12000
    else:
        model = DEFAULT_MODEL
        per_call_tokens = 12000 if "reasoner" in model else 6000

    if not resume_text:
        return jsonify({"ok": False, "error": "请粘贴简历文本或上传文件"}), 400
    if is_text_too_short(resume_text):
        return jsonify({"ok": False, "error": "简历文本过短（≥500 中文字符或 ≥300 英文词）"}), 400

    has_jd = bool(job_description)
    base_user = {
        "resume_text": resume_text,
        "job_description": job_description if has_jd else "",
        "target_title": target_title,
        "target_location": target_location,
        "target_industry": target_industry
    }

    # 缓存 key
    def key_for(section):
        raw = json.dumps(base_user, ensure_ascii=False) + f"|{section}|{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # prompts（含新模块）
    prompts = {
        "summary_highlights": f"""你是"{BRAND_NAME}"。仅生成 JSON：
{{"summary":"<160-220字职业概要>","highlights":["…"]}}
- highlights≥8，完整句子，含数字/规模/结果（20-60字/条）。""",

        "improvements": """仅生成 JSON：
{"resume_improvements":[{"issue":"","fix":"","why":""}...]}
- ≥10条；结构“问题点→改进方案→原因解释”；可执行且具体；禁止空话。""",

        "career_diagnosis": """仅生成 JSON：
{"career_diagnosis":[
  {"issue":"跳槽频繁","risk":"稳定性被质疑（平均不到24个月）","advice":"在每段经历中补充沉淀成果与成长逻辑；未来求职优先选择能持续3年以上的平台并说明换岗原因"},
  {"issue":"","risk":"","advice":""}
]}
- 生成≥6–10条，覆盖：跳槽频繁、履历断层、教育背景一般、单一文化/单一行业、职责堆砌无结果、管理跨度不足、对外成果少等；
- 每条必须有 issue/risk/advice；直击 HR/老板关注点；用具体语言。""",

        "career_level": """仅生成 JSON：
{"career_level_analysis":{
  "level":"Junior|Middle|Senior|Executive",
  "reason":"基于年限/头衔/团队规模/业绩的判定理由（≤60字）",
  "path":[
    "下一步目标与达成条件（项目规模/跨部门/管理人数/营收指标等）",
    "平台建议（大厂/独角兽/外企/MNC/本地龙头/创业公司）与选择标准",
    "短板补强清单（证书/作品集/跨文化/资本沟通等）"
  ],
  "interview_focus":{
    "junior":["学习力/执行力/项目参与度/作品集要点","…"],
    "middle":["独立负责/跨部门协作/量化成果/带新人","…"],
    "senior":["领导力/业务结果/战略理解/组织搭建","…"],
    "executive":["战略眼光/资本效率/平台匹配/治理与风险","…"]
  }
}}
- level 必须从简历推断；四档关注点各给2–4条。""",

        "keywords_career": """仅生成 JSON：
{"keywords":["…"],"career_suggestions":{"short_term":["…"],"mid_term":["…"],"long_term":["…"]}}
- keywords≥12（行业/技能/工具术语，标准表达）
- 短/中/长期分别≥5/≥5/≥4：含平台类型（大厂/独角兽/外企/本地龙头/咨询）、行动步骤、衡量指标；对教育/履历弱势给补强路径。""",

        "interview": """仅生成 JSON：
{"interview_handbook":{
 "answer_logic":["…"],"level_differences":{"junior":["…"],"senior":["…"]},
 "interviewer_focus":{"HR":["…"],"hiring_manager":["…"],"executive":["…"]},
 "star_sets":[
   {"project_title":"","question":"","how_to_answer":["…"]},
   {"project_title":"","question":"","how_to_answer":["…"]},
   {"project_title":"","question":"","how_to_answer":["…"]}
 ]
}}
- answer_logic≥6；junior/senior各≥5；HR/负责人/老板各≥5；每套STAR含3–5步。""",

        "ats": """仅生成 JSON：
{"ats":{"enabled":true,"total_score":0,"sub_scores":{"skills":0,"experience":0,"education":0,"keywords":0},
 "reasons":{"skills":["…"],"experience":["…"],"education":["…"],"keywords":["…"]},
 "gap_keywords":["…"],"improvement_advice":["…"]}}
- reasons各3–5条；gap_keywords≥10；improvement_advice≥6（可逐条映射 JD）。""",

        "salary": """仅生成 JSON：
{"salary_insights":{"title":"","city":"","currency":"CNY","low":0,"mid":0,"high":0,"factors":["…"],"notes":"模型估算，供参考"}}
- low<mid<high；给5个影响因子（公司体量/区域/行业热度/作品集质量/是否管理岗等）。"""
    }

    # 任务列表（并发执行）
    sections = [
        "summary_highlights",
        "improvements",
        "career_diagnosis",
        "career_level",
        "keywords_career",
        "interview",
        "salary"
    ]
    if has_jd:
        sections.append("ats")

    qout = queue.Queue()

    def run_section(section):
        ck = key_for(section)
        cached = cache.get(ck)
        if cached is not None:
            qout.put({"section": section, "data": cached, "cached": True})
            return
        msgs = [
            {"role": "system", "content": prompts[section]},
            {"role": "user", "content": json.dumps(base_user, ensure_ascii=False)}
        ]
        payload = make_payload(msgs, model=model, temperature=0.25, max_tokens=per_call_tokens)
        try:
            raw = call_llm(payload, json_mode=True)
            obj = _extract_json(raw)
            if section == "ats" and not has_jd:
                obj = {"ats": {"enabled": False}}
            cache.set(ck, obj)
            qout.put({"section": section, "data": obj})
        except Exception as e:
            qout.put({"section": section, "error": str(e)})

    with ThreadPoolExecutor(max_workers=min(7, len(sections))) as ex:
        for sec in sections:
            ex.submit(run_section, sec)

        def streamer():
            yield "retry: 1500\n"
            finished, total = 0, len(sections)
            while finished < total:
                item = qout.get()
                if "error" in item:
                    payload = {"section": item["section"], "error": item["error"]}
                else:
                    payload = {"section": item["section"], "data": item["data"]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                finished += 1
            meta = {"elapsed_ms": int((time.time()-t0)*1000), "model_alias": BRAND_NAME, "has_jd": has_jd}
            yield f"data: {json.dumps({'section':'done','data':{'meta':meta}}, ensure_ascii=False)}\n\n"

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
        return Response(stream_with_context(streamer()), headers=headers)

# 健康探针
@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

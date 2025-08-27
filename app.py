import os, time, json, re, hashlib, threading, queue
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

# ===== 环境变量 =====
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
DEFAULT_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("MODEL_NAME")
    or "deepseek-reasoner"
)

MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS", "8000"))
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT", "1200"))
MAX_TEXT_CHARS      = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS        = int(os.getenv("MAX_JD_CHARS", "10000"))
SECTION_TIMEOUT     = int(os.getenv("SECTION_TIMEOUT", "600"))

BRAND_NAME = "Alsos AI Resume"

# ===== 简单缓存 =====
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

# ===== 工具 =====
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
    if not payload.get("model"):
        raise RuntimeError("缺少模型参数")
    url = LLM_API_BASE.rstrip("/") + "/v1/chat/completions"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API 错误：{r.status_code} {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]

def make_payload(messages, model, temperature=0.25, max_tokens=MAX_TOKENS_DEFAULT):
    return {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

# ===== 页面 =====
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

# ===== 流式优化 =====
@app.route("/optimize_stream", methods=["POST"])
def optimize_stream():
    t0 = time.time()
    data = request.get_json(force=True) or {}
    resume_text     = compress_context(truncate(data.get("resume_text",""), MAX_TEXT_CHARS), 9000)
    job_description = compress_context(truncate(data.get("job_description",""), MAX_JD_CHARS), 6000)
    target_title    = clean_text(data.get("target_title",""))
    target_location = clean_text(data.get("target_location",""))
    target_industry = clean_text(data.get("target_industry",""))

    model_choice = (data.get("model") or "").strip().lower()
    if model_choice in ("speed","fast"):
        model, per_call_tokens = "deepseek-chat", 3000
    else:
        model, per_call_tokens = "deepseek-reasoner", 8000

    if not resume_text:
        return jsonify({"ok": False, "error": "请粘贴简历文本"}), 400
    if is_text_too_short(resume_text):
        return jsonify({"ok": False, "error": "简历过短"}), 400

    base_user = {
        "resume_text": resume_text,
        "job_description": job_description,
        "target_title": target_title,
        "target_location": target_location,
        "target_industry": target_industry
    }
    has_jd = bool(job_description)

    # ===== prompts（简化版：全中文+温度） =====
    prompts = {
        "summary_highlights": f"""你是资深猎头。输出 JSON（中文）：{{"summary":"","highlights":[]}}""",
        "improvements": """你是简历优化专家。输出 JSON（中文）：{"resume_improvements":[...] }""",
        "career_diagnosis": """你是猎头，诊断职业轨迹风险（跳槽频繁/断层/学历一般/单一文化等），输出 JSON（中文）。""",
        "career_level": """判定 Level (Junior/Middle/Senior/Executive)，给理由+发展路径，输出 JSON（中文）。""",
        "personalized_strategy": """结合诊断+Level，给短/中/长期策略，针对跳槽频繁等风险给专项建议。输出 JSON（中文）。""",
        "interview": """只给候选人对应 Level 的面试手册，结合诊断风险，输出 JSON（中文）。""",
        "ats": """做 ATS 匹配分析，输出 JSON（中文）。""",
        "salary": """给薪酬估算区间和影响因子，输出 JSON（中文）。"""
    }

    phase1 = ["summary_highlights","improvements","career_diagnosis","career_level"]
    phase2 = ["personalized_strategy","interview","salary"]
    if has_jd: phase2.append("ats")

    qout = queue.Queue()
    phase1_results = {}

    def run_section(section, extra_ctx=None):
        ck_raw = {"base": base_user, "sec": section, "model": model, "extra": extra_ctx}
        ck = hashlib.sha256(json.dumps(ck_raw,ensure_ascii=False).encode()).hexdigest()
        cached = cache.get(ck)
        if cached is not None:
            qout.put({"section": section, "data": cached}); return
        user_payload = dict(base_user)
        if extra_ctx: user_payload["prior_findings"] = extra_ctx
        msgs=[{"role":"system","content":prompts[section]},
              {"role":"user","content":json.dumps(user_payload,ensure_ascii=False)}]
        payload = make_payload(msgs, model=model, max_tokens=per_call_tokens)
        try:
            with ThreadPoolExecutor(max_workers=1) as inner:
                fut = inner.submit(lambda:_extract_json(call_llm(payload)))
                obj=fut.result(timeout=SECTION_TIMEOUT)
            cache.set(ck,obj)
            qout.put({"section":section,"data":obj})
        except Exception as e:
            qout.put({"section":section,"error":str(e)})

    def streamer():
        yield "retry: 1500\n"
        # phase1
        with ThreadPoolExecutor(max_workers=4) as ex1:
            for sec in phase1: ex1.submit(run_section, sec)
        need1=set(phase1); last_beat=time.time()
        while need1:
            if time.time()-last_beat>10:
                yield ": keep-alive\n\n"; last_beat=time.time()
            item=qout.get()
            if "error" in item:
                yield f"data:{json.dumps(item,ensure_ascii=False)}\n\n"
            else:
                phase1_results[item["section"]]=item["data"]
                yield f"data:{json.dumps(item,ensure_ascii=False)}\n\n"
            need1.discard(item["section"])
        # phase2
        extra_ctx={
            "diagnosis":phase1_results.get("career_diagnosis",{}),
            "level":phase1_results.get("career_level",{})
        }
        with ThreadPoolExecutor(max_workers=4) as ex2:
            for sec in phase2: ex2.submit(run_section, sec, extra_ctx)
        need2=set(phase2)
        while need2:
            if time.time()-last_beat>10:
                yield ": keep-alive\n\n"; last_beat=time.time()
            item=qout.get()
            yield f"data:{json.dumps(item,ensure_ascii=False)}\n\n"
            need2.discard(item["section"])
        meta={"elapsed_ms":int((time.time()-t0)*1000),"model":model}
        yield f"data:{json.dumps({'section':'done','data':{'meta':meta}},ensure_ascii=False)}\n\n"

    return Response(stream_with_context(streamer()), headers={"Content-Type":"text/event-stream"})

@app.route("/healthz")
def healthz():
    return "ok",200

if __name__=="__main__":
    port=int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0",port=port,debug=False)

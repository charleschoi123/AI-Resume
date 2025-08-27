import os, time, json, re, hashlib, threading, queue
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

# 可选的 docx 解析
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

app = Flask(__name__)

# ==================== 环境变量 ====================
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
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT", "1200"))  # 整体 API 超时
SECTION_TIMEOUT     = int(os.getenv("SECTION_TIMEOUT", "600"))   # 单段超时
MAX_TEXT_CHARS      = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS        = int(os.getenv("MAX_JD_CHARS", "10000"))

BRAND_NAME = "Alsos AI Resume"

# ==================== 简易 LRU 缓存 ====================
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

# ==================== 工具函数 ====================
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

# 降级与最小补位
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

# ==================== 页面 ====================
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

# ==================== 流式优化（SSE） ====================
@app.route("/optimize_stream", methods=["POST"])
def optimize_stream():
    t0 = time.time()
    data = request.get_json(force=True) or {}
    resume_text     = compress_context(truncate(data.get("resume_text",""), MAX_TEXT_CHARS), 9000)
    job_description = compress_context(truncate(data.get("job_description",""), MAX_JD_CHARS), 6000)
    target_title    = clean_text(data.get("target_title",""))
    target_location = clean_text(data.get("target_location",""))
    target_industry = clean_text(data.get("target_industry",""))

    # 模型选择：默认思考模式
    model_choice = (data.get("model") or "").strip().lower()
    if model_choice in ("speed","fast"):
        model, per_call_tokens = "deepseek-chat", 3000
    else:
        model, per_call_tokens = "deepseek-reasoner", 8000

    if not LLM_API_KEY:
        return jsonify({"ok": False, "error": "未配置 LLM API key"}), 500
    if not resume_text:
        return jsonify({"ok": False, "error": "请粘贴简历文本或上传文件"}), 400
    if is_text_too_short(resume_text):
        return jsonify({"ok": False, "error": "简历文本过短（≥500 中文字符或 ≥300 英文词）"}), 400

    base_user = {
        "resume_text": resume_text,
        "job_description": job_description,
        "target_title": target_title,
        "target_location": target_location,
        "target_industry": target_industry
    }
    has_jd = bool(job_description)

    # —— 强化后的 Prompts（全中文、数量/结构/禁止空话）
    prompts = {
        "summary_highlights": f"""你是"{BRAND_NAME}"的资深猎头。仅输出 JSON（中文）：
{{"summary":"<160-220字职业概要>","highlights":["…"]}}
- 必须写中文，口吻温和、专业、真诚；
- highlights ≥ 8 条；每条 20-60 字；包含场景/动作/结果（有数字/规模/指标）；拒绝口号与空话。""",

        "improvements": """仅输出 JSON（中文）：
{"resume_improvements":[{"issue":"","fix":"","why":""}, ...]}
- 至少 10 条；“问题点→改进方案→原因解释”三段必须齐全且具体，可执行、可验证；严禁“完善表述/强化逻辑”等空话。""",

        "career_diagnosis": """仅输出 JSON（中文）：
{"career_diagnosis":[{"issue":"","risk":"","advice":""}, ...]}
- 6–10 条；从以下高频风险中筛选契合者并给出针对性建议：跳槽频繁、履历断层、教育背景一般、单一文化/单一行业、职责堆砌无结果、管理跨度不足、对外成果少、平台选择偏差；
- “advice”必须是具体动作（如“在现公司稳定 2-3 年并拿到 X 类可量化成果；对外发布作品集”）。""",

        "career_level": """仅输出 JSON（中文）：
{"career_level_analysis":{
  "level":"Junior|Middle|Senior|Executive",
  "reason":"≤60字",
  "path":["…","…","…"],
  "interview_focus":{"junior":["…"],"middle":["…"],"senior":["…"],"executive":["…"]}
}}
- level 必须从年限/头衔/团队/结果判断；path 至少 3 条；各 level 的面试关注点各 ≥ 3 条。""",

        "personalized_strategy": """基于候选人简历、JD（如有）、【职业轨迹诊断】与【Level 判定】做“个性化处方”。仅输出 JSON（中文）：
{"strategy":{
  "assumptions":"≤80字",
  "short_term":[
    "现状评估（点名关键风险点：如跳槽频繁/断层/学历一般/单一文化等）",
    "技能与经验补齐（点名技能/证书/项目/作品集）",
    "风险规避（若跳槽频繁→建议在现公司稳定 2-3 年并拿到可验证成果；若断层→解释模板与复位计划）",
    "小目标设计（3-6 个月，含定量指标）"
  ],
  "mid_term":[
    "目标角色与行业匹配（结合 level 与趋势）",
    "路径拆解（职责→交付物→量化目标）",
    "关键拐点（带团队/跨部门/对外成果/跨文化协作）"
  ],
  "long_term":[
    "角色演进的里程碑与衡量标准",
    "平台选择（大厂/独角兽/MNC/本地龙头/创业公司）取舍逻辑",
    "风险与弹性（行业周期、转型预案）"
  ],
  "if_job_hopping":[
    "若“跳槽频繁”成立：给出统一叙事模板（承认→内/外因→复盘→稳定计划）与下一份工作沉淀清单"
  ]
}}
- 所有条目必须具体可执行；短期策略必须服务于中长期目标；严禁空话。""",

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
    "教育一般：用作品集/成果弥补……"
  ]
}}
- 仅生成候选人对应 level 的手册；各列表至少 5/5/3（answer_logic/关注点/STAR 套数）。""",

        "ats": """仅输出 JSON（中文）：
{"ats":{"enabled":true,"total_score":0,"sub_scores":{"skills":0,"experience":0,"education":0,"keywords":0},
 "reasons":{"skills":["…"],"experience":["…"],"education":["…"],"keywords":["…"]},
 "gap_keywords":["…"],
 "improvement_advice":["…"]}}
- reasons 各 3–5 条；gap_keywords ≥ 10；improvement_advice ≥ 6，尽量贴 JD 条款。""",

        "salary": """仅输出 JSON（中文）：
{"salary_insights":{"title":"","city":"","currency":"CNY","low":0,"mid":0,"high":0,"factors":["…"],"notes":"模型估算，供参考"}}
- low < mid < high；给 5 个影响因子（公司体量/区域/行业热度/作品集质量/是否管理岗等）。"""
    }

    # —— 两阶段任务
    phase1 = ["summary_highlights", "improvements", "career_diagnosis", "career_level"]
    phase2 = ["personalized_strategy", "interview", "salary"]
    if has_jd:
        phase2.append("ats")

    qout = queue.Queue()
    phase1_results = {}

    def run_section(section, extra_ctx=None):
        ck_raw = {"base": base_user, "sec": section, "model": model, "extra": extra_ctx}
        ck = hashlib.sha256(json.dumps(ck_raw, ensure_ascii=False).encode()).hexdigest()
        cached = cache.get(ck)
        if cached is not None:
            qout.put({"section": section, "data": cached})
            return

        user_payload = dict(base_user)
        if extra_ctx:
            user_payload["prior_findings"] = extra_ctx

        msgs = [
            {"role": "system", "content": prompts[section]},
            {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)}
        ]
        payload = make_payload(msgs, model=model, max_tokens=per_call_tokens)

        try:
            # 段内超时 + JSON 失败降级 + 最小补位
            with ThreadPoolExecutor(max_workers=1) as inner:
                fut = inner.submit(lambda: call_llm(payload, json_mode=True))
                raw = fut.result(timeout=SECTION_TIMEOUT)

            try:
                obj = _extract_json(raw)
            except Exception:
                obj = _fallback_from_raw(raw, section)

            obj = _ensure_nonempty(section, obj)
            cache.set(ck, obj)
            qout.put({"section": section, "data": obj})
        except Exception as e:
            qout.put({"section": section, "error": str(e)})

    def streamer():
        yield "retry: 1500\n"
        # 立刻发送启动提示，前端立即有反馈
        yield 'data: {"section":"boot","data":{"msg":"引擎已启动，正在读取与拆解你的简历…"}}\n\n'

        # —— Phase 1
        with ThreadPoolExecutor(max_workers=min(4, len(phase1))) as ex1:
            for sec in phase1:
                ex1.submit(run_section, sec)

        need1 = set(phase1)
        last_beat = time.time()
        while need1:
            if time.time() - last_beat > 10:
                yield ": keep-alive\n\n"; last_beat = time.time()
            item = qout.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            need1.discard(item["section"])
            if "error" not in item and item["section"] in ("career_diagnosis","career_level"):
                phase1_results[item["section"]] = item["data"]

        # —— Phase 2（用 Phase1 的结果做个性化）
        extra_ctx = {
            "diagnosis": phase1_results.get("career_diagnosis", {}),
            "level":     phase1_results.get("career_level", {})
        }
        with ThreadPoolExecutor(max_workers=min(4, len(phase2))) as ex2:
            for sec in phase2:
                ex2.submit(run_section, sec, extra_ctx)

        need2 = set(phase2)
        while need2:
            if time.time() - last_beat > 10:
                yield ": keep-alive\n\n"; last_beat = time.time()
            item = qout.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            need2.discard(item["section"])

        meta = {"elapsed_ms": int((time.time()-t0)*1000), "model": model, "has_jd": has_jd}
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

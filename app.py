import os, time, json, re
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import requests

try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

app = Flask(__name__)

# === 环境变量（兼容） ===
LLM_API_BASE = (
    os.getenv("LLM_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.deepseek.com"
)
LLM_API_KEY  = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("MODEL_NAME")
    or "deepseek-chat"
)
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "16000"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS   = int(os.getenv("MAX_JD_CHARS",   "10000"))
TIMEOUT_SEC    = int(os.getenv("REQUEST_TIMEOUT", "300"))
BRAND_NAME = "Alsos AI Resume"

# === 工具 ===
def clean_text(s): 
    if not s: return ""
    s = s.replace("\r","\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def truncate(s, limit): return (s or "")[:limit]

def is_text_too_short(s):
    if not s: return True
    en_words = len(re.findall(r"[A-Za-z]+", s))
    return not (len(s)>=500 or en_words>=300)

def _extract_json(text: str):
    if not text: raise ValueError("空响应")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m: raise ValueError("未找到有效 JSON")
    return json.loads(m.group(0))

def call_llm(messages, json_mode=True, temperature=0.3, model_override=None):
    model = (model_override or LLM_MODEL or "deepseek-chat").strip()
    url = LLM_API_BASE.rstrip("/") + "/v1/chat/completions"
    payload = {"model": model, "temperature": temperature, "messages": messages, "max_tokens": MAX_TOKENS}
    if json_mode: payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
    if r.status_code >= 400: raise RuntimeError(f"LLM API 错误：{r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]

# === 页面 ===
@app.route("/")
def index(): return render_template("index.html", brand=BRAND_NAME)

@app.route("/extract-text", methods=["POST"])
def extract_text():
    if "file" not in request.files: return jsonify({"ok":False,"error":"未收到文件"}),400
    f = request.files["file"]; name = (f.filename or "").lower()
    if name.endswith(".txt"): text = f.read().decode("utf-8", errors="ignore")
    elif name.endswith(".docx") and HAS_DOCX: text = "\n".join(p.text for p in Document(f).paragraphs)
    else: return jsonify({"ok":False,"error":"仅支持 .txt / .docx"}),400
    return jsonify({"ok":True,"text":clean_text(text)})

# === 旧的整包接口（保留兜底） ===
@app.route("/optimize", methods=["POST"])
def optimize_whole():
    return jsonify({"ok": False, "error": "建议使用 /optimize_stream（流式输出）"}), 410

# === 新：流式接口（分模块生成 → 边生成边下发） ===
@app.route("/optimize_stream", methods=["POST"])
def optimize_stream():
    t0 = time.time()
    data = request.get_json(force=True) or {}
    resume_text     = clean_text(truncate(data.get("resume_text",""), MAX_TEXT_CHARS))
    target_title    = clean_text(data.get("target_title",""))
    target_location = clean_text(data.get("target_location",""))
    target_industry = clean_text(data.get("target_industry",""))
    job_description = clean_text(truncate(data.get("job_description",""), MAX_JD_CHARS))
    model_override  = (data.get("model") or "").strip() or None

    if not resume_text: return jsonify({"ok":False,"error":"请粘贴简历文本或上传文件"}),400
    if is_text_too_short(resume_text): return jsonify({"ok":False,"error":"简历文本过短（≥500 中文字符或 ≥300 英文词）"}),400
    has_jd = bool(job_description)

    base_user = {
        "resume_text": resume_text,
        "job_description": job_description if has_jd else "",
        "target_title": target_title,
        "target_location": target_location,
        "target_industry": target_industry
    }

    def gen():
        # SSE 头：兼容反向代理
        yield "retry: 1500\n"
        # 1) Summary
        sys_summary = f"""你是"{BRAND_NAME}"。仅生成 JSON：{{"summary": "<160-260字职业概要>","highlights":[...]}}。
- highlights≥8，完整句子、含数字/规模/结果（20-60字/条）。"""
        out1 = _extract_json(call_llm(
            [{"role":"system","content":sys_summary},
             {"role":"user","content":json.dumps(base_user, ensure_ascii=False)}],
            json_mode=True, temperature=0.25, model_override=model_override))
        yield f"data: {json.dumps({'section':'summary_highlights','data':out1}, ensure_ascii=False)}\n\n"

        # 2) 简历优化建议
        sys_improve = f"""仅生成 JSON：{{"resume_improvements":[{{"issue":"","fix":"","why":""}}...]}}
- ≥10条；结构固定为“问题点→改进方案→原因解释”；避免空话，给可执行改法与招聘逻辑理由。"""
        out2 = _extract_json(call_llm(
            [{"role":"system","content":sys_improve},
             {"role":"user","content":json.dumps(base_user, ensure_ascii=False)}],
            json_mode=True, temperature=0.3, model_override=model_override))
        yield f"data: {json.dumps({'section':'improvements','data':out2}, ensure_ascii=False)}\n\n"

        # 3) 关键词 + 求职方向
        sys_ck = """仅生成 JSON：
{"keywords": ["…"], "career_suggestions":{"short_term":["…"],"mid_term":["…"],"long_term":["…"]}}
- keywords≥12（行业/技能/工具标准术语）
- 短/中/长期分别≥5/≥5/≥4条：包含平台类型（大厂/独角兽/外企/本地龙头/咨询）、行动步骤、衡量指标；给出在年龄/教育/履历不占优时的补强路径。"""
        out3 = _extract_json(call_llm(
            [{"role":"system","content":sys_ck},
             {"role":"user","content":json.dumps(base_user, ensure_ascii=False)}],
            json_mode=True, temperature=0.3, model_override=model_override))
        yield f"data: {json.dumps({'section':'keywords_career','data':out3}, ensure_ascii=False)}\n\n"

        # 4) 面试手册
        sys_interview = """仅生成 JSON：
{"interview_handbook":{
  "answer_logic":["…"], "level_differences":{"junior":["…"],"senior":["…"]},
  "interviewer_focus":{"HR":["…"],"hiring_manager":["…"],"executive":["…"]},
  "star_sets":[{"project_title":"","question":"","how_to_answer":["…"]}, {"project_title":"","question":"","how_to_answer":["…"]}, {"project_title":"","question":"","how_to_answer":["…"]}]
}}
- answer_logic≥6；junior/senior各≥5；HR/负责人/老板各≥5；每套STAR含3-5步。"""
        out4 = _extract_json(call_llm(
            [{"role":"system","content":sys_interview},
             {"role":"user","content":json.dumps(base_user, ensure_ascii=False)}],
            json_mode=True, temperature=0.3, model_override=model_override))
        yield f"data: {json.dumps({'section':'interview','data':out4}, ensure_ascii=False)}\n\n"

        # 5) ATS（可选）
        if has_jd:
            sys_ats = """仅生成 JSON：
{"ats":{"enabled":true,"total_score":0,"sub_scores":{"skills":0,"experience":0,"education":0,"keywords":0},
 "reasons":{"skills":["…"],"experience":["…"],"education":["…"],"keywords":["…"]},
 "gap_keywords":["…"], "improvement_advice":["…"]}}
- reasons各3-5条；gap_keywords≥10；improvement_advice≥6，与JD条款可映射。"""
            out5 = _extract_json(call_llm(
                [{"role":"system","content":sys_ats},
                 {"role":"user","content":json.dumps(base_user, ensure_ascii=False)}],
                json_mode=True, temperature=0.25, model_override=model_override))
            yield f"data: {json.dumps({'section':'ats','data':out5}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'section':'ats','data':{'ats':{'enabled':False}}}, ensure_ascii=False)}\n\n"

        # 6) 薪酬
        sys_salary = """仅生成 JSON：
{"salary_insights":{"title":"","city":"","currency":"CNY","low":0,"mid":0,"high":0,"factors":["…"],"notes":"模型估算，供参考"}}
- low<mid<high；给5个影响因子（公司体量/区域/行业热度/作品集质量/是否管理岗等）。"""
        out6 = _extract_json(call_llm(
            [{"role":"system","content":sys_salary},
             {"role":"user","content":json.dumps(base_user, ensure_ascii=False)}],
            json_mode=True, temperature=0.25, model_override=model_override))
        yield f"data: {json.dumps({'section':'salary','data':out6}, ensure_ascii=False)}\n\n"

        # 完成
        meta = {"elapsed_ms": int((time.time()-t0)*1000), "model_alias": BRAND_NAME, "has_jd": has_jd}
        yield f"data: {json.dumps({'section':'done','data':{'meta':meta}}, ensure_ascii=False)}\n\n"

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"  # 禁止某些代理缓冲
    }
    return Response(stream_with_context(gen()), headers=headers)

@app.route("/healthz")
def healthz(): return "ok", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

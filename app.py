import os, json, hashlib, time
from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
CACHE = {}

def _k(payload: dict):
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def ds_chat(messages, max_tokens=700, response_json=True, temperature=0.2, timeout=60):
    body = {"model": MODEL_NAME, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_json:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{OPENAI_BASE_URL}/v1/chat/completions",
                      headers=HEADERS, data=json.dumps(body), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

@app.route("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "Alsos NeuroMatch"})

# —— 子能力：解析 / 优化 / ATS / 面试 / 职业建议（保留） ——
@app.post("/parse_resume")
def parse_resume():
    js = request.get_json(force=True); text = (js.get("resume_text") or "").strip()
    if len(text) > 25000: text = text[:25000]
    payload = {"resume_text": text}
    key = _k({"route": "parse_resume", **payload})
    if key in CACHE: return jsonify(CACHE[key])
    messages = [
        {"role":"system","content":(
            "你是ATS解析器。将输入的中文/英文简历文本解析为结构化JSON："
            "{basics, summary, education[], experiences[], skills_core[], skills_optional[], keywords[]}。"
            "时间用YYYY-MM；experiences[].bullets 每条≤30字并量化；只输出JSON。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    data = json.loads(ds_chat(messages, max_tokens=600, response_json=True))
    CACHE[key] = data; return jsonify(data)

@app.post("/resume_optimize")
def resume_optimize():
    js = request.get_json(force=True)
    payload = {
        "resume_struct": js.get("resume_struct", {}),
        "target_role": js.get("target_role", ""),
        "target_industry": js.get("target_industry", ""),
        "language": js.get("language", "zh")
    }
    key = _k({"route": "resume_optimize", **payload})
    if key in CACHE: return jsonify(CACHE[key])
    messages = [
        {"role":"system","content":(
            "你是顶级简历教练。基于 resume_struct，输出可直接用于替换的内容，仅输出JSON："
            "{section_order[], summary_cn, summary_en, bullets_to_add[], bullets_to_tighten[], "
            "skills_keywords_core[], skills_keywords_optional[], title_suggestions[]}。"
            "策略：动词+量化；行业中性；避免夸张。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    data = json.loads(ds_chat(messages, max_tokens=800, response_json=True))
    CACHE[key] = data; return jsonify(data)

@app.post("/ats_score")
def ats_score():
    js = request.get_json(force=True)
    payload = {
        "resume_struct": js.get("resume_struct", {}),
        "jd_text": js.get("jd_text", ""),
        "scoring_weights": js.get("scoring_weights", {"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15})
    }
    key = _k({"route": "ats_score", **payload})
    if key in CACHE: return jsonify(CACHE[key])
    messages = [
        {"role":"system","content":(
            "你是ATS评分器。比较 resume_struct 与 jd_text，输出JSON："
            "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
            "rewrite_bullets[], priority_actions[]}。"
            "评分=0.6*关键词重合+0.25*职责覆盖+0.15*加分项。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    data = json.loads(ds_chat(messages, max_tokens=700, response_json=True))
    CACHE[key] = data; return jsonify(data)

@app.post("/interview_qa")
def interview_qa():
    js = request.get_json(force=True)
    payload = {
        "resume_struct": js.get("resume_struct", {}),
        "target_role": js.get("target_role", ""),
        "target_industry": js.get("target_industry", ""),
        "level": js.get("level", "Senior"),
        "num": int(js.get("num", 12))
    }
    key = _k({"route": "interview_qa", **payload})
    if key in CACHE: return jsonify(CACHE[key])
    messages = [
        {"role":"system","content":(
            "你是面试官与教练。结合履历与目标岗位/行业，输出JSON："
            "{questions:[{category, question, how_to_answer, sample_answer, pitfalls[]}]}。"
            "要求：STAR结构；样例答案≤120字；避免套话。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    data = json.loads(ds_chat(messages, max_tokens=800, response_json=True))
    CACHE[key] = data; return jsonify(data)

@app.post("/career_advice")
def career_advice():
    js = request.get_json(force=True)
    payload = {
        "resume_struct": js.get("resume_struct", {}),
        "time_horizon": js.get("time_horizon", "3-5y"),
        "constraints": js.get("constraints", {})
    }
    key = _k({"route": "career_advice", **payload})
    if key in CACHE: return jsonify(CACHE[key])
    messages = [
        {"role":"system","content":(
            "你是职业规划顾问。输出3条可执行路径，每条包含："
            "{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}。"
            "仅输出JSON。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    data = json.loads(ds_chat(messages, max_tokens=700, response_json=True))
    CACHE[key] = data; return jsonify(data)

# —— 一键生成：整合为一份报告 ——
@app.post("/full_report")
def full_report():
    js = request.get_json(force=True)

    resume_text = (js.get("resume_text") or "").strip()
    target_role  = (js.get("target_role") or "").strip()
    location     = (js.get("location") or "").strip()
    jd_text      = (js.get("jd_text") or "").strip()
    industry     = (js.get("industry") or "").strip()

    if len(resume_text) > 25000: resume_text = resume_text[:25000]

    payload_sig = _k({
        "route":"full_report", "resume_text":resume_text,
        "target_role":target_role, "location":location,
        "jd_text":jd_text, "industry":industry
    })
    if payload_sig in CACHE:
        return jsonify(CACHE[payload_sig])

    # 1. 解析
    parsed = json.loads(ds_chat([
        {"role":"system","content":(
            "将简历文本解析为结构化JSON：{basics, summary, education[], experiences[], "
            "skills_core[], skills_optional[], keywords[]}；时间YYYY-MM；bullets≤30字并量化；仅输出JSON。")},
        {"role":"user","content": json.dumps({"resume_text": resume_text}, ensure_ascii=False)}
    ], max_tokens=600, response_json=True))

    # 2. 无JD优化
    optimized = json.loads(ds_chat([
        {"role":"system","content":(
            "基于 resume_struct 输出可直接替换的内容，仅输出JSON："
            "{section_order[], summary_cn, summary_en, bullets_to_add[], bullets_to_tighten[], "
            "skills_keywords_core[], skills_keywords_optional[], title_suggestions[]}。"
            "策略：动词+量化；行业中性；避免夸张。")},
        {"role":"user","content": json.dumps({
            "resume_struct": parsed, "target_role": target_role,
            "target_industry": industry, "language": "bilingual"
        }, ensure_ascii=False)}
    ], max_tokens=850, response_json=True))

    # 3. ATS（可选）
    ats = None
    if jd_text:
        ats = json.loads(ds_chat([
            {"role":"system","content":(
                "比较 resume_struct 与 jd_text，输出JSON："
                "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
                "rewrite_bullets[], priority_actions[]}。评分=0.6关键词+0.25职责+0.15加分项。")},
            {"role":"user","content": json.dumps({
                "resume_struct": parsed, "jd_text": jd_text,
                "scoring_weights":{"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15}
            }, ensure_ascii=False)}
        ], max_tokens=700, response_json=True))

    # 4. 面试问答
    qa = json.loads(ds_chat([
        {"role":"system","content":(
            "结合履历与岗位/行业，输出JSON：{questions:[{category, question, how_to_answer, "
            "sample_answer, pitfalls[]}]}；STAR结构；样例答案≤120字。")},
        {"role":"user","content": json.dumps({
            "resume_struct": parsed, "target_role": target_role or "General",
            "target_industry": industry or "General", "level":"Senior", "num":12
        }, ensure_ascii=False)}
    ], max_tokens=850, response_json=True))

    # 5. 职业建议（3–5年路径）
    advice = json.loads(ds_chat([
        {"role":"system","content":(
            "输出3条职业路径，每条含：{title, why_now, gap_to_fill[], "
            "skills_to_learn[], network_to_build[], 90_day_plan[]}；仅输出JSON。")},
        {"role":"user","content": json.dumps({
            "resume_struct": parsed, "time_horizon":"3-5y",
            "constraints":{"location": location}
        }, ensure_ascii=False)}
    ], max_tokens=800, response_json=True))

    result = {"parsed": parsed, "optimized": optimized, "ats": ats, "qa": qa, "advice": advice}
    CACHE[payload_sig] = result
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

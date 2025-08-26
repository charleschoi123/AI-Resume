import os, json, hashlib
from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

CACHE = {}

def cache_get(payload: dict):
    key = hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return key, CACHE.get(key)

def cache_set(key, value):
    CACHE[key] = value

def ds_chat(messages, max_tokens=700, response_json=True, temperature=0.2):
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if response_json:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{OPENAI_BASE_URL}/v1/chat/completions",
                      headers=HEADERS, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/parse_resume")
def parse_resume():
    js = request.get_json(force=True)
    text = (js.get("resume_text") or "").strip()
    if len(text) > 25000: text = text[:25000]
    payload = {"resume_text": text}
    key, cached = cache_get({"route": "parse_resume", **payload})
    if cached: return jsonify(cached)
    messages = [
        {"role":"system","content":(
            "你是ATS解析器。将输入的中文/英文简历文本解析为结构化JSON："
            "{basics, summary, education[], experiences[], skills_core[], skills_optional[], keywords[]}。"
            "时间用YYYY-MM；experiences[].bullets 每条≤30字并量化；只输出JSON。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    out = ds_chat(messages, max_tokens=600, response_json=True)
    data = json.loads(out); cache_set(key, data); return jsonify(data)

@app.post("/resume_optimize")
def resume_optimize():
    js = request.get_json(force=True)
    payload = {
        "resume_struct": js.get("resume_struct", {}),
        "target_role": js.get("target_role", ""),
        "target_industry": js.get("target_industry", ""),
        "language": js.get("language", "zh")
    }
    key, cached = cache_get({"route": "resume_optimize", **payload})
    if cached: return jsonify(cached)
    messages = [
        {"role":"system","content":(
            "你是Top简历教练。基于 resume_struct，输出可直接替换到简历中的内容，只输出JSON："
            "{section_order[], summary_cn, summary_en, bullets_to_add[], bullets_to_tighten[], "
            "skills_keywords_core[], skills_keywords_optional[], title_suggestions[]}。"
            "策略：动词+量化；行业中性；避免夸张。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    out = ds_chat(messages, max_tokens=800, response_json=True)
    data = json.loads(out); cache_set(key, data); return jsonify(data)

@app.post("/ats_score")
def ats_score():
    js = request.get_json(force=True)
    payload = {
        "resume_struct": js.get("resume_struct", {}),
        "jd_text": js.get("jd_text", ""),
        "scoring_weights": js.get("scoring_weights", {"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15})
    }
    key, cached = cache_get({"route": "ats_score", **payload})
    if cached: return jsonify(cached)
    messages = [
        {"role":"system","content":(
            "你是ATS评分器。比较 resume_struct 与 jd_text，输出JSON："
            "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
            "rewrite_bullets[], priority_actions[]}。"
            "评分=0.6*关键词重合+0.25*职责覆盖+0.15*加分项。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    out = ds_chat(messages, max_tokens=700, response_json=True)
    data = json.loads(out); cache_set(key, data); return jsonify(data)

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
    key, cached = cache_get({"route": "interview_qa", **payload})
    if cached: return jsonify(cached)
    messages = [
        {"role":"system","content":(
            "你是面试官与教练。结合履历与目标岗位/行业，输出JSON："
            "{questions:[{category, question, how_to_answer, sample_answer, pitfalls[]}]}。"
            "要求：STAR结构，样例答案≤120字，避免套话。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    out = ds_chat(messages, max_tokens=800, response_json=True)
    data = json.loads(out); cache_set(key, data); return jsonify(data)

@app.post("/career_advice")
def career_advice():
    js = request.get_json(force=True)
    payload = {
        "resume_struct": js.get("resume_struct", {}),
        "time_horizon": js.get("time_horizon", "6-12m"),
        "constraints": js.get("constraints", {})
    }
    key, cached = cache_get({"route": "career_advice", **payload})
    if cached: return jsonify(cached)
    messages = [
        {"role":"system","content":(
            "你是职业规划顾问。输出3条可执行路径，每条含："
            "{title, why_now, gap_to_fill[], 90_day_plan[]}。仅输出JSON。"
        )},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]
    out = ds_chat(messages, max_tokens=700, response_json=True)
    data = json.loads(out); cache_set(key, data); return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

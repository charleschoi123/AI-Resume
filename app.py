import os, json, hashlib, re
from flask import Flask, request, jsonify, render_template
import requests
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
CACHE = {}

def _sig(payload: dict) -> str:
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

# ---------- JSON 清洗 & 修复 ----------
def _strip_json_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M)
    return t

def load_json_strict(raw: str):
    if not isinstance(raw, str):
        raise RuntimeError("LLM返回的非字符串内容")
    t = _strip_json_fence(raw)
    # 抓取第一个 { ... } 块
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m: t = m.group(0)
    # 删除末尾多余逗号 ,} ,]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # 删除控制字符
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", t)
    return json.loads(t)

def repair_json_with_llm(bad_text: str, hint: str = ""):
    """调用模型把不合规文本修成严格 JSON。"""
    sys = (
        "你是JSON修复器。把用户提供的文本修正为**严格合法的JSON**："
        "1) 不要任何解释/注释/前后缀；2) 只能输出一个JSON对象；3) 不要省略号；4) 键用英文双引号；"
        "5) 不能有尾随逗号；6) 值为空用空数组/空字符串/0代替。"
    )
    if hint:
        sys += f" 结构提示：{hint}"
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"请修复为合法JSON：\n{bad_text[:12000]}"}
        ]
    }
    r = requests.post(f"{OPENAI_BASE_URL}/v1/chat/completions", headers=HEADERS, data=json.dumps(body), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"JSON修复失败: HTTP {r.status_code} {r.text[:200]}")
    data = r.json()
    fixed = data["choices"][0]["message"]["content"]
    return load_json_strict(fixed)

def call_json(messages, max_tokens=800, hint=""):
    """一次调用 → 解析；失败则自动用修复器重试一次。"""
    out = ds_chat(messages, max_tokens=max_tokens, response_json=True)
    try:
        return load_json_strict(out)
    except Exception as e:
        # 打印片段，便于日志定位
        print("JSON_PARSE_FAIL:", str(e), "| FRAGMENT:", out[:300])
        return repair_json_with_llm(out, hint)

# ---------- 调用大模型 ----------
def ds_chat(messages, max_tokens=700, response_json=True, temperature=0.2, timeout=60):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("缺少 API Key：请在 Render 环境变量中配置 DEEPSEEK_API_KEY")
    body = {"model": MODEL_NAME, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_json:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{OPENAI_BASE_URL}/v1/chat/completions", headers=HEADERS, data=json.dumps(body), timeout=timeout)
    if r.status_code != 200:
        print("LLM_ERROR_STATUS", r.status_code, r.text[:500])
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if "choices" not in data or not data["choices"]:
        print("LLM_BAD_PAYLOAD", data)
        raise RuntimeError("LLM empty choices or bad payload")
    return data["choices"][0]["message"]["content"]

# ---------- 基础路由 ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "Alsos NeuroMatch"})

# ---------- 子能力 ----------
@app.post("/parse_resume")
def parse_resume():
    try:
        js = request.get_json(force=True)
        text = (js.get("resume_text") or "").strip()
        if len(text) > 25000: text = text[:25000]
        payload = {"resume_text": text}
        key = _sig({"route":"parse_resume", **payload})
        if key in CACHE: return jsonify(CACHE[key])

        messages = [
            {"role":"system","content":(
                "你是ATS解析器。将输入的中文/英文简历文本解析为结构化JSON："
                "{basics, summary, education[], experiences[], skills_core[], skills_optional[], keywords[]}。"
                "时间用YYYY-MM；experiences[].bullets 每条≤30字并量化；只输出JSON。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=650, hint="{basics, summary, education[], experiences[], skills_*[], keywords[]}")
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/resume_optimize")
def resume_optimize():
    try:
        js = request.get_json(force=True)
        payload = {
            "resume_struct": js.get("resume_struct", {}),
            "target_role": js.get("target_role", ""),
            "target_industry": js.get("target_industry", ""),
            "language": js.get("language", "zh")
        }
        key = _sig({"route":"resume_optimize", **payload})
        if key in CACHE: return jsonify(CACHE[key])

        messages = [
            {"role":"system","content":(
                "你是顶级简历教练。基于 resume_struct，输出可直接用于替换的内容，仅输出JSON："
                "{section_order[], summary_cn, summary_en, bullets_to_add[], bullets_to_tighten[], "
                "skills_keywords_core[], skills_keywords_optional[], title_suggestions[]}。"
                "策略：动词+量化；行业中性；避免夸张。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=850, hint="{section_order[], summary_*, bullets_*[], skills_keywords_*[], title_suggestions[]}")
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/ats_score")
def ats_score():
    try:
        js = request.get_json(force=True)
        payload = {
            "resume_struct": js.get("resume_struct", {}),
            "jd_text": js.get("jd_text", ""),
            "scoring_weights": js.get("scoring_weights", {"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15})
        }
        key = _sig({"route":"ats_score", **payload})
        if key in CACHE: return jsonify(CACHE[key])

        messages = [
            {"role":"system","content":(
                "你是ATS评分器。比较 resume_struct 与 jd_text，输出JSON："
                "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
                "rewrite_bullets[], priority_actions[]}。评分=0.6*关键词重合+0.25*职责覆盖+0.15*加分项。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=750, hint="{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], rewrite_bullets[], priority_actions[]}")
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/interview_qa")
def interview_qa():
    try:
        js = request.get_json(force=True)
        payload = {
            "resume_struct": js.get("resume_struct", {}),
            "target_role": js.get("target_role", ""),
            "target_industry": js.get("target_industry", ""),
            "level": js.get("level", "Senior"),
            "num": int(js.get("num", 12))
        }
        key = _sig({"route":"interview_qa", **payload})
        if key in CACHE: return jsonify(CACHE[key])

        messages = [
            {"role":"system","content":(
                "你是面试官与教练。结合履历与目标岗位/行业，输出JSON："
                "{questions:[{category, question, how_to_answer, sample_answer, pitfalls[]}]}。"
                "要求：STAR结构；样例答案≤120字；避免套话。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=900, hint='{questions:[{category,question,how_to_answer,sample_answer,pitfalls[]}]}')
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/career_advice")
def career_advice():
    try:
        js = request.get_json(force=True)
        payload = {
            "resume_struct": js.get("resume_struct", {}),
            "time_horizon": js.get("time_horizon", "3-5y"),
            "constraints": js.get("constraints", {})
        }
        key = _sig({"route":"career_advice", **payload})
        if key in CACHE: return jsonify(CACHE[key])

        messages = [
            {"role":"system","content":(
                "你是职业规划顾问。输出3条可执行路径，每条包含："
                "{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}。"
                "仅输出JSON。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=800, hint="{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}")
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- 一键整合 ----------
@app.post("/full_report")
def full_report():
    try:
        js = request.get_json(force=True)
        resume_text = (js.get("resume_text") or "").strip()
        target_role  = (js.get("target_role") or "").strip()
        location     = (js.get("location") or "").strip()
        jd_text      = (js.get("jd_text") or "").strip()
        industry     = (js.get("industry") or "").strip()
        if len(resume_text) > 25000: resume_text = resume_text[:25000]

        cache_key = _sig({
            "route":"full_report", "resume_text":resume_text, "target_role":target_role,
            "location":location, "jd_text":jd_text, "industry":industry
        })
        if cache_key in CACHE: return jsonify(CACHE[cache_key])

        # 1) 解析
        parsed = call_json([
            {"role":"system","content":(
                "将简历文本解析为结构化JSON：{basics, summary, education[], experiences[], "
                "skills_core[], skills_optional[], keywords[]}；时间YYYY-MM；bullets≤30字并量化；仅输出JSON。")},
            {"role":"user","content": json.dumps({"resume_text": resume_text}, ensure_ascii=False)}
        ], max_tokens=650, hint="{basics, summary, education[], experiences[], skills_*[], keywords[]}")

        # 2) 无JD优化
        optimized = call_json([
            {"role":"system","content":(
                "基于 resume_struct 输出可直接替换的内容，仅输出JSON："
                "{section_order[], summary_cn, summary_en, bullets_to_add[], bullets_to_tighten[], "
                "skills_keywords_core[], skills_keywords_optional[], title_suggestions[]}。"
                "策略：动词+量化；行业中性；避免夸张。")},
            {"role":"user","content": json.dumps({
                "resume_struct": parsed, "target_role": target_role,
                "target_industry": industry, "language": "bilingual"
            }, ensure_ascii=False)}
        ], max_tokens=900, hint="{section_order[], summary_*, bullets_*[], skills_keywords_*[], title_suggestions[]}")

        # 3) ATS（可选）
        ats = None
        if jd_text:
            ats = call_json([
                {"role":"system","content":(
                    "比较 resume_struct 与 jd_text，输出JSON："
                    "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
                    "rewrite_bullets[], priority_actions[]}。评分=0.6关键词+0.25职责+0.15加分项。")},
                {"role":"user","content": json.dumps({
                    "resume_struct": parsed, "jd_text": jd_text,
                    "scoring_weights":{"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15}
                }, ensure_ascii=False)}
            ], max_tokens=800, hint="{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], rewrite_bullets[], priority_actions[]}")

        # 4) 面试问答
        qa = call_json([
            {"role":"system","content":(
                "结合履历与岗位/行业，输出JSON：{questions:[{category, question, how_to_answer, "
                "sample_answer, pitfalls[]}]}；STAR结构；样例答案≤120字。")},
            {"role":"user","content": json.dumps({
                "resume_struct": parsed, "target_role": target_role or "General",
                "target_industry": industry or "General", "level":"Senior", "num":12
            }, ensure_ascii=False)}
        ], max_tokens=900, hint='{questions:[{category,question,how_to_answer,sample_answer,pitfalls[]}]}')

        # 5) 职业建议
        advice = call_json([
            {"role":"system","content":(
                "输出3条职业路径，每条含：{title, why_now, gap_to_fill[], "
                "skills_to_learn[], network_to_build[], 90_day_plan[]}；仅输出JSON。")},
            {"role":"user","content": json.dumps({
                "resume_struct": parsed, "time_horizon":"3-5y",
                "constraints":{"location": location}
            }, ensure_ascii=False)}
        ], max_tokens=850, hint="{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}")

        result = {"parsed": parsed, "optimized": optimized, "ats": ats, "qa": qa, "advice": advice}
        CACHE[cache_key] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- 全局兜底 ----------
@app.errorhandler(Exception)
def handle_any_error(e):
    code = e.code if isinstance(e, HTTPException) else 500
    return jsonify({"error": str(e)}), code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

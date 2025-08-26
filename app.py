import os, json, hashlib, re
from flask import Flask, request, jsonify, render_template
import requests

# ========================
# 基础配置
# ========================
app = Flask(__name__)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
CACHE = {}

def _sig(payload: dict) -> str:
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

# ========================
# 安全 JSON 解析（自修复）
# ========================
def load_json_strict(s: str):
    """
    将模型文本尽力清洗为合法 JSON：
    - 去掉 ```/```json 代码围栏
    - 抓取首个 { ... } 块
    - 去掉末尾多余逗号
    - 报错时把片段返回，便于定位
    """
    if not isinstance(s, str):
        raise RuntimeError("LLM返回的非字符串内容")
    t = s.strip()
    # 去除markdown代码围栏
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    # 抓取第一个花括号JSON块
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        t = m.group(0)
    # 删除对象/数组前的多余逗号 ,}
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # 删除不可见控制字符
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", t)
    try:
        return json.loads(t)
    except Exception as e:
        raise RuntimeError(f"模型未返回有效JSON: {e} | 片段: {t[:200]}...")

# ========================
# 调用大模型（带稳健报错）
# ========================
def ds_chat(messages, max_tokens=700, response_json=True, temperature=0.2, timeout=60):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("缺少 API Key：请在 Render 环境变量中配置 DEEPSEEK_API_KEY")
    body = {"model": MODEL_NAME, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_json:
        body["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(f"{OPENAI_BASE_URL}/v1/chat/completions",
                          headers=HEADERS, data=json.dumps(body), timeout=timeout)
        if r.status_code != 200:
            print("LLM_ERROR_STATUS", r.status_code, r.text[:500])
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        if "choices" not in data or not data["choices"]:
            print("LLM_BAD_PAYLOAD", data)
            raise RuntimeError("LLM empty choices or bad payload")
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

# ========================
# 基础路由
# ========================
@app.route("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "Alsos NeuroMatch"})

# ========================
# 子能力：解析 / 优化 / ATS / 面试 / 职业建议
# ========================
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
        data = load_json_strict(ds_chat(messages, max_tokens=600, response_json=True))
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
        data = load_json_strict(ds_chat(messages, max_tokens=800, response_json=True))
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
        data = load_json_strict(ds_chat(messages, max_tokens=700, response_json=True))
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
        data = load_json_strict(ds_chat(messages, max_tokens=800, response_json=True))
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
        data = load_json_strict(ds_chat(messages, max_tokens=700, response_json=True))
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# 一键生成：整合报告
# ========================
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
        parsed = load_json_strict(ds_chat([
            {"role":"system","content":(
                "将简历文本解析为结构化JSON：{basics, summary, education[], experiences[], "
                "skills_core[], skills_optional[], keywords[]}；时间YYYY-MM；bullets≤30字并量化；仅输出JSON。")},
            {"role":"user","content": json.dumps({"resume_text": resume_text}, ensure_ascii=False)}
        ], max_tokens=600, response_json=True))

        # 2) 无JD优化
        optimized = load_json_strict(ds_chat([
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

        # 3) ATS（可选）
        ats = None
        if jd_text:
            ats = load_json_strict(ds_chat([
                {"role":"system","content":(
                    "比较 resume_struct 与 jd_text，输出JSON："
                    "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
                    "rewrite_bullets[], priority_actions[]}。评分=0.6关键词+0.25职责+0.15加分项。")},
                {"role":"user","content": json.dumps({
                    "resume_struct": parsed, "jd_text": jd_text,
                    "scoring_weights":{"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15}
                }, ensure_ascii=False)}
            ], max_tokens=700, response_json=True))

        # 4) 面试问答
        qa = load_json_strict(ds_chat([
            {"role":"system","content":(
                "结合履历与岗位/行业，输出JSON：{questions:[{category, question, how_to_answer, "
                "sample_answer, pitfalls[]}]}；STAR结构；样例答案≤120字。")},
            {"role":"user","content": json.dumps({
                "resume_struct": parsed, "target_role": target_role or "General",
                "target_industry": industry or "General", "level":"Senior", "num":12
            }, ensure_ascii=False)}
        ], max_tokens=850, response_json=True))

        # 5) 职业建议
        advice = load_json_strict(ds_chat([
            {"role":"system","content":(
                "输出3条职业路径，每条含：{title, why_now, gap_to_fill[], "
                "skills_to_learn[], network_to_build[], 90_day_plan[]}；仅输出JSON。")},
            {"role":"user","content": json.dumps({
                "resume_struct": parsed, "time_horizon":"3-5y",
                "constraints":{"location": location}
            }, ensure_ascii=False)}
        ], max_tokens=800, response_json=True))

        result = {"parsed": parsed, "optimized": optimized, "ats": ats, "qa": qa, "advice": advice}
        CACHE[cache_key] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# 本地启动
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

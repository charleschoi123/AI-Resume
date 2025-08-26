import os, time, json, re
from flask import Flask, request, jsonify, render_template
import requests

# 可选：支持 .docx 文本提取
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

app = Flask(__name__)

# ----------- 环境变量 -----------
LLM_API_BASE   = os.getenv("LLM_API_BASE", os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.deepseek.com")
LLM_API_KEY    = os.getenv("LLM_API_KEY",  os.getenv("DEEPSEEK_API_KEY", ""))
LLM_MODEL      = os.getenv("LLM_MODEL",    "deepseek-chat")

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "18000"))  # 简历最大长度
MAX_JD_CHARS   = int(os.getenv("MAX_JD_CHARS",   "10000"))  # JD 最大长度
TIMEOUT_SEC    = int(os.getenv("REQUEST_TIMEOUT", "120"))

BRAND_NAME = "Alsos AI Resume"

# ----------- 工具函数 -----------
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def truncate(s: str, limit: int) -> str:
    if s and len(s) > limit:
        return s[:limit]
    return s or ""

def is_text_too_short(s: str) -> bool:
    if not s:
        return True
    # 粗略规则：>=500中文字符 或 >=300英文词
    en_words = len(re.findall(r"[A-Za-z]+", s))
    return not (len(s) >= 500 or en_words >= 300)

def docx_to_text(file_storage) -> str:
    if not HAS_DOCX:
        return ""
    try:
        doc = Document(file_storage)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

# ----------- 路由 -----------
@app.route("/")
def index():
    return render_template("index.html", brand=BRAND_NAME)

@app.route("/extract-text", methods=["POST"])
def extract_text():
    """接收 .txt / .docx 文件，返回纯文本"""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "未收到文件"}), 400
    f = request.files["file"]
    name = (f.filename or "").lower()

    text = ""
    if name.endswith(".txt"):
        text = f.read().decode("utf-8", errors="ignore")
    elif name.endswith(".docx") and HAS_DOCX:
        text = docx_to_text(f)
    else:
        return jsonify({"ok": False, "error": "仅支持 .txt / .docx"}), 400

    text = clean_text(text)
    return jsonify({"ok": True, "text": text})

@app.route("/optimize", methods=["POST"])
def optimize():
    t0 = time.time()
    try:
        data = request.get_json(force=True, silent=False) or {}
        resume_text     = clean_text(truncate(data.get("resume_text", ""), MAX_TEXT_CHARS))
        target_title    = clean_text(data.get("target_title", ""))
        target_location = clean_text(data.get("target_location", ""))
        target_industry = clean_text(data.get("target_industry", ""))
        job_description = clean_text(truncate(data.get("job_description", ""), MAX_JD_CHARS))

        if not resume_text:
            return jsonify({"ok": False, "error": "请粘贴简历文本或上传文件"}), 400

        if is_text_too_short(resume_text):
            return jsonify({"ok": False, "error": "简历文本过短（≥500 中文字符或 ≥300 英文词）"}), 400

        has_jd = bool(job_description)
        # ---- 构造提示词（严格 JSON 输出）----
        system_prompt = f"""
You are the engine behind a job-seeking assistant called "{BRAND_NAME}".
Return a SINGLE compact JSON object that strictly follows the schema below.
Do NOT add commentary or markdown. Output must be valid JSON.

SCHEMA:
{{
  "meta": {{
    "has_jd": <bool>,
    "model_alias": "{BRAND_NAME}",
    "elapsed_ms": <number>
  }},
  "summary": <string>,
  "highlights": [<string>],
  "keywords": [<string>],
  "career_suggestions": {{
    "short_term": [<string>],
    "mid_term":   [<string>],
    "long_term":  [<string>]
  }},
  "interview_prep": {{
    "general": [<string>],
    "role_specific": [<string>],
    "star_tips": <string>
  }},
  "ats": {{
    "enabled": <bool>,
    "total_score": <number>, 
    "sub_scores": {{
      "skills": <number>,
      "experience": <number>,
      "education": <number>,
      "keywords": <number>
    }},
    "gap_keywords": [<string>],
    "improvement_advice": [<string>]
  }},
  "salary_insights": {{
    "title": <string>,
    "city": <string>,
    "low": <number>,
    "mid": <number>,
    "high": <number>,
    "notes": "模型估算，供参考"
  }}
}}

INSTRUCTIONS:
1) If JD is NOT provided, set "ats.enabled" = false and omit ATS scoring details (use zeros/empty-arrays).
2) Keep bullets concise and concrete (≤18 words each). No duplicated ideas.
3) Use Chinese output overall. Keep key terms in English when industry-standard.
4) Salary ranges must be reasonable relative to the title & city; include notes exactly "模型估算，供参考".
5) Ensure valid JSON. No trailing commas. No backticks. No extra keys.
"""

        user_prompt = {
            "resume_text": resume_text,
            "job_description": job_description if has_jd else "",
            "target_title": target_title,
            "target_location": target_location,
            "target_industry": target_industry
        }

        payload = {
            "model": LLM_MODEL,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
            ]
        }

        # 优先尝试 OpenAI 格式的 json_object（若不支持也能回退）
        payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        url = LLM_API_BASE.rstrip("/") + "/v1/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
        if resp.status_code >= 400:
            return jsonify({"ok": False, "error": f"LLM API 错误：{resp.status_code} {resp.text[:200]}"}), 502

        data = resp.json()
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return jsonify({"ok": False, "error": "LLM 返回格式异常"}), 502

        # 解析 JSON
        try:
            result = json.loads(content)
        except Exception:
            # 有些模型可能忽略 response_format，尝试用正则剥离 JSON
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                return jsonify({"ok": False, "error": "解析失败：未找到有效 JSON"}), 502
            result = json.loads(m.group(0))

        # 保底修正 meta.elapsed_ms
        result.setdefault("meta", {})
        result["meta"]["has_jd"] = has_jd
        result["meta"]["model_alias"] = BRAND_NAME
        result["meta"]["elapsed_ms"] = int((time.time() - t0) * 1000)

        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": f"服务器异常：{str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

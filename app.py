import os, time, json, re
from flask import Flask, request, jsonify, render_template
import requests

try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

app = Flask(__name__)

# ===== 环境变量 =====
LLM_API_BASE   = os.getenv("LLM_API_BASE", os.getenv("OPENAI_BASE_URL", "") or "https://api.deepseek.com")
LLM_API_KEY    = os.getenv("LLM_API_KEY",  os.getenv("DEEPSEEK_API_KEY", ""))
LLM_MODEL      = os.getenv("LLM_MODEL",    "deepseek-chat")
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "4096"))       # 提高可输出长度
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS   = int(os.getenv("MAX_JD_CHARS",   "10000"))
TIMEOUT_SEC    = int(os.getenv("REQUEST_TIMEOUT", "120"))

BRAND_NAME = "Alsos AI Resume"

# ===== 基础工具 =====
def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def truncate(s: str, limit: int) -> str:
    return (s or "")[:limit]

def is_text_too_short(s: str) -> bool:
    if not s: return True
    en_words = len(re.findall(r"[A-Za-z]+", s))
    return not (len(s) >= 500 or en_words >= 300)

def docx_to_text(file_storage) -> str:
    if not HAS_DOCX: return ""
    try:
        doc = Document(file_storage)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def call_llm(messages, json_mode=True, temperature=0.3):
    url = LLM_API_BASE.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "messages": messages,
        "max_tokens": MAX_TOKENS
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM API 错误：{resp.status_code} {resp.text[:200]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content

# ===== Web 路由 =====
@app.route("/")
def index():
    return render_template("index.html", brand=BRAND_NAME)

@app.route("/extract-text", methods=["POST"])
def extract_text():
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
    return jsonify({"ok": True, "text": clean_text(text)})

@app.route("/optimize", methods=["POST"])
def optimize():
    t0 = time.time()
    try:
        data = request.get_json(force=True) or {}
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

        # ===== 第 1 阶段：生成草稿 =====
        system_draft = f"""
You are the engine behind "{BRAND_NAME}". Generate a DRAFT JSON following SCHEMA exactly.
Keep language in Chinese except necessary technical terms.

SCHEMA:
{{
  "meta": {{"has_jd": <bool>, "model_alias": "{BRAND_NAME}", "elapsed_ms": 0}},
  "summary": <string 120-220 chars>,
  "highlights": [<string>],         # >=8 bullets, each <=18 chars, quantifiable where possible
  "keywords": [<string>],           # >=12, industry-standard tokens
  "career_suggestions": {{
    "short_term": [<string>],       # >=5, each with concrete action+tool+metric
    "mid_term":   [<string>],       # >=5
    "long_term":  [<string>]        # >=4
  }},
  "interview_prep": {{
    "general": [<string>],          # exactly 10 questions, concise
    "role_specific": [<string>],    # >=6 based on resume/JD
    "star_tips": <string 80-140 chars>
  }},
  "ats": {{
    "enabled": <bool>,
    "total_score": <number 0-100>,
    "sub_scores": {{
      "skills": <number>, "experience": <number>, "education": <number>, "keywords": <number>
    }},
    "reasons": {{
      "skills": [<string>],         # 3-5 items, <=18 chars each
      "experience": [<string>],
      "education": [<string>],
      "keywords": [<string>]
    }},
    "gap_keywords": [<string>],     # >=10
    "improvement_advice": [<string>]# >=6 specific edits aligned to JD
  }},
  "salary_insights": {{
    "title": <string>, "city": <string>, "currency": "CNY",
    "low": <number>, "mid": <number>, "high": <number>,
    "factors": [<string>],          # 5关键影响因子
    "notes": "模型估算，供参考"
  }}
}}

RULES:
- If no JD, set ats.enabled=false, scores=0, arrays=[], but keep the field.
- Avoid generic fluff; prefer numbers, tools, brands, datasets, methods from resume/JD.
- No duplication across bullets. No placeholders like "待完善".
- Valid JSON only.
"""
        user_payload = {
            "resume_text": resume_text,
            "job_description": job_description if has_jd else "",
            "target_title": target_title,
            "target_location": target_location,
            "target_industry": target_industry
        }
        draft = call_llm(
            [
                {"role": "system", "content": system_draft},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            json_mode=True,
            temperature=0.2
        )

        try:
            draft_obj = json.loads(re.search(r"\{.*\}", draft, re.S).group(0))
        except Exception:
            return jsonify({"ok": False, "error": "模型返回异常：无法解析草稿 JSON"}), 502

        # ===== 第 2 阶段：审稿精修 =====
        system_refine = f"""
You are a senior hiring manager and resume coach. REFINE the input DRAFT JSON to be concrete and expert-level.
Tasks:
1) Enforce all minimum counts (bullets/keywords/etc.). If below threshold, add more using resume/JD context.
2) Make bullets actionable: add metrics (% / # / revenue / timeline), tools (CAD/PS/CRM…), and domain nouns.
3) For career_suggestions, ensure each item includes: action+frequency/tool+measure.
4) For interview_prep: keep 10 general Q; add 3 STAR sets: (Question) + (What to prepare) for 3 key projects.
5) For ATS (if enabled): ensure reasons arrays have 3-5 items each; gap_keywords >=10; improvement_advice >=6 with precise edits mapped to JD lines.
6) Salary: keep CNY; ensure low<mid<high; ranges realistic for title+city.
Return VALID JSON only; same schema; concise style.
"""
        refined = call_llm(
            [
                {"role": "system", "content": system_refine},
                {"role": "user", "content": json.dumps({
                    "draft": draft_obj,
                    "resume_text": resume_text,
                    "job_description": job_description if has_jd else "",
                    "target_title": target_title,
                    "target_location": target_location,
                    "target_industry": target_industry
                }, ensure_ascii=False)}
            ],
            json_mode=True,
            temperature=0.3
        )

        try:
            result = json.loads(re.search(r"\{.*\}", refined, re.S).group(0))
        except Exception:
            return jsonify({"ok": False, "error": "模型返回异常：无法解析最终 JSON"}), 502

        # 修正 meta
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

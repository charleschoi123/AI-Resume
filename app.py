import os, time, json, re
from flask import Flask, request, jsonify, render_template
import requests

# 可选：.docx 文本抽取
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
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "4096"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "18000"))
MAX_JD_CHARS   = int(os.getenv("MAX_JD_CHARS",   "10000"))
TIMEOUT_SEC    = int(os.getenv("REQUEST_TIMEOUT", "120"))

BRAND_NAME = "Alsos AI Resume"

# ===== 工具 =====
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
    return data["choices"][0]["message"]["content"]

# ===== 路由 =====
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

        # ===== 阶段1：生成草稿 =====
        system_draft = f"""
You power a resume assistant called "{BRAND_NAME}".
Produce a DRAFT JSON in Chinese (keep tech terms in English) that STRICTLY matches this SCHEMA.

SCHEMA {{
  "meta": {{"has_jd": <bool>, "model_alias": "{BRAND_NAME}", "elapsed_ms": 0}},
  "summary": <string 160-260 chars>,   // 职业总结/概要：教育、总年限、行业、关键技能、代表性成果
  "highlights": [<string>],            // ≥8 条，完整句子、含数字/规模/结果（20-60字）
  "resume_improvements": [             // ≥10 条：问题点→改进方案→原因解释
    {{"issue": <string>, "fix": <string>, "why": <string>}}
  ],
  "keywords": [<string>],              // ≥12 个行业/技能术语（标准表达）
  "career_suggestions": {{
    "short_term": [<string>],          // ≥5，给出平台类型/行动步骤/衡量指标
    "mid_term":   [<string>],          // ≥5
    "long_term":  [<string>]           // ≥4；考虑年龄/教育/履历不占优时的补强路径
  }},
  "interview_handbook": {{
    "answer_logic": [<string>],        // ≥6：如何组织回答（STAR、讲故事+业绩+逻辑等）
    "level_differences": {{
      "junior": [<string>],            // ≥5：Junior 应该突出什么，如何作答
      "senior": [<string>]             // ≥5：Senior 应该突出什么，如何作答
    }},
    "interviewer_focus": {{
      "HR": [<string>],                // ≥5：HR 关注点与应对策略
      "hiring_manager": [<string>],    // ≥5：部门负责人
      "executive": [<string>]          // ≥5：老板/高层
    }},
    "star_sets": [                     // 3 套项目 STAR 提示
      {{
        "project_title": <string>,
        "question": <string>,
        "how_to_answer": [<string>]    // 3-5 步骤提示
      }}
    ]
  }},
  "ats": {{
    "enabled": <bool>,
    "total_score": <number 0-100>,
    "sub_scores": {{
      "skills": <number>, "experience": <number>, "education": <number>, "keywords": <number>
    }},
    "reasons": {{                      // 各子分 3-5 条理由（≤18字/条）
      "skills": [<string>],
      "experience": [<string>],
      "education": [<string>],
      "keywords": [<string>]
    }},
    "gap_keywords": [<string>],        // ≥10
    "improvement_advice": [<string>]   // ≥6，且与 JD 条款对齐（具体编辑/补充点）
  }},
  "salary_insights": {{
    "title": <string>, "city": <string>, "currency": "CNY",
    "low": <number>, "mid": <number>, "high": <number>,
    "factors": [<string>],             // 5 个关键影响因子
    "notes": "模型估算，供参考"
  }}
}}

RULES
- If no JD, set ats.enabled=false and zero out scores/arrays (but keep the field).
- Prefer specific numbers, tools, brands, and scenes from resume/JD; avoid generic fluff.
- No duplication across bullets. No placeholders like "待完善".
- VALID JSON only. No extra text.
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
            temperature=0.25
        )
        try:
            draft_obj = json.loads(re.search(r"\{.*\}", draft, re.S).group(0))
        except Exception:
            return jsonify({"ok": False, "error": "模型返回异常：无法解析草稿 JSON"}), 502

        # ===== 阶段2：审稿精修 =====
        system_refine = f"""
You are a senior hiring manager and resume coach.
REFINE the DRAFT JSON to meet all minimum counts and make each item actionable.
- Highlights: keep full sentences with metrics.
- resume_improvements: each must be "问题点→改进方案→原因解释"，≥10条，避免宽泛词。
- career_suggestions: include platform types (大厂/独角兽/外企/本地龙头/咨询等)、行动步骤、衡量指标，考虑年龄/教育/履历劣势时的取长补短路径。
- interview_handbook: 保持结构，充实要点，给出可执行的表达模板和提醒。
- ATS: 如果 enabled，确保 reasons 每类 3–5 条、gap_keywords ≥10、improvement_advice ≥6，且与 JD 条款逐条可映射。
- Salary: low < mid < high; realistic for title+city; keep currency=CNY。
Return VALID JSON only; same schema; concise but concrete.
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

# app.py  —— Alsos NeuroMatch · 求职助手（完整可用版）
import os, json, re, time
import requests
from flask import Flask, request, jsonify, render_template, send_file, make_response
from docx import Document
from docx.shared import Pt

# -------------------- 环境变量 --------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL_NAME       = os.getenv("MODEL_NAME", "deepseek-chat").strip()

JOOBLE_API_KEY   = os.getenv("JOOBLE_API_KEY", "").strip()
JOOBLE_ENDPOINT  = "https://jooble.org/api/{}/"  # POST JSON

HTTP_TIMEOUT     = 60  # 单次 HTTP 超时

# -------------------- Flask -----------------------
app = Flask(__name__, template_folder="templates")

# -------------------- 工具函数 ---------------------
def _first_json_block(s: str):
    """
    尝试从模型返回里提取第一段合法 JSON（防止前后有多余文本/代码块）
    """
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    # 取第一个 { ... } 块
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            # 尝试去掉反引号、BOM、奇怪的换行
            chunk = chunk.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            chunk = re.sub(r"```json|```", "", chunk)
            try:
                return json.loads(chunk)
            except Exception:
                pass
    # 彻底失败
    return {}

def call_llm_json(prompt: str, temperature: float = 0.2):
    """
    调用 DeepSeek（OpenAI 兼容 Chat Completions），要求返回 JSON。
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY")

    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME or "deepseek-chat",
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "你是一名专业的简历优化与职业顾问，请仅输出JSON，不要输出多余文字。"},
            {"role": "user", "content": prompt}
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:180]}")
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return _first_json_block(content)

def build_prompt(resume_text, target_role, location, jd_text, industry):
    """
    组织提示词，让模型返回结构化JSON
    """
    return f"""
请基于以下信息，返回严格 JSON：
- 简历文本：{resume_text[:4000]}
- 目标职位：{target_role}
- 期望地点：{location}
- 目标行业：{industry}
- （可选）职位JD：{jd_text[:4000]}

JSON 顶层必须包含：
{{
  "resume_optimization": {{
    "section_order": ["basics","summary","education","experiences","skills_core"],
    "skills_keywords_core": [],
    "skills_keywords_optional": [],
    "summary_cn": "",
    "summary_en": "",
    "bullets_to_add": [],
    "bullets_to_tighten": [],
    "title_suggestions": []
  }},
  "ats": {{
    "score": 0,
    "highlights": [],
    "mismatch": [],
    "keywords": []
  }},
  "interview": {{
    "questions": [
      {{"q": "问题1", "a": "作答建议"}},
      {{"q": "问题2", "a": "作答建议"}}
    ],
    "tips": []
  }},
  "career": {{
    "paths": [
      {{
        "title": "方向A",
        "why_now": "",
        "90_day_plan": [],
        "gap_to_fill": [],
        "network_to_build": [],
        "skills_to_learn": []
      }}
    ]
  }}
}}
仅输出 JSON。
""".strip()

# -------------------- 路由：页面 -------------------
@app.get("/")
def home():
    return render_template("index.html")

# 单一健康检查（注意：只保留这一处）
@app.get("/health")
def service_health():
    return jsonify({"ok": True, "service": "Alsos NeuroMatch"})

# -------------------- 路由：生成完整报告 -------------------
@app.post("/full_report")
def full_report():
    """
    入参 JSON：
      {resume_text, target_role, location, jd_text, industry}
    返回 JSON：
      {resume_optimization, ats, interview, career}
    """
    try:
        js = request.get_json(force=True) or {}
        resume_text = (js.get("resume_text") or "").strip()
        target_role = (js.get("target_role") or "").strip()
        location    = (js.get("location") or "").strip()
        jd_text     = (js.get("jd_text") or "").strip()
        industry    = (js.get("industry") or "").strip()

        if not resume_text:
            return jsonify({"error": "缺少简历文本"}), 400

        prompt = build_prompt(resume_text, target_role, location, jd_text, industry)
        data = call_llm_json(prompt, temperature=0.2)

        # 兜底：确保关键字段存在
        data.setdefault("resume_optimization", {})
        data.setdefault("ats", {})
        data.setdefault("interview", {})
        data.setdefault("career", {})

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- 路由：导出 DOCX -------------------
@app.post("/export_docx")
def export_docx():
    """
    入参：{report: <full_report返回的JSON>}
    返回：docx 文件流
    """
    try:
        js = request.get_json(force=True) or {}
        report = js.get("report") or {}

        doc = Document()
        styles = doc.styles['Normal']
        styles.font.name = 'Arial'
        styles.font.size = Pt(10)

        def h1(t): p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(16)
        def h2(t): p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(13)
        def code(obj):
            doc.add_paragraph(json.dumps(obj, ensure_ascii=False, indent=2))

        h1("Alsos NeuroMatch · 求职分析报告")
        doc.add_paragraph(time.strftime("生成时间：%Y-%m-%d %H:%M"))

        h2("简历优化")
        code(report.get("resume_optimization", {}))

        h2("ATS 匹配")
        code(report.get("ats", {}))

        h2("面试问答")
        code(report.get("interview", {}))

        h2("职业建议（3-5年）")
        code(report.get("career", {}))

        path = "AI_Resume_Report.docx"
        doc.save(path)
        return send_file(path, as_attachment=True, download_name=path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- 路由：在线职位匹配 -------------------
@app.post("/job_search")
def job_search():
    """
    入参：{ resume_struct, target_role, location, industry, limit }
    返回：{jobs: [{title, company, location, link, source, desc, score}]}
    """
    try:
        js = request.get_json(force=True) or {}
        resume_struct = js.get("resume_struct", {})
        target_role   = (js.get("target_role") or "").strip()
        location      = (js.get("location") or "").strip()
        industry      = (js.get("industry") or "").strip()
        limit         = int(js.get("limit", 10))

        # 关键词集合
        keywords = set()
        for k in (resume_struct.get("skills_core") or []) + (resume_struct.get("keywords") or []):
            if isinstance(k, str) and 1 <= len(k) <= 30:
                keywords.add(k.lower())
        if target_role:
            for t in re.split(r"[ /,;、，]+", target_role):
                if t:
                    keywords.add(t.lower())
        kw_list = list(keywords)[:8]
        results = []

        # Jooble（有 KEY 优先）
        if JOOBLE_API_KEY:
            payload = {
                "keywords": target_role or " ".join(kw_list) or "engineer",
                "location": location or "",
                "page": 1,
                "searchMode": 1,
                "radius": 40
            }
            try:
                r = requests.post(JOOBLE_ENDPOINT.format(JOOBLE_API_KEY), json=payload, timeout=HTTP_TIMEOUT)
                if r.status_code == 200:
                    data = r.json().get("jobs", [])
                    for j in data[:40]:
                        results.append({
                            "title": j.get("title"),
                            "company": j.get("company"),
                            "location": j.get("location"),
                            "link": j.get("link"),
                            "source": "Jooble",
                            "desc": (j.get("snippet") or "")[:500]
                        })
                else:
                    print("JOOBLE_HTTP", r.status_code, r.text[:200])
            except Exception as e:
                print("JOOBLE_ERR", e)

        # 回退源：Arbeitnow（无 Key 时先保证功能）
        if not results:
            try:
                url = "https://www.arbeitnow.com/api/job-board-api"
                r = requests.get(url, timeout=HTTP_TIMEOUT)
                if r.status_code == 200:
                    items = r.json().get("data", [])[:50]
                    for it in items:
                        results.append({
                            "title": it.get("title"),
                            "company": it.get("company_name"),
                            "location": (it.get("location") or "") + (" | Remote" if it.get("remote") else ""),
                            "link": it.get("url"),
                            "source": "Arbeitnow",
                            "desc": (it.get("description") or "")[:500]
                        })
            except Exception as e:
                print("ARBEITNOW_ERR", e)

        # 简单匹配度
        def score(job):
            text = " ".join([job.get("title",""), job.get("desc","")]).lower()
            hit = sum(1 for k in keywords if k and k in text)
            base = 60 * (hit / max(len(keywords), 1))
            if target_role and target_role.lower() in text: base += 15
            if location and location.lower() in (job.get("location","").lower()): base += 10
            return round(min(100, base), 1)

        for j in results:
            j["score"] = score(j)

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
        if not results and not JOOBLE_API_KEY:
            return jsonify({"error": "未配置 JOOBLE_API_KEY；已尝试回退源。"}), 400

        return jsonify({"jobs": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- 入口 -------------------------
if __name__ == "__main__":
    # 本地调试用；线上由 gunicorn 启动
    app.run(host="0.0.0.0", port=10000, debug=False)

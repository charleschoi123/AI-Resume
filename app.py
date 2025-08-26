# app.py —— Alsos NeuroMatch · 求职助手（JD 有/无 分流 + 强化 Prompt）
import os, json, re, time
import requests
from flask import Flask, request, jsonify, render_template, send_file
from docx import Document
from docx.shared import Pt

# ============ 环境变量 ============
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL_NAME       = os.getenv("MODEL_NAME", "deepseek-chat").strip()

JOOBLE_API_KEY   = os.getenv("JOOBLE_API_KEY", "").strip()
JOOBLE_ENDPOINT  = "https://jooble.org/api/{}/"  # POST JSON

HTTP_TIMEOUT     = 120  # 单次 HTTP 超时（秒）

# ============ Flask ============
app = Flask(__name__, template_folder="templates")

# ============ 工具 ============
def _first_json_block(s: str):
    """尽力从模型输出中抓到第一段合法 JSON。"""
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            chunk = chunk.replace("\u200b", "")
            chunk = re.sub(r"```json|```", "", chunk)
            try:
                return json.loads(chunk)
            except Exception:
                pass
    return {}

def call_llm_json(messages, temperature: float = 0.25):
    """调用 DeepSeek（OpenAI 兼容 Chat Completions），仅返回 JSON。"""
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
        "messages": messages,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return _first_json_block(content)

def ensure_schema(rep: dict) -> dict:
    """兜底补全前端需要的字段结构，避免 KeyError。"""
    rep = rep or {}
    # 简历优化
    rep.setdefault("resume_optimization", {})
    ro = rep["resume_optimization"]
    ro.setdefault("bullets_to_add", [])
    ro.setdefault("bullets_to_tighten", [])
    ro.setdefault("section_order", ["basics","summary","education","experiences","skills_core"])
    ro.setdefault("skills_keywords_core", [])
    ro.setdefault("skills_keywords_optional", [])
    ro.setdefault("summary_cn", "")
    ro.setdefault("summary_en", "")
    ro.setdefault("title_suggestions", [])

    # ATS（仅在有 JD 时有意义）
    rep.setdefault("ats", {})
    ats = rep["ats"]
    ats.setdefault("score", 0)
    ats.setdefault("highlights", [])
    ats.setdefault("mismatch", [])
    ats.setdefault("keywords", [])

    # 面试
    rep.setdefault("interview", {})
    iv = rep["interview"]
    iv.setdefault("questions", [])
    iv.setdefault("tips", [])

    # 职业路径
    rep.setdefault("career", {})
    car = rep["career"]
    car.setdefault("paths", [{
        "title": "",
        "why_now": "",
        "90_day_plan": [],
        "gap_to_fill": [],
        "network_to_build": [],
        "skills_to_learn": []
    }])

    # 无 JD 时的“职业全景洞察”
    rep.setdefault("career_context", {})
    cx = rep["career_context"]
    cx.setdefault("career_summary", "")
    cx.setdefault("highlight_strengths", [])
    # 风险项可为字符串或对象，这里用对象 schema
    cx.setdefault("potential_risks", [{"risk":"", "mitigation":""}])
    cx.setdefault("career_expansion", [{"direction":"", "how_to_start":""}])

    return rep

def need_expand(rep: dict, with_jd: bool) -> bool:
    """判断输出是否“太水”，触发一次加长重试。"""
    rep = ensure_schema(rep)
    ro = rep["resume_optimization"]
    iv = rep["interview"]

    base_short = (
        len((ro.get("summary_cn") or "")) < 80 or
        len((ro.get("summary_en") or "")) < 100 or
        len(ro.get("bullets_to_add") or []) < 5 or
        len(iv.get("questions") or []) < 4
    )
    if with_jd:
        ats = rep["ats"]
        jd_short = (
            len(ats.get("highlights") or []) < 5 or
            len(ats.get("mismatch") or []) < 3 or
            len(ats.get("keywords") or []) < 8
        )
        return base_short or jd_short
    else:
        cx = rep["career_context"]
        nojd_short = (
            len(cx.get("career_summary") or "") < 60 or
            len(cx.get("highlight_strengths") or []) < 5 or
            len(cx.get("potential_risks") or []) < 3 or
            len(cx.get("career_expansion") or []) < 3
        )
        return base_short or nojd_short

def build_messages_with_jd(resume_text, target_role, location, jd_text, industry, reinforce=False):
    strength = "（请在上一版基础上**显著扩展与细化**：每条建议附“原因+示例句子/改写前后对比”，Q&A≥4题，ATS 各项更具体）" if reinforce else ""
    sys = {
        "role":"system",
        "content":(
            "你是资深职业顾问 + 简历优化专家 + 面试官。你的任务是给候选人提供**可落地、结构化**、专业、详尽的建议。\n"
            "所有输出严格 JSON；不要解释文字、不要反引号。拒绝空话与套话。"
        )
    }
    user = {
        "role":"user",
        "content":f"""
【候选人简历】
{resume_text}

【目标职位/行业/地点】
职位：{target_role}
行业：{industry}
地点：{location}

【职位 JD】
{jd_text}

请生成**详细**的分析报告，严格输出如下 JSON 结构：
{{
  "resume_optimization": {{
    "bullets_to_add": ["使用 STAR 原则，给出 5-7 条可直接放进简历的要点，尽量包含 数字/规模/影响"],
    "bullets_to_tighten": ["列出 4-6 条原文可优化或删除的句子，并写“优化理由”与“改写示例”"],
    "section_order": ["basics","summary","education","experiences","skills_core"],
    "skills_keywords_core": ["列至少 12 个 ATS 核心关键词（与目标岗位强相关）"],
    "skills_keywords_optional": ["列 6-10 个可选关键词"],
    "summary_cn": "输出 100-150 字中文职业摘要，包含关键技能+行业聚焦+可量化成果",
    "summary_en": "输出 120-180 字英文 Summary（简洁有力、包含关键成就与能力）",
    "title_suggestions": ["列 3-5 个更贴近目标岗位的头衔"]
  }},
  "ats": {{
    "score": 0,
    "highlights": ["列 5-7 条与 JD 高匹配的点，具体到动作与指标"],
    "mismatch": ["列 4-6 条缺口点，并给出补救思路（如课程/项目/证书）"],
    "keywords": ["列 10-12 个建议加入简历的关键词"]
  }},
  "interview": {{
    "questions": [
      {{"q": "给出至少 4 个**行业真实**深度问题", "a": "每题用 STAR 四段示范答案（S/T/A/R）；若缺经历，给“无此经验时的回答策略”"}}
    ],
    "tips": ["列 5 条可执行的面试技巧，如 Portfolio 组织、量化表达等"]
  }},
  "career": {{
    "paths": [
      {{
        "title": "明确短/中/长期路径（如：短期=资深岗位；中期=团队管理；长期=部门负责人）",
        "why_now": "说明“为何现在是合适时机”",
        "90_day_plan": ["列 6-8 条‘前 90 天’行动（学习任务、交付物、里程碑）"],
        "gap_to_fill": ["列 6-8 条需要补齐的能力/证书/经验（越具体越好）"],
        "network_to_build": ["列 6-8 个要建立的人脉/社群/组织（写到岗位/城市/平台层面）"],
        "skills_to_learn": ["列 8-10 个硬技能+软技能，并在括号里给 1 个资源/证书名"]
      }}
    ]
  }},
  "career_context": {{
    "career_summary": "",
    "highlight_strengths": [],
    "potential_risks": [{{"risk":"", "mitigation":""}}],
    "career_expansion": [{{"direction":"", "how_to_start":""}}]
  }}
}}
写作要求：所有条目避免空话；尽量提供**数字、规模、指标**、真实动作或**资源名**；面试答案必须是 STAR 四段；若信息不足，可基于候选人背景**合理补全**。{strength}
""".strip()
    }
    return [sys, user]

def build_messages_without_jd(resume_text, target_role, location, industry, reinforce=False):
    strength = "（在上一版基础上**显著扩展**：优势条目≥6；风险≥4并逐条附“应对策略”；副业/技能拓展≥4且写“如何切入”）" if reinforce else ""
    sys = {
        "role":"system",
        "content":(
            "你是资深职业顾问 + 简历优化专家 + 面试官。你的任务是给候选人提供**可落地、结构化**、专业、详尽的建议。\n"
            "所有输出严格 JSON；不要解释文字、不要反引号。拒绝空话与套话。"
        )
    }
    user = {
        "role":"user",
        "content":f"""
【候选人简历】
{resume_text}

【目标职位/行业/地点】
职位：{target_role}
行业：{industry}
地点：{location}

⚠️ 注意：**没有 JD**。因此不要输出 ATS score/keywords。转而输出“职业全景洞察”。

请严格输出如下 JSON 结构：
{{
  "resume_optimization": {{
    "bullets_to_add": ["使用 STAR 原则，给出 5-7 条可直接放进简历的要点，尽量包含 数字/规模/影响"],
    "bullets_to_tighten": ["列出 4-6 条原文可优化或删除的句子，并写“优化理由”与“改写示例”"],
    "section_order": ["basics","summary","education","experiences","skills_core"],
    "skills_keywords_core": ["列至少 12 个 ATS 通用核心关键词（与目标岗位强相关）"],
    "skills_keywords_optional": ["列 6-10 个可选关键词"],
    "summary_cn": "输出 100-150 字中文职业摘要",
    "summary_en": "输出 120-180 字英文 Summary",
    "title_suggestions": ["列 3-5 个更贴近目标岗位的头衔"]
  }},
  "ats": {{}},
  "interview": {{
    "questions": [
      {{"q": "给出至少 4 个**行业真实**深度问题", "a": "每题用 STAR 四段示范答案（S/T/A/R）；若缺经历，给“无此经验时的回答策略”"}}
    ],
    "tips": ["列 5 条可执行的面试技巧"]
  }},
  "career": {{
    "paths": [
      {{
        "title": "明确短/中/长期路径（如：短期=资深岗位；中期=团队管理；长期=部门负责人）",
        "why_now": "说明“为何现在是合适时机”",
        "90_day_plan": ["列 6-8 条‘前 90 天’行动"],
        "gap_to_fill": ["列 6-8 条需要补齐的能力/证书/经验"],
        "network_to_build": ["列 6-8 个要建立的人脉/社群/组织"],
        "skills_to_learn": ["列 8-10 个硬技能+软技能，并给 1 个资源/证书名"]
      }}
    ]
  }},
  "career_context": {{
    "career_summary": "职业概述（80-120 字），用中文，客观凝练",
    "highlight_strengths": ["列 5-7 条候选人履历/个人优势，最好可量化"],
    "potential_risks": [
      {{"risk":"结合年龄/婚姻/行业常规的挑战点（例如 36 岁未婚在某些行业转岗）","mitigation":"具体应对策略，给出动作或资源"}},
      {{"risk":"…","mitigation":"…"}}
    ],
    "career_expansion": [
      {{"direction":"可拓展的副业或未来技能方向（如行业自媒体/咨询/培训/数据分析/AI+本职）","how_to_start":"如何切入（平台、作品、课程、里程碑）"}},
      {{"direction":"…","how_to_start":"…"}}
    ]
  }}
}}
写作要求：不要输出 ATS 评分；“职业全景洞察”写得务实、可执行，尤其是“风险→应对策略”“副业/技能→如何切入”。{strength}
""".strip()
    }
    return [sys, user]

# ============ 页面 ============
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"ok": True})

# ============ 生成报告 ============
@app.post("/full_report")
def full_report():
    try:
        js = request.get_json(force=True) or {}
        resume_text = (js.get("resume_text") or "").trim() if hasattr(str,"trim") else (js.get("resume_text") or "").strip()
        target_role = (js.get("target_role") or "").strip()
        location    = (js.get("location") or "").strip()
        jd_text     = (js.get("jd_text") or "").strip()
        industry    = (js.get("industry") or "").strip()

        if not resume_text:
            return jsonify({"error": "缺少简历文本"}), 400

        with_jd = bool(jd_text)

        # 首次生成
        if with_jd:
            msgs = build_messages_with_jd(resume_text, target_role, location, jd_text, industry, reinforce=False)
        else:
            msgs = build_messages_without_jd(resume_text, target_role, location, industry, reinforce=False)
        rep = call_llm_json(msgs)
        rep = ensure_schema(rep)

        # 输出质量不足 → 二次加长
        if need_expand(rep, with_jd):
            if with_jd:
                msgs2 = build_messages_with_jd(resume_text, target_role, location, jd_text, industry, reinforce=True)
            else:
                msgs2 = build_messages_without_jd(resume_text, target_role, location, industry, reinforce=True)
            rep2 = call_llm_json(msgs2)
            rep = ensure_schema(rep2)

        return jsonify(rep)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ 导出 DOCX ============
@app.post("/export_docx")
def export_docx():
    try:
        js = request.get_json(force=True) or {}
        report = ensure_schema(js.get("report") or {})

        doc = Document()
        styles = doc.styles['Normal']
        styles.font.name = 'Arial'
        styles.font.size = Pt(10)

        def H1(t): p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(16)
        def H2(t): p=doc.add_paragraph(); r=p.add_run(t); r.bold=True; r.font.size=Pt(13)

        H1("Alsos NeuroMatch · 求职分析报告")
        doc.add_paragraph(time.strftime("生成时间：%Y-%m-%d %H:%M"))

        # 简历优化
        ro = report["resume_optimization"]
        H2("简历优化（可直接粘贴到简历）")
        doc.add_paragraph("职业摘要（中文）").add_run("\n"+ro["summary_cn"])
        doc.add_paragraph("Summary（English）").add_run("\n"+ro["summary_en"])
        doc.add_paragraph("职位标题建议： " + " / ".join(ro["title_suggestions"]))
        doc.add_paragraph("推荐版块顺序： " + " → ".join(ro["section_order"]))
        doc.add_paragraph("核心关键词： " + "、".join(ro["skills_keywords_core"]))
        doc.add_paragraph("可选关键词： " + "、".join(ro["skills_keywords_optional"]))
        doc.add_paragraph("\n【建议添加的要点】")
        for b in ro["bullets_to_add"]: doc.add_paragraph("• "+b)
        doc.add_paragraph("\n【建议精简/收紧】")
        for b in ro["bullets_to_tighten"]: doc.add_paragraph("• "+b)

        # ATS（可能空）
        ats = report.get("ats") or {}
        if ats.get("highlights") or ats.get("mismatch") or ats.get("keywords") or ats.get("score"):
            H2("ATS 匹配（如提供 JD）")
            if ats.get("score") is not None:
                doc.add_paragraph(f"ATS 评分：{ats.get('score',0)}")
            doc.add_paragraph("匹配亮点：");  [doc.add_paragraph("• "+x) for x in ats.get("highlights",[])]
            doc.add_paragraph("不匹配/缺口：");[doc.add_paragraph("• "+x) for x in ats.get("mismatch",[])]
            if ats.get("keywords"):
                doc.add_paragraph("建议加入的关键词： " + "、".join(ats.get("keywords",[])))

        # 职业全景洞察（无 JD）
        cx = report.get("career_context") or {}
        if cx.get("career_summary") or cx.get("highlight_strengths") or cx.get("career_expansion"):
            H2("职业全景洞察（无 JD）")
            doc.add_paragraph("职业概述：").add_run("\n"+(cx.get("career_summary") or ""))
            if cx.get("highlight_strengths"):
                doc.add_paragraph("履历/个人亮点：")
                for s in cx["highlight_strengths"]: doc.add_paragraph("• "+s)
            if cx.get("potential_risks"):
                doc.add_paragraph("潜在挑战与应对：")
                for item in cx["potential_risks"]:
                    if isinstance(item, dict):
                        doc.add_paragraph(f"• 风险：{item.get('risk','')}  |  应对：{item.get('mitigation','')}")
                    else:
                        doc.add_paragraph("• "+str(item))
            if cx.get("career_expansion"):
                doc.add_paragraph("副业/技能拓展：")
                for item in cx["career_expansion"]:
                    if isinstance(item, dict):
                        doc.add_paragraph(f"• 方向：{item.get('direction','')}  |  切入：{item.get('how_to_start','')}")
                    else:
                        doc.add_paragraph("• "+str(item))

        # 面试
        H2("面试问答")
        iv = report["interview"]
        for i,qa in enumerate(iv["questions"],1):
            doc.add_paragraph(f"Q{i}: {qa.get('q','')}")
            doc.add_paragraph(f"A: {qa.get('a','')}")
        doc.add_paragraph("面试提示："); [doc.add_paragraph("• "+x) for x in iv["tips"]]

        # 职业建议
        H2("职业建议（3-5年）")
        for p in report["career"]["paths"]:
            doc.add_paragraph("方向： " + p.get("title",""))
            doc.add_paragraph("Why now： " + p.get("why_now",""))
            doc.add_paragraph("90 天行动：");       [doc.add_paragraph("• "+x) for x in p.get("90_day_plan",[])]
            doc.add_paragraph("需补齐能力/经历："); [doc.add_paragraph("• "+x) for x in p.get("gap_to_fill",[])]
            doc.add_paragraph("要建立的人脉：");   [doc.add_paragraph("• "+x) for x in p.get("network_to_build",[])]
            doc.add_paragraph("要学习的技能：");   [doc.add_paragraph("• "+x) for x in p.get("skills_to_learn",[])]

        path = "AI_Resume_Report.docx"
        doc.save(path)
        return send_file(path, as_attachment=True, download_name=path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ 在线职位匹配 ============
@app.post("/job_search")
def job_search():
    try:
        js = request.get_json(force=True) or {}
        resume_struct = js.get("resume_struct", {})
        target_role   = (js.get("target_role") or "").strip()
        location      = (js.get("location") or "").strip()
        industry      = (js.get("industry") or "").strip()
        limit         = int(js.get("limit", 10))

        keywords = set()
        for k in (resume_struct.get("skills_keywords_core") or []) + (resume_struct.get("keywords") or []):
            if isinstance(k, str) and 1 <= len(k) <= 30:
                keywords.add(k.lower())
        if target_role:
            for t in re.split(r"[ /,;、，]+", target_role):
                if t:
                    keywords.add(t.lower())
        kw_list = list(keywords)[:8]
        results = []

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
                    for j in data[:50]:
                        results.append({
                            "title": j.get("title"),
                            "company": j.get("company"),
                            "location": j.get("location"),
                            "link": j.get("link"),
                            "source": "Jooble",
                            "desc": (j.get("snippet") or "")[:500]
                        })
            except Exception as e:
                print("JOOBLE_ERR", e)

        if not results:
            try:
                url = "https://www.arbeitnow.com/api/job-board-api"
                r = requests.get(url, timeout=HTTP_TIMEOUT)
                if r.status_code == 200:
                    items = r.json().get("data", [])[:60]
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

# ============ 入口 ============
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

import os, json, hashlib, re, io
from flask import Flask, request, jsonify, render_template, send_file
import requests
from werkzeug.exceptions import HTTPException
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== 可选：职位搜索 API（Jooble） ======
JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY", "")  # 去 https://jooble.org/api/about 申请，免费
JOOBLE_ENDPOINT = "https://jooble.org/api/{}"

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
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m: t = m.group(0)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", t)
    return json.loads(t)

def repair_json_with_llm(bad_text: str, hint: str = ""):
    sys = (
        "你是JSON修复器。把用户提供的文本修成**严格合法JSON**："
        "仅输出一个JSON对象；不能包含注释/说明/省略号；键用双引号；禁止尾随逗号；"
        "不确定的值用空数组或空字符串。"
    )
    if hint: sys += f" 结构提示：{hint}"
    body = {"model": MODEL_NAME, "messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"请修复为合法JSON：\n{bad_text[:12000]}"},
    ]}
    r = requests.post(f"{OPENAI_BASE_URL}/v1/chat/completions", headers=HEADERS, data=json.dumps(body), timeout=40)
    if r.status_code != 200:
        raise RuntimeError(f"JSON修复失败: HTTP {r.status_code} {r.text[:200]}")
    fixed = r.json()["choices"][0]["message"]["content"]
    return load_json_strict(fixed)

def call_json(messages, max_tokens=700, hint=""):
    out = ds_chat(messages, max_tokens=max_tokens, response_json=True, timeout=40)
    try:
        return load_json_strict(out)
    except Exception as e:
        print("JSON_PARSE_FAIL:", str(e), "| FRAGMENT:", out[:300])
        return repair_json_with_llm(out, hint)

# ---------- 调用大模型 ----------
def ds_chat(messages, max_tokens=600, response_json=True, temperature=0.2, timeout=40):
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

# ---------- 子能力（与之前一致） ----------
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
                "时间YYYY-MM；experiences[].bullets 每条≤30字并量化；仅输出JSON。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=600, hint="{basics, summary, education[], experiences[], skills_*[], keywords[]}")
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/resume_optimize")
def resume_optimize():
    try:
        js = request.get_json(force=True)
        payload = {"resume_struct": js.get("resume_struct", {}), "target_role": js.get("target_role", ""),
                   "target_industry": js.get("target_industry", ""), "language": js.get("language", "zh")}
        key = _sig({"route":"resume_optimize", **payload})
        if key in CACHE: return jsonify(CACHE[key])
        messages = [
            {"role":"system","content":(
                "你是顶级简历教练。基于 resume_struct，输出可直接用于替换的内容，仅输出JSON："
                "{section_order[], summary_cn, summary_en, bullets_to_add[], bullets_to_tighten[], "
                "skills_keywords_core[], skills_keywords_optional[], title_suggestions[]}。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=750,
                         hint="{section_order[], summary_*, bullets_*[], skills_keywords_*[], title_suggestions[]}")
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/ats_score")
def ats_score():
    try:
        js = request.get_json(force=True)
        payload = {"resume_struct": js.get("resume_struct", {}), "jd_text": js.get("jd_text", ""),
                   "scoring_weights": js.get("scoring_weights", {"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15})}
        key = _sig({"route":"ats_score", **payload})
        if key in CACHE: return jsonify(CACHE[key])
        messages = [
            {"role":"system","content":(
                "你是ATS评分器。比较 resume_struct 与 jd_text，输出JSON："
                "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
                "rewrite_bullets[], priority_actions[]}。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=700,
                         hint="{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], rewrite_bullets[], priority_actions[]}")
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/interview_qa")
def interview_qa():
    try:
        js = request.get_json(force=True)
        payload = {"resume_struct": js.get("resume_struct", {}), "target_role": js.get("target_role", ""),
                   "target_industry": js.get("target_industry", ""), "level": js.get("level", "Senior"),
                   "num": int(js.get("num", 12))}
        key = _sig({"route":"interview_qa", **payload})
        if key in CACHE: return jsonify(CACHE[key])
        messages = [
            {"role":"system","content":(
                "结合履历与目标岗位/行业，输出JSON：{questions:[{category, question, how_to_answer, sample_answer, pitfalls[]}]}；"
                "STAR结构；样例答案≤120字。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=800, hint='{questions:[{category,question,how_to_answer,sample_answer,pitfalls[]}]}')
        CACHE[key] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/career_advice")
def career_advice():
    try:
        js = request.get_json(force=True)
        payload = {"resume_struct": js.get("resume_struct", {}), "time_horizon": js.get("time_horizon", "3-5y"),
                   "constraints": js.get("constraints", {})}
        key = _sig({"route":"career_advice", **payload})
        if key in CACHE: return jsonify(CACHE[key])
        messages = [
            {"role":"system","content":(
                "输出3条职业路径：{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}。仅输出JSON。")},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]
        data = call_json(messages, max_tokens=750,
                         hint="{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}")
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

        cache_key = _sig({"route":"full_report","resume_text":resume_text,"target_role":target_role,
                          "location":location,"jd_text":jd_text,"industry":industry})
        if cache_key in CACHE: return jsonify(CACHE[cache_key])

        parsed = call_json([
            {"role":"system","content":(
                "将简历文本解析为结构化JSON：{basics, summary, education[], experiences[], "
                "skills_core[], skills_optional[], keywords[]}；时间YYYY-MM；bullets≤30字并量化；仅输出JSON。")},
            {"role":"user","content": json.dumps({"resume_text": resume_text}, ensure_ascii=False)}
        ], max_tokens=600, hint="{basics, summary, education[], experiences[], skills_*[], keywords[]}")

        def task_opt():
            return call_json([
                {"role":"system","content":(
                    "基于 resume_struct 输出可直接替换的内容，仅输出JSON："
                    "{section_order[], summary_cn, summary_en, bullets_to_add[], bullets_to_tighten[], "
                    "skills_keywords_core[], skills_keywords_optional[], title_suggestions[]}。")},
                {"role":"user","content": json.dumps({
                    "resume_struct": parsed, "target_role": target_role,
                    "target_industry": industry, "language": "bilingual"
                }, ensure_ascii=False)}
            ], max_tokens=850, hint="{section_order[], summary_*, bullets_*[], skills_keywords_*[], title_suggestions[]}")

        def task_ats():
            if not jd_text: return None
            return call_json([
                {"role":"system","content":(
                    "比较 resume_struct 与 jd_text，输出JSON："
                    "{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], "
                    "rewrite_bullets[], priority_actions[]}。")},
                {"role":"user","content": json.dumps({
                    "resume_struct": parsed, "jd_text": jd_text,
                    "scoring_weights":{"keywords":0.6,"responsibilities":0.25,"nice_to_have":0.15}
                }, ensure_ascii=False)}
            ], max_tokens=750, hint="{match_score, overlap_keywords[], gap_keywords[], responsibility_coverage[], rewrite_bullets[], priority_actions[]}")

        def task_qa():
            return call_json([
                {"role":"system","content":(
                    "结合履历与岗位/行业，输出JSON：{questions:[{category, question, how_to_answer, "
                    "sample_answer, pitfalls[]}]}；STAR结构；样例答案≤120字。")},
                {"role":"user","content": json.dumps({
                    "resume_struct": parsed, "target_role": target_role or "General",
                    "target_industry": industry or "General", "level":"Senior", "num":12
                }, ensure_ascii=False)}
            ], max_tokens=850, hint='{questions:[{category,question,how_to_answer,sample_answer,pitfalls[]}]}')

        def task_adv():
            return call_json([
                {"role":"system","content":(
                    "输出3条职业路径：{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}。仅输出JSON。")},
                {"role":"user","content": json.dumps({
                    "resume_struct": parsed, "time_horizon":"3-5y",
                    "constraints":{"location": location}
                }, ensure_ascii=False)}
            ], max_tokens=800, hint="{title, why_now, gap_to_fill[], skills_to_learn[], network_to_build[], 90_day_plan[]}")

        results = {"parsed": parsed, "optimized": None, "ats": None, "qa": None, "advice": None}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(task_opt): "optimized", ex.submit(task_qa): "qa", ex.submit(task_adv): "advice"}
            if jd_text: futs[ex.submit(task_ats)] = "ats"
            for f in as_completed(futs):
                key = futs[f]
                try: results[key] = f.result()
                except Exception as e: results[key] = {"error": str(e)}

        CACHE[cache_key] = results
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- 导出 DOCX ----------
@app.post("/export_docx")
def export_docx():
    """
    传入：{ report: <full_report_json>, filename: "报告.docx"(可选) }
    返回：DOCX 文件
    """
    try:
        js = request.get_json(force=True)
        report = js.get("report", {})
        filename = js.get("filename", "NeuroMatch-报告.docx")

        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()
        style = doc.styles['Normal']
        style.font.name = 'Microsoft YaHei'
        style.font.size = Pt(10.5)

        def h1(t):
            p = doc.add_paragraph()
            r = p.add_run(t)
            r.bold = True; r.font.size = Pt(16)
        def h2(t):
            p = doc.add_paragraph()
            r = p.add_run(t)
            r.bold = True; r.font.size = Pt(13)

        h1('Alsos NeuroMatch · 求职助手报告')
        doc.add_paragraph('')

        # 简历优化
        opt = report.get('optimized', {})
        h2('一、简历优化（可直接复制到简历）')
        if opt.get('summary_cn'): doc.add_paragraph('【中文总结】'+opt['summary_cn'])
        if opt.get('summary_en'): doc.add_paragraph('【English Summary】'+opt['summary_en'])
        if opt.get('section_order'):
            doc.add_paragraph('【推荐板块顺序】'+'、'.join(opt['section_order']))
        if opt.get('skills_keywords_core'):
            doc.add_paragraph('【核心关键词】'+'、'.join(opt['skills_keywords_core']))
        if opt.get('bullets_to_add'):
            doc.add_paragraph('【应补充要点】')
            for b in opt['bullets_to_add']: doc.add_paragraph('· '+b, style=None)
        if opt.get('bullets_to_tighten'):
            doc.add_paragraph('【可精简要点】')
            for b in opt['bullets_to_tighten']: doc.add_paragraph('· '+b, style=None)
        if opt.get('title_suggestions'):
            doc.add_paragraph('【推荐头衔】'+'、'.join(opt['title_suggestions']))

        # ATS
        ats = report.get('ats')
        h2('二、ATS 匹配')
        if ats:
            doc.add_paragraph(f'【匹配度】{int(round(ats.get("match_score",0)))}')
            if ats.get('overlap_keywords'):
                doc.add_paragraph('【关键词覆盖】'+'、'.join(ats['overlap_keywords']))
            if ats.get('gap_keywords'):
                doc.add_paragraph('【关键词缺口】'+'、'.join(ats['gap_keywords']))
            if ats.get('priority_actions'):
                doc.add_paragraph('【优先行动】')
                for a in ats['priority_actions']: doc.add_paragraph('· '+a)
            if ats.get('rewrite_bullets'):
                doc.add_paragraph('【推荐改写】')
                for a in ats['rewrite_bullets']: doc.add_paragraph('· '+a)
        else:
            doc.add_paragraph('未提供 JD，暂不评分。')

        # 面试问答
        qa = (report.get('qa') or {}).get('questions', [])
        h2('三、行业面试问答')
        for i,q in enumerate(qa[:12],1):
            doc.add_paragraph(f'Q{i}. {q.get("question","")}')
            if q.get('how_to_answer'): doc.add_paragraph('答题思路：'+q['how_to_answer'])
            if q.get('sample_answer'): doc.add_paragraph('示例答案：'+q['sample_answer'])
            if q.get('pitfalls'):
                doc.add_paragraph('易踩坑：')
                for p in q['pitfalls']: doc.add_paragraph('· '+p)

        # 职业建议
        paths = report.get('advice',{}).get('career_paths') or report.get('advice',[])
        h2('四、职业建议（3–5年）')
        if paths:
            for idx,p in enumerate(paths,1):
                doc.add_paragraph(f'路径{idx}：{p.get("title","")}')
                if p.get('why_now'): doc.add_paragraph('为什么现在：'+p['why_now'])
                if p.get('90_day_plan'): 
                    doc.add_paragraph('90天计划：'); 
                    for x in p['90_day_plan']: doc.add_paragraph('· '+x)
                if p.get('gap_to_fill'):
                    doc.add_paragraph('需要补齐：'); 
                    for x in p['gap_to_fill']: doc.add_paragraph('· '+x)
                if p.get('skills_to_learn'):
                    doc.add_paragraph('要学习的技能：'); 
                    for x in p['skills_to_learn']: doc.add_paragraph('· '+x)
                if p.get('network_to_build'):
                    doc.add_paragraph('要建立的人脉：'); 
                    for x in p['network_to_build']: doc.add_paragraph('· '+x)
        else:
            doc.add_paragraph('（暂无）')

        bio = io.BytesIO()
        doc.save(bio); bio.seek(0)
        return send_file(bio, as_attachment=True, download_name=filename,
                         mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- 在线职位匹配 ----------
# ---------- 在线职位匹配（Jooble + 回退 Arbeitnow） ----------
@app.post("/job_search")
def job_search():
    """
    入参：{ resume_struct, target_role, location, industry, limit }
    返回：{jobs: [{title, company, location, link, source, desc, score}]}
    """
    try:
        js = request.get_json(force=True)
        resume_struct = js.get("resume_struct", {})
        target_role = (js.get("target_role") or "").strip()
        location = (js.get("location") or "").strip()
        industry = (js.get("industry") or "").strip()
        limit = int(js.get("limit", 10))

        # 1) 关键词集合（来自解析 + 目标职位）
        keywords = set()
        for k in (resume_struct.get("skills_core") or []) + (resume_struct.get("keywords") or []):
            if isinstance(k, str) and 1 <= len(k) <= 30:
                keywords.add(k.lower())
        if target_role:
            for t in re.split(r"[ /,;、，]+", target_role):
                if t: keywords.add(t.lower())
        kw_list = list(keywords)[:8]
        results = []

        # 2) Jooble（需要环境变量 JOOBLE_API_KEY）
        if JOOBLE_API_KEY:
            payload = {
                "keywords": target_role or " ".join(kw_list) or "engineer",
                "location": location or "",
                "page": 1,
                "searchMode": 1,
                "radius": 40
            }
            try:
                r = requests.post(JOOBLE_ENDPOINT.format(JOOBLE_API_KEY), json=payload, timeout=30)
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

        # 3) 回退源：Arbeitnow
        if not results:
            try:
                url = "https://www.arbeitnow.com/api/job-board-api"
                r = requests.get(url, timeout=25)
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

        # 4) 简单匹配度
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

# ---------- 全局兜底 ----------
@app.errorhandler(Exception)
def handle_any_error(e):
    code = e.code if isinstance(e, HTTPException) else 500
    return jsonify({"error": str(e)}), code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "Alsos NeuroMatch"})

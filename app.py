import os
import json
import re
import time
from typing import Any, Dict, Tuple

import requests
from flask import Flask, render_template, request, make_response

# ==============================
# 配置 & 常量
# ==============================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL_NAME       = os.getenv("MODEL_NAME", "deepseek-chat")
JOOBLE_API_KEY   = os.getenv("JOOBLE_API_KEY", "")

HTTP_TIMEOUT     = 120      # 单次 HTTP 请求超时（秒）
MAX_TOKENS       = 1800     # LLM 最大 tokens（DeepSeek 免费层建议不要太大）
RETRY_TIMES      = 2        # LLM 调用重试次数
RETRY_BACKOFF    = 1.5      # 退避

app = Flask(__name__, template_folder="templates", static_folder=None)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 允许较大文本

# 统一响应头（避免代理缓存/缓冲导致“Failed to fetch”）
@app.after_request
def add_headers(resp):
    # 通用头
    resp.headers['Cache-Control'] = 'no-store'
    resp.headers['Connection'] = 'keep-alive'
    resp.headers['X-Accel-Buffering'] = 'no'

    # 如果是 HTML，就保留 text/html，别改成 json
    content_type = (resp.headers.get('Content-Type') or '').lower()
    if 'text/html' in content_type:
        return resp

    # 其他（JSON 接口）统一声明为 JSON
    resp.headers['Content-Type'] = 'application/json; charset=utf-8'
    return resp



# ==============================
# 工具函数
# ==============================
def _extract_json_block(text: str) -> str:
    """
    从模型返回中提取 JSON 代码块（```json ... ``` 或首个 { ... }）。
    """
    if not text:
        return ""
    # 优先找 ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.M)
    if m:
        return m.group(1)
    # 次选首个 { ... } 大括号
    m2 = re.search(r"(\{(?:[^{}]|(?1))*\})", text, flags=re.S)
    return m2.group(1) if m2 else text


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # 有时会以单引号或 JSON5 风格返回，尽量修一下
        s2 = s.replace("\n", " ").replace("\r", " ")
        s2 = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s2)  # key 补双引号
        s2 = s2.replace("'", '"')
        return json.loads(s2)


def call_deepseek(system_prompt: str, user_prompt: str) -> Tuple[bool, str]:
    """
    以 OpenAI 兼容接口调用 DeepSeek。返回 (ok, text)
    """
    url = f"{OPENAI_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user",   "content": user_prompt.strip()},
        ],
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "text"},
    }

    last_err = ""
    for i in range(RETRY_TIMES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code} {r.text[:300]}"
                time.sleep(RETRY_BACKOFF * (i + 1))
                continue
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return True, text
        except Exception as e:
            last_err = str(e)
            time.sleep(RETRY_BACKOFF * (i + 1))
    return False, last_err


def build_prompt(resume_text: str, target_title: str, target_city: str, target_industry: str, jd_text: str) -> str:
    """
    根据是否提供 JD，生成不同的要求。模型最终必须输出 JSON。
    """
    jd_part = "（未提供 JD；请不要做 ATS 评分，也不要强行匹配 JD。）"
    ats_req = ""
    if jd_text.strip():
        jd_part = f"（已提供 JD，见下文《职位 JD》。）"
        ats_req = """
- 做一段 **ATS 匹配**（0-100 分），并给出：
  - `score`：整数；
  - `highlights`：3-6 条优势要点；
  - `keywords`：建议在简历中补充/加粗的关键词（逗号分隔即可）；
  - `mismatch`：2-5 条不匹配点 + 补救建议（用中文短句）。
        """

    return f"""
你是一名资深猎头 + 简历教练。你将基于候选人的中文简历，输出**结构化 JSON**（UTF-8），字段名用英文。
不要输出任何说明文字或 Markdown，只输出 JSON 对象。严格遵守 JSON 语法。

场景：求职系统是一键生成，不希望多步操作。{jd_part}
期望输出四个板块（**resume_optimization**, **interview**, **career_plan**, 以及有 JD 时的 **ats**）：

1) resume_optimization（简历优化）
   - bullets_to_add：建议增加的要点（2-6 条，短句）
   - bullets_to_tighten：建议压缩/改写的表述（2-6 条）
   - section_order：建议的简历板块顺序（如 ["basics","summary","education","experiences","skills_core"]）
   - skills_keywords_core：该候选人**核心技能关键词**（6-15 个）
   - summary_cn：精炼的中文 2-3 句“个人简介”
   - summary_en：等价英文（1-2 句）

2) interview（面试辅导）
   - top_questions：结合其履历与目标岗位，列出 5-8 个必问问题（问法简洁）
   - tips：3-6 条总体面试建议（中文短句）
   - drills：如需准备的“演练/作业”（2-5 条）

3) career_plan（个性化职业建议，**有年龄/婚育可推断时可参考**）
   - career_paths：建议 1-2 条**不同方向**（例如“冲击管理岗 / 转咨询岗”），每条里包含：
     - title：方向名
     - why_now：为什么现在适合这样做（3-6 句）
     - 90_day_plan：入职/转型 90 天要做的 4-8 项
     - gap_to_fill：需要补齐的能力/证书/经验（3-8 条）
     - network_to_build：需要建立的人脉圈（3-6 类）
     - skills_to_learn：要学习的技能课程关键词（3-8 条）

4) {('ats（仅当有 JD）' if jd_text.strip() else 'ats（不需要输出此字段）')}
   {ats_req}

《候选人简历（原文）》
{resume_text}

《候选人目标信息》
- 目标职位：{target_title or '（未填）'}
- 期望地点：{target_city or '（未填）'}
- 目标行业：{target_industry or '（未填）'}

《职位 JD》
{jd_text or '（未提供）'}

请务必只输出 JSON 对象，不要任何解释。
    """.strip()


def post_process(json_obj: Dict[str, Any], has_jd: bool) -> Dict[str, Any]:
    """
    兜底：保证缺字段时也返回空结构，防止前端渲染炸掉。
    """
    safe = {
        "resume_optimization": {
            "bullets_to_add": [],
            "bullets_to_tighten": [],
            "section_order": [],
            "skills_keywords_core": [],
            "summary_cn": "",
            "summary_en": ""
        },
        "interview": {
            "top_questions": [],
            "tips": [],
            "drills": []
        },
        "career_plan": {
            "career_paths": []
        }
    }
    for k in safe:
        if k not in json_obj or not isinstance(json_obj[k], dict):
            json_obj[k] = safe[k]
        else:
            for kk, vv in safe[k].items():
                if kk not in json_obj[k]:
                    json_obj[k][kk] = vv

    if has_jd:
        json_obj.setdefault("ats", {"score": 0, "highlights": [], "keywords": [], "mismatch": []})
    else:
        # 没 JD 不返回 ats 字段，避免误导
        json_obj.pop("ats", None)
    return json_obj


# ==============================
# 路由
# ==============================
@app.route("/", methods=["GET"])
def index():
    # 首页交给模板
    resp = make_response(render_template("index.html"))
    # 首页返回 HTML，不受 after_request 的 json 头影响
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


@app.route("/health", methods=["GET"])
def health():
    raw = json.dumps({"ok": True, "ts": int(time.time())}, ensure_ascii=False)
    return make_response(raw, 200)


@app.route("/full_report", methods=["POST"])
def full_report():
    try:
        payload = request.get_json(force=True, silent=False)
        resume_text     = (payload.get("resume_text") or "").strip()
        target_title    = (payload.get("target_title") or "").strip()
        target_city     = (payload.get("target_city") or "").strip()
        target_industry = (payload.get("target_industry") or "").strip()
        jd_text         = (payload.get("jd_text") or "").strip()

        sys_prompt = "你是资深猎头/简历教练，严格输出 JSON。"
        user_prompt = build_prompt(resume_text, target_title, target_city, target_industry, jd_text)

        ok, text = call_deepseek(sys_prompt, user_prompt)
        if not ok:
            raw = json.dumps({"error": f"LLM 调用失败：{text}"}, ensure_ascii=False)
            return make_response(raw, 500)

        # 提取并解析 JSON
        json_str = _extract_json_block(text)
        try:
            obj = _safe_json_loads(json_str)
        except Exception as e:
            # 把原始返回片段也带回去，前端会弹窗显示
            raw = json.dumps({
                "error": f"模型未返回有效 JSON：{e}",
                "fragment": text[:1200]
            }, ensure_ascii=False)
            return make_response(raw, 500)

        obj = post_process(obj, has_jd=bool(jd_text))
        raw = json.dumps(obj, ensure_ascii=False)
        return make_response(raw, 200)
    except Exception as e:
        raw = json.dumps({"error": f"后端异常：{e}"}, ensure_ascii=False)
        return make_response(raw, 500)


if __name__ == "__main__":
    # 本地调试
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=False)

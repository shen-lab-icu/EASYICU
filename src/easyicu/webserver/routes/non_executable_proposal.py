"""Read-only candidate display, with all authority resolved on the host."""
from html import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from easyicu.webserver.non_executable_proposal_view import project_proposal
from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError

router = APIRouter()
_FIELD_LABELS = {
    "question": "研究问题",
    "cohort": "研究人群",
    "primary_exposure": "暴露",
    "outcome": "结局",
}


@router.get('/api/pi/proposal/{project_id}/{session_id}/{run_id}', response_class=HTMLResponse)
def proposal_view(project_id: str, session_id: str, run_id: str):
    try:
        payload = project_proposal(service=get_pi_copilot_service(), project_id=project_id,
                                   session_id=session_id, run_id=run_id)
    except (ValueError, PiCopilotError) as exc:
        raise HTTPException(status_code=409, detail="候选草案暂不可用，请核对项目、会话与研究配置。") from exc
    proposal = payload["proposal"]
    selected = {row['decision_id']: row for row in proposal['decisions']}
    rows = []
    for field, label in _FIELD_LABELS.items():
        row = selected.get(field)
        value = row['text'] if row else ''
        # The host projection includes a human-readable prefix; show it once.
        if value.startswith(f'{label}：'):
            value = value[len(label) + 1:]
        detail = escape(value) if value else '尚未填写'
        status = '人群定义尚未核验；来源与文献尚未逐项绑定' if field == 'cohort' else '来源与文献：尚未逐项绑定'
        rows.append(f'<article><h2>{escape(label)}</h2><p>{detail}</p>'
                    f'<p>{status}</p></article>')
    for row in proposal['decisions']:
        if row['decision_id'] not in _FIELD_LABELS:
            rows.append(f'<article><h2>{escape(row["decision_id"])}</h2><p>{escape(row["text"])}</p>'
                        '<p>来源与文献：尚未逐项绑定</p></article>')
    original = escape(proposal['original_question']) if proposal['original_question'] else '未能核实'
    candidates = payload.get('literature_candidates')
    if candidates:
        keys = candidates['retained_citation_keys']
        literature = f'<p>本次运行留存 {len(keys)} 条可引用文献标识；尚未与具体决定绑定。</p>'
        literature += '<p>' + ', '.join(escape(key) for key in keys) + '</p>'
    else:
        literature = '<p>本次运行尚无可核验的文献记录。</p>'
    return HTMLResponse('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font:16px/1.7 system-ui,sans-serif;background:#f8f8f5;color:#24302e;margin:0}'
        'main{max-width:820px;margin:0 auto;padding:32px 24px}h1{font-size:28px}'
        'h2{font-size:18px;margin-top:28px}article{background:white;border:1px solid #dde5e1;'
        'border-radius:12px;margin:12px 0;padding:12px 20px}article h2{margin:0;font-size:15px}'
        'article p:last-child{font-size:13px;color:#596864}p{overflow-wrap:anywhere}</style>'
        '<title>待审研究配置摘录</title><main><h1>待审研究配置摘录</h1>'
        '<p>这不是完整研究方案，仅供讨论；不可执行、未批准、未授权提取患者数据。</p>'
        f'<h2>原始问题</h2><p>{original}</p>{"".join(rows)}'
        f'<h2>已登记的文献候选</h2>{literature}'
        '<h2>仍需核验</h2><p>来源映射、每项决定的文献依据及时间定义尚未核验。</p>'
        '</main></html>', headers={"Cache-Control": "no-store"})

"""Read-only candidate display, with all authority resolved on the host."""
from html import escape
import re
from urllib.parse import parse_qs, quote, urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from easyicu.webserver.non_executable_proposal_view import (
    CandidateLiteratureReference, project_proposal, save_candidate_reference,
)
from easyicu.webserver.pi_copilot import get_pi_copilot_service
from easyicu.webserver.pi_copilot.contracts import PiCopilotError

router = APIRouter()
_FIELD_LABELS = {
    "question": "研究问题",
    "cohort": "研究人群",
    "primary_exposure": "暴露",
    "outcome": "结局",
    "method_background": "方法背景",
}
_ROLE_LABELS = {
    'direct_comparator': '直接对照候选', 'design_analogue': '设计类比',
    'definition': '定义', 'method': '方法背景', 'database': '数据库',
    'related_context': '相关背景',
}


def _literature_html(candidates: dict) -> str:
    """Render recorded provenance without implying independent support."""
    keys = candidates['retained_citation_keys']
    parts = [f'<p>本次运行留存 {len(keys)} 条可引用文献标识；文献清单本身不代表支持具体决定。</p>']
    if 'screened' not in candidates:
        parts.append('<p>' + ', '.join(escape(key) for key in keys) + '</p>')
        return ''.join(parts)
    parts.append(
        f'<p>本次筛选 {candidates["screened"]} 条，纳入 {candidates["included"]} 条，'
        f'排除 {candidates["excluded"]} 条；纳入的直接对照候选 '
        f'{candidates["included_direct_comparators"]} 条。筛选结果不代表相关研究不存在。</p>'
    )
    parts.append('<p>以下是本次记录的筛选角色与文献标识；摘要卡不是全文审阅，'
                 '列出文献也不等于它支持本方案。</p>')
    for record in candidates['records']:
        title = escape(str(record.get('title') or record['key']))
        key = escape(record['key'])
        screen = record.get('screening')
        status = '已纳入' if screen == 'include' else ('已排除' if screen == 'exclude' else '本次未记录筛选')
        role = _ROLE_LABELS.get(record.get('evidence_role'), record.get('evidence_role') or '未标记角色')
        details = [f'筛选记录：{status}', f'角色：{escape(str(role))}', f'标识：{key}']
        if record.get('year'):
            details.append(f'年份：{escape(str(record["year"]))}')
        if record.get('outcome_match') is not None:
            details.append('结局匹配：' + ('是' if record['outcome_match'] else '否'))
        if record.get('card_access_mode'):
            access = '摘要' if record['card_access_mode'] == 'abstract' else str(record['card_access_mode'])
            details.append(f'证据卡范围：{escape(access)}')
        links = []
        pmid = str(record.get('pmid') or '')
        if re.fullmatch(r'\d{1,12}', pmid):
            links.append(f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/">PMID {pmid}</a>')
        elif pmid:
            details.append(f'PMID：{escape(pmid)}')
        doi = str(record.get('doi') or '')
        if re.fullmatch(r'10\.\d{4,9}/[^\s<>"\']{1,200}', doi, flags=re.I):
            links.append(f'<a href="https://doi.org/{escape(quote(doi, safe="/"), quote=True)}">DOI {escape(doi)}</a>')
        elif doi:
            details.append(f'DOI：{escape(doi)}')
        parts.append(f'<article><h3>{title}</h3><p>{" · ".join(details)}</p>'
                     f'<p>{" · ".join(links)}</p>')
        if record.get('screening_rationale'):
            parts.append(f'<p>筛选理由：{escape(str(record["screening_rationale"]))}</p>')
        parts.append('</article>')
    if candidates['records_total'] > len(candidates['records']):
        parts.append(f'<p>本页仅显示前 {len(candidates["records"])} 条，请查看原始文献证据了解其余记录。</p>')
    return ''.join(parts)


def _binding_status(row: dict | None) -> str:
    bindings = (row or {}).get('bindings') or []
    parts = []
    for binding in bindings:
        if binding.get('kind') == 'literature':
            role = _ROLE_LABELS.get(binding.get('literature_role'), '未标记角色')
            parts.append('<p>候选绑定：已登记摘要原文与筛选记录，'
                         f'{escape(str(binding.get("citation_key") or ""))} · {escape(role)}；'
                         '该片段对本决定的适用性尚未核验。</p>'
                         f'<p>摘要证据 ID：{escape(str(binding.get("source_evidence_id") or ""))} · '
                         f'SHA-256：{escape(str(binding.get("source_evidence_sha256") or ""))}</p>'
                         f'<blockquote>{escape(str(binding.get("verbatim_excerpt") or ""))}</blockquote>')
    if not parts:
        parts.append('<p>文献：尚未逐项绑定</p>')
    if not any(binding.get('kind') == 'source' for binding in bindings):
        parts.append('<p>来源：尚未逐项绑定</p>')
    return ''.join(parts)


@router.get('/api/pi/proposal/{project_id}/{session_id}/{run_id}', response_class=HTMLResponse)
def proposal_view(project_id: str, session_id: str, run_id: str):
    try:
        payload = project_proposal(service=get_pi_copilot_service(), project_id=project_id,
                                   session_id=session_id, run_id=run_id)
    except (ValueError, PiCopilotError) as exc:
        raise HTTPException(status_code=409, detail="候选草案暂不可用，请核对项目、会话与研究配置。") from exc
    return _render_proposal(payload, project_id, session_id, run_id)


@router.post('/api/pi/proposal/{project_id}/{session_id}/{run_id}/candidate', response_class=HTMLResponse)
async def preview_candidate_reference(project_id: str, session_id: str, run_id: str, request: Request):
    """Validate a transient suggestion against host-bound evidence; never approve or persist it."""
    if request.headers.get('content-type', '').split(';', 1)[0] != 'application/x-www-form-urlencoded':
        raise HTTPException(status_code=415, detail='请使用页面表单提交候选绑定。')
    body = await request.body()
    if len(body) > 4096:
        raise HTTPException(status_code=413, detail='候选绑定内容过长。')
    try:
        fields = parse_qs(body.decode('utf-8'), keep_blank_values=True, strict_parsing=True, max_num_fields=4)
        if set(fields) != {'decision_id', 'citation_key', 'source_evidence_id', 'verbatim_excerpt'} or any(
                len(values) != 1 for values in fields.values()):
            raise ValueError('invalid_candidate_fields')
        suggestion = CandidateLiteratureReference.model_validate({key: values[0] for key, values in fields.items()})
        payload = project_proposal(service=get_pi_copilot_service(), project_id=project_id,
            session_id=session_id, run_id=run_id, proposed=suggestion)
    except (ValueError, UnicodeError, PiCopilotError) as exc:
        raise HTTPException(status_code=409, detail='候选绑定未通过核验。请核对决定、文献、已登记摘要及逐字片段。') from exc
    return _render_proposal(payload, project_id, session_id, run_id, transient=True)


@router.post('/api/pi/proposal/{project_id}/{session_id}/{run_id}/candidate/save', response_class=HTMLResponse)
async def save_candidate(project_id: str, session_id: str, run_id: str, request: Request):
    if request.headers.get('origin') != f'{request.url.scheme}://{request.url.netloc}':
        raise HTTPException(status_code=403, detail='请从当前页面保存候选绑定。')
    if request.headers.get('content-type', '').split(';', 1)[0] != 'application/x-www-form-urlencoded':
        raise HTTPException(status_code=415, detail='请使用页面表单提交候选绑定。')
    body = await request.body()
    if len(body) > 4096:
        raise HTTPException(status_code=413, detail='候选绑定内容过长。')
    try:
        fields = parse_qs(body.decode('utf-8'), keep_blank_values=True, strict_parsing=True, max_num_fields=4)
        if set(fields) != {'decision_id', 'citation_key', 'source_evidence_id', 'verbatim_excerpt'} or any(
                len(values) != 1 for values in fields.values()):
            raise ValueError('invalid_candidate_fields')
        suggestion = CandidateLiteratureReference.model_validate({key: values[0] for key, values in fields.items()})
        payload = save_candidate_reference(service=get_pi_copilot_service(), project_id=project_id,
            session_id=session_id, run_id=run_id, proposed=suggestion)
    except (ValueError, UnicodeError, PiCopilotError) as exc:
        raise HTTPException(status_code=409, detail='无法保存候选绑定，请核对摘要原文和当前研究配置。') from exc
    return _render_proposal(payload, project_id, session_id, run_id)


@router.post('/api/pi/proposal/{project_id}/{session_id}/{run_id}/pubmed-abstract', response_class=HTMLResponse)
async def collect_pubmed_abstract(project_id: str, session_id: str, run_id: str, request: Request):
    if request.headers.get('origin') != f'{request.url.scheme}://{request.url.netloc}':
        raise HTTPException(status_code=403, detail='请从当前页面登记摘要。')
    if request.headers.get('content-type', '').split(';', 1)[0] != 'application/x-www-form-urlencoded':
        raise HTTPException(status_code=415, detail='请使用页面表单获取原摘要。')
    body = await request.body()
    if len(body) > 1024:
        raise HTTPException(status_code=413, detail='文献标识过长。')
    try:
        fields = parse_qs(body.decode('utf-8'), keep_blank_values=True, strict_parsing=True, max_num_fields=1)
        if set(fields) != {'citation_key'} or len(fields['citation_key']) != 1:
            raise ValueError('invalid_citation')
        citation_key = fields['citation_key'][0]
        if not 1 <= len(citation_key) <= 200:
            raise ValueError('invalid_citation')
        payload = project_proposal(service=get_pi_copilot_service(), project_id=project_id,
            session_id=session_id, run_id=run_id, collect_citation_key=citation_key)
    except (ValueError, UnicodeError, PiCopilotError) as exc:
        raise HTTPException(status_code=409, detail='无法登记摘要：请核对纳入记录、PubMed 来源及网络连接。') from exc
    return _render_proposal(payload, project_id, session_id, run_id)


def _render_proposal(payload: dict, project_id: str, session_id: str, run_id: str, *, transient: bool = False):
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
        status = ('<p>人群定义尚未核验</p>' if field == 'cohort' else '') + _binding_status(row)
        rows.append(f'<article><h2>{escape(label)}</h2><p>{detail}</p>'
                    f'{status}</article>')
    for row in proposal['decisions']:
        if row['decision_id'] not in _FIELD_LABELS:
            rows.append(f'<article><h2>{escape(row["decision_id"])}</h2><p>{escape(row["text"])}</p>'
                        f'{_binding_status(row)}</article>')
    original = escape(proposal['original_question']) if proposal['original_question'] else '未能核实'
    return_href = '/?' + urlencode({'pi_project': project_id, 'pi_session': session_id}) + '#guided'
    candidates = payload.get('literature_candidates')
    if candidates:
        literature = _literature_html(candidates)
    else:
        literature = '<p>本次运行尚无可核验的文献记录。</p>'
    candidate_href = '/api/pi/proposal/' + '/'.join(quote(value, safe='') for value in
        (project_id, session_id, run_id)) + '/candidate'
    collect_href = candidate_href.removesuffix('/candidate') + '/pubmed-abstract'
    options = ''.join(f'<option value="{escape(key, quote=True)}">{escape(label)}</option>'
                      for key, label in _FIELD_LABELS.items() if key in selected)
    default_key = escape(str(payload.get('candidate_abstract_citation_key') or ''), quote=True)
    default_evidence = escape(str(payload.get('candidate_abstract_evidence_id') or ''), quote=True)
    default_excerpt = escape(str(payload.get('candidate_abstract_excerpt') or ''))
    form = (f'<form method="post" action="{escape(candidate_href, quote=True)}">'
            '<label>研究决定 <select name="decision_id">' + options + '</select></label>'
            f'<label>文献标识 <input name="citation_key" value="{default_key}" required maxlength="200"></label>'
            f'<label>已登记原摘要证据 ID <input name="source_evidence_id" value="{default_evidence}" required maxlength="200"></label>'
            f'<label>原摘要逐字片段 <textarea name="verbatim_excerpt" required minlength="12" maxlength="240">{default_excerpt}</textarea></label>'
            '<button type="submit">核验候选绑定</button> '
            f'<button type="submit" formaction="{escape(candidate_href + "/save", quote=True)}">保存待审候选</button></form>') if options else ''
    preview = '<p>当前是临时候选预览，未保存、未审核，也不授权执行。</p>' if transient else ''
    collected = payload.get('candidate_abstract_evidence_id')
    collected_note = (f'<p>已登记候选原摘要，证据 ID：<strong>{escape(collected)}</strong>。'
                      '请使用该 ID 和原文片段核验候选绑定；适用性仍需审核。</p>') if collected else ''
    saved = payload.get('candidate_binding_evidence_id')
    saved_note = (f'<p>候选绑定已保存（{escape(saved)}），适用性未审核。</p>') if saved else ''
    collect_form = (f'<form method="post" action="{escape(collect_href, quote=True)}">'
        '<label>已纳入的 PubMed 文献标识 <input name="citation_key" required maxlength="200"></label>'
        '<button type="submit">获取并登记原摘要</button></form>')
    return HTMLResponse('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font:16px/1.7 system-ui,sans-serif;background:#f8f8f5;color:#24302e;margin:0}'
        'main{max-width:820px;margin:0 auto;padding:32px 24px}h1{font-size:28px}'
        'h2{font-size:18px;margin-top:28px}article{background:white;border:1px solid #dde5e1;'
        'border-radius:12px;margin:12px 0;padding:12px 20px}article h2{margin:0;font-size:15px}'
        'article p:last-child{font-size:13px;color:#596864}p{overflow-wrap:anywhere}'
        'form label{display:block;margin:12px 0}input,select,textarea{display:block;width:100%;'
        'box-sizing:border-box;padding:8px}button{padding:9px 16px}</style>'
        '<title>待审研究配置摘录</title><main>'
        f'<a href="{escape(return_href, quote=True)}">返回研究对话</a>'
        '<h1>待审研究配置摘录</h1>'
        '<p>这不是完整研究方案，仅供讨论；不可执行、未批准、未授权提取患者数据。</p>'
        f'{preview}{saved_note}'
        f'<h2>原始问题</h2><p>{original}</p>{"".join(rows)}'
        f'<h2>已登记的文献候选</h2>{literature}'
        f'<h2>登记 PubMed 原摘要</h2><p>仅支持本次已纳入且身份可核验的 PubMed 文献；'
        f'摘要另存为当前会话的候选证据，不修改原运行。</p>{collected_note}{collect_form}'
        f'<h2>提出候选绑定</h2><p>填写已登记原摘要证据和逐字片段，核验仅供预览；'
        f'适用性仍需审核。当前没有已登记的原摘要时，请先取得并登记原始来源。</p>{form}'
        '<h2>仍需核验</h2><p>来源映射、每项决定的文献依据及时间定义尚未核验。</p>'
        '</main></html>', headers={"Cache-Control": "no-store"})

"""Register a public PubMed abstract as an unscreened candidate, not plan authority."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.literature import PubMedLiteratureClient


def register_pubmed_candidate(*, pmid: str, root: Path) -> tuple[str, dict]:
    if not re.fullmatch(r'[0-9]{1,12}', pmid, flags=re.ASCII):
        raise ValueError('proposal_candidate_pmid_invalid')
    client = PubMedLiteratureClient(timeout=15)
    params = client._with_etiquette({'db': 'pubmed', 'id': pmid, 'retmode': 'xml'})
    body = None
    for _ in range(3):
        body = client._http_get('efetch.fcgi', params)
        if body:
            break
    if not body or len(body) > 5_000_000:
        raise ValueError('proposal_pubmed_unavailable')
    try:
        root_xml = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError('proposal_pubmed_invalid') from exc
    articles = root_xml.findall('.//PubmedArticle')
    if len(articles) != 1:
        raise ValueError('proposal_pubmed_identity_mismatch')
    article = articles[0]
    fetched_pmid = ''.join(article.findtext('./MedlineCitation/PMID', default='').split())
    title_node = article.find('./MedlineCitation/Article/ArticleTitle')
    title = ' '.join(''.join(title_node.itertext()).split()) if title_node is not None else ''
    abstract = ' '.join(' '.join(''.join(node.itertext()).split())
                        for node in article.findall('./MedlineCitation/Article/Abstract/AbstractText')).strip()
    dois = [''.join(node.itertext()).strip() for node in article.findall('./PubmedData/ArticleIdList/ArticleId')
            if node.get('IdType') == 'doi']
    if fetched_pmid != pmid or not title or not abstract or len(dois) > 1:
        raise ValueError('proposal_pubmed_identity_mismatch')
    content = {'schema_version': 'easyicu.pubmed-unscreened-candidate/1', 'citation_key': f'pmid:{pmid}',
               'pmid': pmid, 'doi': dois[0] if dois else None, 'title': title, 'abstract': abstract,
               'source_url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
               'retrieved_at': datetime.now(timezone.utc).isoformat(),
               'screening': 'not_screened', 'applicability_verified': False}
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True)
    eid = 'candidate_pubmed_' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]
    EvidenceStore(root).register_text(kind='log', description='Unscreened PubMed candidate; applicability not reviewed',
        text=encoded, filename='pubmed_candidate.json', evidence_id=eid)
    return eid, content

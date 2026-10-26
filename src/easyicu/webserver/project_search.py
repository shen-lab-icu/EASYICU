"""Host-owned, bounded in-process search; refresh metadata on demand, no daemon."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .search_contracts import MAX_HITS, MAX_QUERY, normalized, terms, tokens
from .search_sources import SearchCatalog


@dataclass(frozen=True)
class SearchLimits:
    documents: int = 10_000
    text_bytes: int = 8 * 1024 * 1024
    postings: int = 250_000

    def __post_init__(self):
        if min(self.documents, self.text_bytes, self.postings) < 1:
            raise ValueError("search_limit_invalid")


class SearchIndex:
    """Caches only allowlisted projected text, with independent hard budgets.

    Catalog membership is checked each query. Changed inode/size/mtime/ctime
    replaces that source's projection; deleted sources disappear immediately.
    A restart lazily rebuilds the index. There is no disk copy of search text.
    """

    def __init__(self, *, limits: SearchLimits | None = None):
        self.limits = limits or SearchLimits()
        self._cache = {}
        self._owner = None
        self._lock = threading.RLock()
        self.stats = dict(documents=0, text_bytes=0, postings=0)

    def search(self, q: str, *, scope: str | None = None, project: str | None = None, limit: int = 20) -> dict:
        if not isinstance(q, str) or len(q) > MAX_QUERY:
            raise ValueError("search_query_invalid")
        if scope not in (None, "tasks", "files", "projects"):
            raise ValueError("search_scope_invalid")
        if project is not None and (not isinstance(project, str) or len(project) > 160):
            raise ValueError("search_project_invalid")
        if type(limit) is not int or not 1 <= limit <= MAX_HITS:
            raise ValueError("search_limit_invalid")
        result = dict(hits=[], has_more=False, index_truncated=False, issues=[])
        if not terms(q):
            return result
        with self._lock:
            catalog = SearchCatalog()
            owner = (catalog.state.resolve(), catalog.projects.resolve())
            if self._owner != owner:
                self._cache.clear()
                self._owner = owner
            catalog.discover()
            documents, inverted, cache = {}, {}, {}
            used_bytes = used_postings = 0

            def add(document):
                nonlocal used_bytes, used_postings
                identity = document.identity
                if identity in documents:
                    return True
                text = document.searchable
                words = tokens(text)
                size = len(text.encode("utf-8"))
                if (len(documents) >= self.limits.documents or used_bytes + size > self.limits.text_bytes
                        or used_postings + len(words) > self.limits.postings):
                    catalog.issues.add("index_limit")
                    return False
                used_bytes += size
                used_postings += len(words)
                documents[identity] = document
                for word in words:
                    inverted.setdefault(word, set()).add(identity)
                return True

            for document in catalog.documents:
                if not add(document):
                    break
            for source in catalog.sources:
                stamp = catalog.stamp(source.root, source.path)
                old = self._cache.get(source)
                if old and old[0] == stamp and old[1]:
                    projected, issues = old[1:]
                    catalog.issues.update(issues)
                else:
                    before = catalog.issues.copy()
                    projected = catalog.load(source)
                    issues = catalog.issues - before
                kept = []
                for document in projected:
                    if not add(document):
                        break
                    kept.append(document)
                else:
                    cache[source] = (stamp, tuple(kept), issues)
                    continue
                # A partially admitted source must be reconsidered if space frees.
                break
            self._cache = cache
            self.stats = dict(documents=len(documents), text_bytes=used_bytes, postings=used_postings)
            candidates = None
            for word in tokens(q):
                posting = inverted.get(word, set())
                candidates = posting.copy() if candidates is None else candidates & posting
            query_terms = terms(q)
            matches = [documents[key] for key in candidates or ()
                       if (scope is None or documents[key].scope == scope)
                       and (not project or documents[key].project_id == project)
                       and all(term in normalized(documents[key].searchable) for term in query_terms)]
            matches.sort(key=lambda doc: (
                -sum(term in normalized(doc.title) for term in query_terms), doc.project_id, doc.identity,
            ))
            return dict(hits=[doc.hit(q) for doc in matches[:limit]], has_more=len(matches) > limit,
                        index_truncated=bool(catalog.issues), issues=sorted(catalog.issues))


INDEX = SearchIndex()

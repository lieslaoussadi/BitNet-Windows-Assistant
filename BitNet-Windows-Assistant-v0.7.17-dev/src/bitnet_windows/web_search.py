"""Small opt-in public Web search using HTML pages, not an AI/search API."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse, parse_qs, urlunparse, unquote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


class _DuckParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.results=[]; self._title=None; self._snippet=None; self._capture=None; self._depth=0
    def handle_starttag(self, tag, attrs):
        a=dict(attrs); classes=a.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self._title={"url":a.get("href", ""), "text": ""}; self._capture="title"
        elif tag in ("a", "div", "td") and ("result__snippet" in classes or "result-snippet" in classes):
            self._snippet={"text":""}; self._capture="snippet"
    def handle_data(self, data):
        if self._capture == "title" and self._title is not None: self._title["text"] += data
        elif self._capture == "snippet" and self._snippet is not None: self._snippet["text"] += data
    def handle_endtag(self, tag):
        if tag == "a" and self._capture == "title" and self._title is not None:
            url=self._clean_url(self._title["url"]); title=" ".join(self._title["text"].split())
            if url and title: self.results.append({"title":title,"url":url,"snippet":""})
            self._title=None; self._capture=None
        elif self._capture == "snippet" and tag in ("a", "div", "td") and self._snippet is not None:
            text=" ".join(self._snippet["text"].split())
            if text and self.results and not self.results[-1]["snippet"]: self.results[-1]["snippet"]=text
            self._snippet=None; self._capture=None
    @staticmethod
    def _clean_url(raw):
        raw=raw.strip()
        if raw.startswith("//"): raw="https:"+raw
        elif raw.startswith("/"): raw="https://duckduckgo.com"+raw
        parsed=urlparse(raw)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            raw=parse_qs(parsed.query).get("uddg", [""])[0]; parsed=urlparse(raw)
        if parsed.scheme not in ("http", "https") or not parsed.netloc: return ""
        if parsed.netloc.endswith("duckduckgo.com"): return ""
        return raw


class _BingParser(HTMLParser):
    """Extract ordinary public result links from Bing's HTML page."""
    def __init__(self):
        super().__init__(); self.results=[]; self._capture=None; self._current=None; self._snippet_depth=0
    def handle_starttag(self, tag, attrs):
        a=dict(attrs); classes=a.get("class", "").split()
        if tag == "li" and "b_algo" in classes:
            self._current={"title":"", "url":"", "snippet":""}
        elif self._current is not None and tag == "a" and not self._current["title"]:
            href=a.get("href", ""); parsed=urlparse(href)
            if parsed.scheme in ("http", "https") and parsed.netloc and not parsed.netloc.endswith("bing.com"):
                self._current["url"]=href; self._capture="title"
        elif self._current is not None and tag in ("p", "div") and ("b_caption" in classes or tag == "p"):
            self._capture="snippet"
    def handle_data(self, data):
        if self._current is not None and self._capture in ("title", "snippet"):
            self._current[self._capture] += data
    def handle_endtag(self, tag):
        if self._current is not None and tag == "a" and self._capture == "title": self._capture=None
        if self._current is not None and tag in ("p", "div") and self._capture == "snippet": self._capture=None
        if self._current is not None and tag == "li":
            row={k:" ".join(v.split()) for k,v in self._current.items()}
            if row["url"] and row["title"]: self.results.append(row)
            self._current=None; self._capture=None


def _fetch_html(url: str, timeout: float) -> str:
    request=Request(url, headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36"})
    with urlopen(request, timeout=timeout) as response:
        status=getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"HTTP {status} (la page n'est pas disponible immédiatement).")
        return response.read(2_000_000).decode("utf-8", errors="replace")


def _canonical_url(raw: str) -> str:
    """Normalize URLs enough to avoid duplicate search results."""
    parsed = urlparse(raw.strip())
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return ""
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    # Tracking parameters often make the same page appear more than once.
    tracking = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
    query = [(k, v) for k, values in parse_qs(parsed.query, keep_blank_values=True).items()
             if k.lower() not in tracking for v in values]
    from urllib.parse import urlencode
    return urlunparse(("https", host, path, "", urlencode(query), ""))


def deduplicate_results(rows: list[dict], limit: int = 5) -> list[WebResult]:
    """Deduplicate by normalized URL and then title, preserving useful snippets."""
    results: list[WebResult] = []
    by_url: dict[str, int] = {}
    seen_titles: set[str] = set()
    max_results = max(1, min(int(limit), 10))
    for row in rows:
        title = " ".join(str(row.get("title", "")).split()).strip()
        target = str(row.get("url", "")).strip()
        snippet = " ".join(str(row.get("snippet", "")).split()).strip()
        key = _canonical_url(target)
        if not title or not key:
            continue
        title_key = title.casefold()
        if key in by_url:
            idx = by_url[key]
            old = results[idx]
            if len(snippet) > len(old.snippet):
                results[idx] = WebResult(old.title, old.url, snippet)
            continue
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        by_url[key] = len(results)
        results.append(WebResult(title, target, snippet))
        if len(results) >= max_results:
            break
    return results


def prepare_search_query(query: str, *, today: date | None = None) -> str:
    """Make relative-date weather queries explicit for search engines."""
    normalized = " ".join(query.split()).strip()
    if not normalized:
        return normalized
    today = today or date.today()
    tomorrow = today + timedelta(days=1)
    weather = re.search(r"\b(m[ée]t[ée]o|m[ée]t[ée]orologique|temps|temp[ée]rature|pluie|averses|pr[ée]visions?)\b", normalized, re.I)
    relative = re.search(r"\b(demain|tomorrow)\b", normalized, re.I)
    if weather and relative:
        date_text = tomorrow.strftime("%d/%m/%Y")
        # Search explicitly for daily forecast fields, not generic/current weather pages.
        return f"{normalized} prévisions météo journalières {date_text} température minimale maximale probabilité de pluie"
    return normalized


def search_web(query: str, *, limit: int = 5, timeout: float = 8.0) -> list[WebResult]:
    """Search public HTML pages with a fallback. Called only when Web is enabled."""
    query=prepare_search_query(query)
    if not query: raise ValueError("La recherche Web nécessite une question.")
    if len(query)>500: query=query[:500]
    engines = [
        ("DuckDuckGo", "https://html.duckduckgo.com/html/?q="+quote_plus(query), _DuckParser),
        ("Bing", "https://www.bing.com/search?q="+quote_plus(query), _BingParser),
    ]
    errors=[]
    for engine, url, parser_type in engines:
        try:
            html=_fetch_html(url, timeout)
            parser=parser_type(); parser.feed(html)
            results=deduplicate_results(parser.results, limit)
            if results: return results
            errors.append(f"{engine}: aucun résultat exploitable (page vide ou format modifié)")
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    details=" ; ".join(errors)
    raise RuntimeError("Recherche Web impossible avec les moteurs disponibles. Vérifie l'accès Internet de Python/BitNet, puis réessaie. Détails : "+details)


def format_web_context(results: list[WebResult]) -> str:
    lines=["Résultats de recherche Web : ce sont des titres et extraits de résultats, pas nécessairement le contenu complet des pages. Réponds seulement avec les faits réellement présents. Si une donnée importante manque ou si les sources se contredisent, indique-le au lieu d'inventer. Cite les sources par [W1], [W2], etc. La réponse est générée par le modèle local.",
           "Règles de qualité : ne déduis jamais une prévision du seul titre d'une page. Distingue météo actuelle et prévision du jour demandé. N'affirme pas 'sans pluie' sans indication explicite de précipitations. Ne transforme pas des valeurs horaires en minimum/maximum quotidiens sans preuve. Si les extraits ne confirment pas la date et les chiffres, dis clairement que les résultats récupérés ne permettent pas de confirmer la prévision."]
    if any(re.search(r"\b(m[ée]t[ée]o|temps|temp[ée]rature|pluie|averses|pr[ée]visions?)\b", r.title + " " + r.snippet, re.I) for r in results):
        lines.append("Pour une question météo : fournis la date exacte, la température minimale et maximale uniquement si elles sont explicitement confirmées, les conditions et le risque de pluie uniquement si la source le précise. Sinon indique les champs non confirmés. Ne présente pas une estimation inventée comme une donnée météorologique.")
    for i,r in enumerate(results,1):
        lines.append(f"[W{i}] {r.title}\nURL: {r.url}\nExtrait: {r.snippet or '(aucun extrait disponible)'}")
    return "\n\n".join(lines)


def format_web_references(results: list[WebResult]) -> str:
    unique = deduplicate_results([{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results], limit=10)
    return "Sources Web consultées (ouvrir les liens dans un navigateur) :\n"+"\n".join(f"[W{i}] {r.title} — {r.url}" for i,r in enumerate(unique,1))

import hashlib
import io
import time
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import requests #type:ignore
from bs4 import BeautifulSoup #type:ignore
from pypdf import PdfReader #type:ignore


OFFICIAL_DOMAINS = (
    "who.int",
    "nhs.uk",
    "cdc.gov",
    "nih.gov",
    "nice.org.uk",
)

GROUNDING_HOST = "vertexaisearch.cloud.google.com"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def official_domain(url):
    parts = urlsplit(url)

    if (
        parts.scheme != "https"
        or parts.username
        or parts.password
        or parts.port is not None
    ):
        return None

    host = (parts.hostname or "").lower()

    return next(
        (
            domain
            for domain in OFFICIAL_DOMAINS
            if host == domain or host.endswith("." + domain)
        ),
        None,
    )


def allowed_fetch_url(url):
    if official_domain(url):
        return True

    parts = urlsplit(url)

    return (
        parts.scheme == "https"
        and parts.hostname == GROUNDING_HOST
        and not parts.username
        and not parts.password
        and parts.port is None
        and parts.path.startswith("/grounding-api-redirect/")
    )


def parse_html(content, url):
    soup = BeautifulSoup(content, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else url
    )

    dates = {}

    for meta in soup.find_all("meta"):
        key = meta.get("property") or meta.get("name") or ""

        if any(
            term in key.lower()
            for term in ("published", "modified", "reviewed", "date")
        ):
            if meta.get("content"):
                dates[key] = meta["content"]

    for item in soup.find_all("time", datetime=True):
        dates.setdefault("time_datetime", item["datetime"])

    for node in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "form",
    ]):
        node.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    links = []

    for link in main.find_all("a", href=True):
        target = urljoin(url, link["href"])

        if (
            official_domain(target)
            and target not in [item["url"] for item in links]
        ):
            links.append({
                "url": target,
                "title": link.get_text(" ", strip=True)[:240],
            })

    paragraphs = [
        node.get_text(" ", strip=True)
        for node in main.find_all([
            "h1", "h2", "h3", "h4", "p", "li", "table"
        ])
    ]

    text = "\n\n".join(
        dict.fromkeys(paragraph for paragraph in paragraphs if paragraph)
    )

    if not text:
        text = main.get_text(" ", strip=True)

    return {
        "title": title,
        "text": text,
        "dates_as_reported": dates,
        "links": links[:30],
    }


def parse_pubmed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    records = []

    for item in root.findall(".//PubmedArticle"):
        citation = item.find("MedlineCitation")

        if citation is None:
            continue

        article = citation.find("Article")

        if article is None:
            continue

        pmid = citation.findtext("PMID")

        if not pmid:
            continue

        sections = []

        for node in article.findall("Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            label = node.get("Label")

            sections.append(
                (label + ": " if label else "") + text
            )

        title_node = article.find("ArticleTitle")
        journal_date = article.find("Journal/JournalIssue/PubDate")

        notices = [
            {
                "type": node.get("RefType"),
                "pmid": node.findtext("PMID"),
            }
            for node in citation.findall(
                "CommentsCorrectionsList/CommentsCorrections"
            )
        ]

        records.append({
            "kind": "pubmed_abstract",
            "pmid": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "title": (
                "".join(title_node.itertext())
                if title_node is not None
                else ""
            ),
            "text": "\n\n".join(sections),
            "organization": "PubMed record",
            "publication_types": [
                "".join(node.itertext())
                for node in article.findall(
                    "PublicationTypeList/PublicationType"
                )
            ],
            "dates_as_reported": (
                {
                    "journal_pubdate": " ".join(
                        journal_date.itertext()
                    )
                }
                if journal_date is not None
                else {}
            ),
            "correction_notices": notices,
            "retrieved_at": utc_now(),
            "text_scope": "abstract_only",
        })

    return records


class SourceTools:
    def __init__(self, llm, email, session=None):
        self.llm = llm
        self.email = email
        self.session = session or requests.Session()

        self.session.headers.update({
            "User-Agent": "HealthClaimResearch/0.2"
        })

        self.last_ncbi_request = 0.0

    def search_official(self, query, domain):
        if domain not in OFFICIAL_DOMAINS:
            raise ValueError("Unsupported official domain")

        # Grounded search discovers links; its generated summary
        # is deliberately not returned as scientific evidence.
        response = self.llm.grounded_search(
            f"Search site:{domain} for official pages or guidance "
            f"relevant to: {query}. "
            "Find sources, including evidence of no benefit or uncertainty. "
            "Do not invent URLs."
        )

        urls = []

        for candidate in response.candidates or []:
            metadata = candidate.grounding_metadata

            if not metadata:
                continue

            for chunk in metadata.grounding_chunks or []:
                web = chunk.web

                if not web or not web.uri:
                    continue

                if not allowed_fetch_url(web.uri):
                    continue

                if official_domain(web.uri) not in (None, domain):
                    continue

                if web.uri not in [item["url"] for item in urls]:
                    urls.append({
                        "url": web.uri,
                        "title": web.title or "",
                        "expected_domain": domain,
                        "origin": "grounded_search",
                    })

        return {
            "domain": domain,
            "query": query,
            "candidates": urls[:6],
            "status": (
                "discovery_only"
                if urls
                else "no_usable_grounded_links"
            ),
        }

    def fetch_official(self, candidate):
        url = candidate["url"]
        redirects = []

        for _ in range(6):
            if not allowed_fetch_url(url):
                raise ValueError(
                    "URL or redirect outside permitted official sources"
                )

            with self.session.get(
                url,
                timeout=(10, 30),
                stream=True,
                allow_redirects=False,
            ) as response:

                if response.status_code in (301, 302, 303, 307, 308):
                    target = response.headers.get("Location")

                    if not target:
                        raise ValueError(
                            "Redirect has no destination"
                        )

                    redirects.append(url)
                    url = urljoin(url, target)
                    continue

                response.raise_for_status()

                domain = official_domain(url)

                if (
                    domain is None
                    or domain != candidate["expected_domain"]
                ):
                    raise ValueError(
                        "Grounded link did not resolve to its "
                        "expected official domain"
                    )

                chunks = []
                size = 0

                for chunk in response.iter_content(65536):
                    size += len(chunk)

                    if size > 8 * 1024 * 1024:
                        raise ValueError(
                            "Document exceeds 8 MiB download limit"
                        )

                    chunks.append(chunk)

                raw = b"".join(chunks)

                mime = response.headers.get(
                    "Content-Type", ""
                ).lower()

            if "pdf" in mime or raw.startswith(b"%PDF"):
                reader = PdfReader(io.BytesIO(raw))

                if len(reader.pages) > 100:
                    raise ValueError(
                        "PDF exceeds 100-page extraction limit"
                    )

                parsed = {
                    "title": str(
                        (reader.metadata or {}).get("/Title") or url
                    ),
                    "text": "\n\n".join(
                        page.extract_text() or ""
                        for page in reader.pages
                    ),
                    "dates_as_reported": {},
                    "links": [],
                }

                kind = "official_pdf"

            elif (
                "html" in mime
                or raw.lstrip().lower().startswith(
                    (b"<!doctype html", b"<html")
                )
            ):
                parsed = parse_html(raw, url)
                kind = "official_html"

            else:
                raise ValueError(
                    "Unsupported document content type"
                )

            if not parsed["text"].strip():
                raise ValueError(
                    "No extractable text; scanned or dynamic source "
                    "requires another tool"
                )

            return {
                **parsed,
                "kind": kind,
                "url": url,
                "organization": domain,
                "retrieved_at": utc_now(),
                "raw_sha256": sha256(raw),
                "redirect_chain": redirects,
                "text_scope": "extracted_page_or_pdf",
            }

        raise ValueError("Too many redirects")

    def _ncbi(self, endpoint, params):
        # Stay below the unkeyed E-utilities request-rate limit.
        delay = 0.36 - (
            time.monotonic() - self.last_ncbi_request
        )

        if delay > 0:
            time.sleep(delay)

        self.last_ncbi_request = time.monotonic()

        response = self.session.get(
            (
                "https://eutils.ncbi.nlm.nih.gov/"
                f"entrez/eutils/{endpoint}"
            ),
            params={
                **params,
                "email": self.email,
                "tool": "health_claim_verifier",
            },
            timeout=(10, 30),
        )

        response.raise_for_status()
        return response

    def search_pubmed(self, query):
        if not self.email:
            raise ValueError(
                "Set NCBI_EMAIL before using the PubMed tool"
            )

        result = self._ncbi(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 5,
                "sort": "relevance",
            },
        ).json()

        search = result.get("esearchresult")

        if not isinstance(search, dict):
            raise ValueError("Invalid PubMed search response")

        if "ERROR" in result or search.get("ERROR"):
            raise ValueError("PubMed reported a query error")

        ids = search.get("idlist", [])
        records = []

        if ids:
            response = self._ncbi(
                "efetch.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                },
            )

            records = parse_pubmed(response.content)

        return {
            "query": query,
            "total_matches": search.get("count"),
            "query_translation": search.get("querytranslation"),
            "warnings": search.get("warninglist", {}),
            "requested_pmids": ids,
            "unresolved_pmids": sorted(
                set(ids) - {record["pmid"] for record in records}
            ),
            "records": records,
            "exhaustive": False,
        }
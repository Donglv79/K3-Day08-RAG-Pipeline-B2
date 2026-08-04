"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Phương pháp:
    1. Dùng Crawl4AI (AsyncWebCrawler) làm phương pháp chính.
    2. Nếu crawl4ai chưa có playwright / gặp lỗi (403, Executable doesn't exist...),
       tự động fallback về requests + HTMLParser (stdlib) để trích title + nội dung
       chính — vẫn ra đúng schema JSON. Các trang nguồn RMIT trả 200 cho requests
       thông thường nên fallback chạy chắc chắn.

Schema mỗi file JSON:
    {
        "url": str,
        "title": str,
        "date_crawled": str (ISO format),
        "content_markdown": str
    }
"""

import asyncio
import json
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Các bài viết (đã verify HTTP 200) — chủ đề university services (student life, thư viện)
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/news/all-news/2026/jul/rmit-student-finds-global-purpose-at-un-leadership-program",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2026/r-loop-event-recap",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/10-years-book-swap",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/rmit-vietnam-library-launches-adobe-express-champions",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/library-welcomes-visitors-from-can-tho-university",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


class ArticleHTMLParser(HTMLParser):
    """
    Trích tiêu đề + nội dung chính từ HTML về dạng markdown gọn.

    Chỉ lấy h1/h2/h3/p/li nằm trong <main> (nếu có) để loại bỏ nav menu —
    fallback về toàn trang nếu không tìm thấy <main>.
    """

    def __init__(self):
        super().__init__()
        self.title = ""
        self.lines = []
        self._in_title = False
        self._buf = []
        self._tag = None
        self._main_depth = None  # depth của thẻ <main>, None = chưa thấy
        self._depth = 0
        self._saw_main = False

    def _inside_main(self):
        if self._main_depth is None:
            return True  # chưa thấy <main> → emit hết (fallback)
        return self._depth >= self._main_depth

    def handle_starttag(self, tag, attrs):
        self._depth += 1
        classes = " ".join(v for k, v in attrs if k == "class").lower()
        is_container = tag in ("main", "article") or (
            tag == "div" and "text-component" in classes
        )
        if is_container and self._main_depth is None:
            self._saw_main = True
            self._main_depth = self._depth
            self.lines = []  # bỏ nội dung trước container (nav/header)
        if tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "p", "li") and self._inside_main():
            self._tag = tag
            self._buf = []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "p", "li") and self._tag == tag:
            text = " ".join("".join(self._buf).split())
            if text and self._inside_main():
                prefix = {"h1": "# ", "h2": "## ", "h3": "### ", "li": "- "}.get(tag, "")
                self.lines.append(prefix + text)
            self._tag = None
            self._buf = []
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._tag is not None:
            self._buf.append(data)


def fetch_article_requests(url: str) -> dict:
    """Fallback: tải HTML bằng requests và trích nội dung bằng stdlib parser."""
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding

    parser = ArticleHTMLParser()
    parser.feed(resp.text)

    title = (parser.title or "Unknown").strip()
    content = "\n\n".join(parser.lines).strip()
    if not content:
        raise RuntimeError(f"Không trích được nội dung từ {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Dùng Crawl4AI trước; nếu lỗi (thiếu playwright, 403...) thì fallback requests.
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success and result.markdown and len(result.markdown.strip()) > 200:
                return {
                    "url": url,
                    "title": getattr(result.metadata, "get", lambda k, d=None: None)(
                        "title", None
                    )
                    or "Unknown",
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
            raise RuntimeError("crawl4ai trả về nội dung rỗng/thất bại")
    except Exception as e:
        print(f"  ↻ crawl4ai không dùng được ({e.__class__.__name__}: {e}) — fallback requests")
        return fetch_article_requests(url)


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    ok, failed = 0, []
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
            print(f"  ✓ Saved: {filepath.name} ({len(article['content_markdown'])} chars)")
            ok += 1
        except Exception as e:
            print(f"  ✗ Lỗi {url}: {e}")
            failed.append(url)

    print(f"\n✓ Crawl xong {ok}/{len(ARTICLE_URLS)} bài vào {DATA_DIR}")
    if failed:
        print(f"✗ Thất bại: {failed}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        asyncio.run(crawl_all())

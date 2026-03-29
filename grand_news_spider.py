import scrapy
import random
import os
import json

def is_junk_url(url):
    url = url.lower()
    return any(x in url for x in [
        "live", "video", "podcast", "gallery",
        "/tag/", "/author/", "/topic/",
        "spreaker", "apple.com", "audio",
        ".mp3", ".mp4"
    ])
    
ALLOWED_DOMAINS = ["reuters.com", "theguardian.com", "aljazeera.com"]

def is_allowed(url):
    return any(domain in url for domain in ALLOWED_DOMAINS)

class GrandNewsSpider(scrapy.Spider):
    name = "grand_news"

    DOWNLOAD_DELAY = 0.5
    CONCURRENT_REQUESTS_PER_DOMAIN = 2

    TARGET_PER_SITE = 1000

    SOURCES = {
        "reuters": "https://www.reuters.com/world/",
        "guardian": "https://www.theguardian.com/international",
        #"bbc": "https://www.bbc.com/news",
        #"npr": "https://www.npr.org/sections/news/",
        "aljazeera": "https://www.aljazeera.com/news/"
    }

    counts = {}
    visited = set()

    def start_requests(self):
        for source, url in self.SOURCES.items():
            self.counts[source] = 0
            print(f"\n=== Starting {source} ===")
            yield scrapy.Request(url, callback=self.parse_listing, meta={"source": source})

    def parse_listing(self, response):
        source = response.meta["source"]

        links = response.css("a::attr(href)").getall()
        random.shuffle(links)

        for link in links:
            link = response.urljoin(link)
            if link.startswith("#"):
                continue
            if not is_allowed(link):
                continue
            if is_junk_url(link):
                continue
            if self.counts[source] >= self.TARGET_PER_SITE:
                return

            if not link:
                continue

            if link.startswith("/"):
                link = response.urljoin(link)

            # 🔥 skip junk pages
            if any(x in link for x in [
                "live", "video", "podcast", "gallery",
                "/tag/", "/author/", "/topic/"
            ]):
                continue

            # 🔥 avoid duplicates
            if link in self.visited:
                continue
            self.visited.add(link)

            # 🔥 basic article heuristic
            if not any(x in link for x in ["/202", "/news", "/article"]):
                continue

            yield scrapy.Request(
                link,
                callback=self.parse_article,
                meta={"source": source}
            )

        # 🔁 keep crawling deeper (important!)
        for next_link in links[:20]:
            if not next_link:
                continue

            next_link = response.urljoin(next_link)

            if next_link.startswith("#"):
                continue

            if not is_allowed(next_link):
                continue

            if is_junk_url(next_link):
                continue

            if next_link in self.visited:
                continue

            self.visited.add(next_link)

            yield scrapy.Request(
                next_link,
                callback=self.parse_listing,
                meta={"source": source}
            )

    def parse_article(self, response):
        if not is_allowed(response.url):
            return
        if is_junk_url(response.url):
            return
        source = response.meta["source"]

        if self.counts[source] >= self.TARGET_PER_SITE:
            return

        print("Parsing:", response.url)

        # 🧹 cleaner text extraction
        paragraphs = response.css("p::text").getall()
        text = " ".join([
            p.strip()
            for p in paragraphs
            if len(p.strip()) > 50
        ])

        if not text or len(text) < 300:
            return

        self.counts[source] += 1

        os.makedirs(f"Scrapy/{source}", exist_ok=True)

        filename = f"Scrapy/{source}/{self.counts[source]:04}_{abs(hash(response.url))}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "source": source,
                "url": response.url,
                "title": response.css("title::text").get(),
                "text": text
            }, f, indent=2)

        print(f"✅ Saved ({self.counts[source]}):", filename)
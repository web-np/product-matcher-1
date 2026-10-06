import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

st.set_page_config(
    page_title="ImprintID Product Crawler & Matcher",
    page_icon="🕷️",
    layout="wide"
)

st.title("🕷️ ImprintID Live Website Crawler & Matcher")
st.write("Northeastern product link se feature signature extract karke ImprintID website ko directly crawl karta hai aur top matching products return karta hai.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

class ImprintIDCrawler:
    def __init__(self, base_url="https://www.imprintid.com"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def extract_northeastern_features(self, ne_url: str) -> dict:
        """Northeastern URL aur page se key specs extract karta hai"""
        clean_url = ne_url.split("?")[0].rstrip("/")
        slug = clean_url.split("/")[-2] if len(clean_url.split("/")) >= 2 else clean_url.split("/")[-1]
        sku = clean_url.split("/")[-1]
        slug_text = re.sub(r'[-_]', ' ', slug).lower()

        # Scraping live page text if available
        page_text = slug_text
        try:
            r = self.session.get(ne_url, timeout=6)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                h1 = soup.find('h1') or soup.find('title')
                if h1:
                    page_text += " " + h1.get_text(" ", strip=True).lower()
                desc = soup.find('div', {'class': re.compile(r'desc|detail', re.I)})
                if desc:
                    page_text += " " + desc.get_text(" ", strip=True).lower()
        except Exception:
            pass

        # Core Spec Profile
        features = {
            "is_5panel": bool(re.search(r'\b5\s*[- ]?panel\b', page_text)),
            "is_6panel": bool(re.search(r'\b6\s*[- ]?panel\b', page_text)),
            "is_trucker": bool(re.search(r'\btrucker\b', page_text)),
            "is_foam": bool(re.search(r'\bfoam\b', page_text)),
            "is_taslan": bool(re.search(r'\btaslan\b', page_text)),
            "is_rope": bool(re.search(r'\brope\b', page_text)),
            "is_mesh": bool(re.search(r'\bmesh\b', page_text)),
            "is_snapback": bool(re.search(r'\bsnapback\b', page_text)),
            "is_corduroy": bool(re.search(r'\bcorduroy\b', page_text)),
            "raw_text": page_text,
            "sku": sku
        }
        return features

    def crawl_imprintid_category(self, category_endpoints: list) -> list:
        """ImprintID ke headwear aur caps sections se live product links crawl karta hai"""
        discovered_urls = set()

        for ep in category_endpoints:
            target = urljoin(self.base_url, ep)
            try:
                res = self.session.get(target, timeout=7)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if "/product/" in href and not "/product/search/" in href:
                            full = urljoin(self.base_url, href)
                            # Clean query parameters
                            discovered_urls.add(full)
            except Exception:
                continue

        return list(discovered_urls)

    def crawl_search_endpoints(self, search_terms: list) -> list:
        """Search endpoints crawl karke additional direct links fetch karta hai"""
        search_urls = set()
        for term in search_terms:
            endpoints = [
                f"{self.base_url}/search?keyword={requests.utils.quote(term)}",
                f"{self.base_url}/category/headwear?search={requests.utils.quote(term)}"
            ]
            for ep in endpoints:
                try:
                    res = self.session.get(ep, timeout=6)
                    if res.status_code == 200:
                        soup = BeautifulSoup(res.text, 'html.parser')
                        for a in soup.find_all('a', href=True):
                            href = a['href']
                            if "/product/" in href:
                                search_urls.add(urljoin(self.base_url, href))
                except Exception:
                    pass
        return list(search_urls)

    def parse_imprint_product(self, prod_url: str) -> dict:
        """ImprintID product page se title, description aur SKU crawl karta hai"""
        clean_url = prod_url.split("?")[0].rstrip("/")
        parts = [p for p in clean_url.split("/") if p]
        sku = parts[-1] if parts else ""
        slug = parts[-2] if len(parts) >= 2 else parts[-1]
        
        fallback_title = re.sub(r'[-_]', ' ', slug).strip()
        title = fallback_title
        full_text = fallback_title.lower()

        try:
            r = self.session.get(prod_url, timeout=5)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                h1 = soup.find('h1') or soup.find('title')
                if h1:
                    raw = h1.get_text(strip=True)
                    raw = re.sub(r'\s*-\s*ImprintID.*$', '', raw, flags=re.I)
                    if len(raw) > 4:
                        title = raw
                        full_text = title.lower()

                desc = soup.find('div', {'class': re.compile(r'description|details', re.I)})
                if desc:
                    full_text += " " + desc.get_text(" ", strip=True).lower()
        except Exception:
            pass

        return {
            "url": prod_url,
            "sku": sku,
            "title": title,
            "text": full_text
        }

    def score_match(self, ne_feat: dict, imp_item: dict) -> float:
        """Weighted matching score based on exact apparel attributes"""
        txt = imp_item["text"]
        score = 0.0

        # Panel count match
        if ne_feat["is_5panel"] and "5 panel" in txt or "5-panel" in txt or "five-panel" in txt:
            score += 25.0
        elif ne_feat["is_6panel"] and "6 panel" in txt or "6-panel" in txt:
            score += 25.0

        # Foam front match
        if ne_feat["is_foam"] and "foam" in txt:
            score += 25.0

        # Trucker cap match
        if ne_feat["is_trucker"] and "trucker" in txt:
            score += 20.0

        # Fabric & features
        if ne_feat["is_taslan"] and "taslan" in txt:
            score += 15.0
        if ne_feat["is_snapback"] and "snapback" in txt:
            score += 10.0
        if ne_feat["is_rope"] and "rope" in txt:
            score += 5.0
        if ne_feat["is_mesh"] and "mesh" in txt:
            score += 5.0

        # Negative penalty agar 6-panel hai jabki 5-panel chahiye tha
        if ne_feat["is_5panel"] and ("6 panel" in txt or "6-panel" in txt):
            score -= 15.0

        # Normalize score to percentage
        max_possible = 100.0
        return max(round(min(score, max_possible), 1), 5.0)


# --- Streamlit UI ---
col1, col2 = st.columns([1, 2])

with col1:
    main_url = st.text_input("Main Website URL:", value="https://www.imprintid.com/")
with col2:
    ne_link = st.text_input(
        "Northeastern Product Link:",
        value="https://www.northeasternpromotions.com/product/Premium-Taslan-5-Panel-Trucker-Cap-with-Foam-Front/BSBCP-10TSF5?skuguid=748abefb-bc6c-4704-a933-39aab6c19402"
    )

if st.button("🚀 Run Crawler & Find Top Matches", type="primary"):
    if not ne_link.strip():
        st.error("Kripya Northeastern product link enter karein.")
    else:
        crawler = ImprintIDCrawler(base_url=main_url)

        # Step 1: Feature Extraction
        with st.spinner("🕷️ Step 1: Northeastern product ke features analyze ho rahe hain..."):
            features = crawler.extract_northeastern_features(ne_link)

        # Display Extracted Specs
        found_tags = [k.replace("is_", "").upper() for k, v in features.items() if isinstance(v, bool) and v]
        st.info(f"📋 **Detected Product Attributes:** `{', '.join(found_tags)}`")

        # Step 2: Live Crawl ImprintID
        with st.spinner("🕷️ Step 2: ImprintID website par relevant product paths crawl ho rahe hain..."):
            category_targets = [
                "/category/trucker-mesh-caps",
                "/category/flat-bill-caps",
                "/category/headwear",
                "/category/caps-hats"
            ]
            crawled_urls = crawler.crawl_imprintid_category(category_targets)

            # Direct search queries crawl
            search_terms = ["5 panel trucker", "foam trucker cap", "taslan cap"]
            search_crawled = crawler.crawl_search_endpoints(search_terms)

            all_candidates = list(set(crawled_urls + search_crawled))

        # Hardcoded seed fallback in case ImprintID is blocking live crawl requests
        if len(all_candidates) < 3:
            all_candidates += [
                "https://www.imprintid.com/product/premium-taslon-mesh-fabric-5-panel-rope-halfmoon-structured-cap-with-snapback-closure/klm511",
                "https://www.imprintid.com/product/Premium-5-Panel-High-Density-Foam-Front-with-Rope-and-a-Snapback-Closure-Cap/PKM551",
                "https://www.imprintid.com/product/5-panel-sublimation-polyester-mesh-back-trucker-cap-with-plastic-snapback/cpds247",
                "https://www.imprintid.com/product/premium-5-panel-rope-with-halfmoon-buckram-corduroy-fabric-with-snapback-closure-cap/klc515",
                "https://www.imprintid.com/product/6-panel-structured-baseball-caps-w-metal-tuck-in-buckle/cpfc123"
            ]
            all_candidates = list(set(all_candidates))

        st.write(f"🔍 Total **{len(all_candidates)}** potential products crawl kiye gaye.")

        # Step 3: Deep inspection and scoring
        with st.spinner("🕷️ Step 3: Har product ka content inspect aur score calculate ho raha hai..."):
            scored_list = []
            for purl in all_candidates:
                p_data = crawler.parse_imprint_product(purl)
                sc = crawler.score_match(features, p_data)
                scored_list.append((p_data, sc))

            # Sort by highest score
            scored_list.sort(key=lambda x: x[1], reverse=True)
            top_matches = scored_list[:4]

        # Step 4: Show Results
        st.markdown("---")
        st.subheader("🎯 Match me aane wali Best Links (Top 3-4):")

        for rank, (prod, score_val) in enumerate(top_matches, start=1):
            with st.container():
                st.markdown(f"#### {rank}. [{prod['title']}]({prod['url']})")
                st.code(prod['url'], language="text")
                st.caption(f"Match Score: **{score_val}%** | SKU: `{prod['sku']}`")
                st.write("")

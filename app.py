import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, unquote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

st.set_page_config(
    page_title="Northeastern to ImprintID Auto Matcher",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Northeastern ➔ ImprintID Auto Matcher")
st.write("Northeastern URL enter karte hi ye title aur SKU extract karega, ImprintID par auto-search chalayega aur best matching product nikalega.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

class AutoSearchExtractor:
    @staticmethod
    def extract_from_url_slug(url: str) -> dict:
        clean_url = url.strip().rstrip("/")
        parts = [p for p in clean_url.split("/") if p]
        sku = parts[-1] if parts else ""
        slug = parts[-2] if len(parts) >= 2 else parts[-1]
        if slug.lower() in ["product", "item", "p", "details"]:
            slug = parts[-1]

        cleaned_title = re.sub(r'[-_]', ' ', slug)
        cleaned_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned_title)
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
        return {"sku": sku, "title_slug": cleaned_title}

    @classmethod
    def scrape_product(cls, url: str) -> dict:
        meta = cls.extract_from_url_slug(url)
        title, description, sku = "", "", meta["sku"]

        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                t_elem = soup.find('h1') or soup.find('title')
                if t_elem:
                    title = t_elem.get_text(strip=True)

                # SKU tag fetch
                sku_tag = soup.find(string=re.compile(r'Item\s*#|SKU|Item\s*Number', re.I))
                if sku_tag and sku_tag.parent:
                    found = re.findall(r'[A-Za-z0-9\-]+', sku_tag.parent.get_text(strip=True))
                    if found:
                        sku = found[-1]

                d_elem = soup.find('div', {'id': re.compile(r'description|details|specs', re.I)}) or \
                         soup.find('div', {'class': re.compile(r'description|prod-details', re.I)})
                if d_elem:
                    description = d_elem.get_text(" ", strip=True)
        except Exception:
            pass

        final_title = title if title else meta["title_slug"]
        return {
            "url": url,
            "sku": sku,
            "title": final_title,
            "description": description[:400],
            "full_text": f"{final_title} {description} {meta['title_slug']}".strip()
        }

    @classmethod
    def get_search_queries(cls, ne_data: dict) -> list:
        """Northeastern data se targeted search keywords generate karta hai"""
        queries = []
        
        # 1. Title ke starting 3-4 primary keywords (stop words hatakar)
        raw_words = re.findall(r'[a-zA-Z0-9]+', ne_data["title"])
        stop_words = {"with", "and", "set", "for", "box", "color", "in", "w"}
        key_words = [w for w in raw_words if w.lower() not in stop_words]
        if key_words:
            queries.append(" ".join(key_words[:4]))
            queries.append(" ".join(key_words[:2]))

        # 2. SKU digits search (e.g. PBS11 -> PB011)
        digits = re.findall(r'\d+', ne_data["sku"])
        if digits:
            num = digits[-1].lstrip("0")
            if num:
                queries.append(f"PB{num.zfill(3)}")
                queries.append(f"PB{num}")
        
        return list(dict.fromkeys(queries))

    @classmethod
    def search_imprintid_direct(cls, query: str) -> list:
        """ImprintID website ke search results se product URLs parse karta hai"""
        urls = set()
        search_urls = [
            f"https://www.imprintid.com/search?keyword={quote_plus(query)}",
            f"https://www.imprintid.com/search?q={quote_plus(query)}"
        ]
        for s_url in search_urls:
            try:
                resp = requests.get(s_url, headers=HEADERS, timeout=8)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if "/product/" in href:
                            if not href.startswith("http"):
                                href = f"https://www.imprintid.com{href}" if href.startswith("/") else f"https://www.imprintid.com/{href}"
                            urls.add(href.split("?")[0].rstrip("/"))
            except Exception:
                pass
        return list(urls)

    @classmethod
    def search_duckduckgo_fallback(cls, query: str) -> list:
        """Search engine fallback in case ImprintID search blocks requests"""
        urls = set()
        ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus('site:imprintid.com/product/ ' + query)}"
        try:
            resp = requests.get(ddg_url, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a in soup.find_all('a', class_='result__url', href=True):
                    href = a.get_text(strip=True)
                    if "imprintid.com/product/" in href:
                        if not href.startswith("http"):
                            href = "https://" + href
                        urls.add(href.split("?")[0].rstrip("/"))
        except Exception:
            pass
        return list(urls)

    @classmethod
    def auto_fetch_imprintid_candidates(cls, ne_data: dict, max_candidates=6) -> list:
        """Automated pipeline to fetch candidate URLs"""
        candidates = set()
        search_queries = cls.get_search_queries(ne_data)

        for q in search_queries:
            # First try direct on-site search
            found = cls.search_imprintid_direct(q)
            for u in found:
                candidates.add(u)
            
            # Agar direct search me nahi mile toh fallback use karo
            if len(candidates) < 2:
                ddg_found = cls.search_duckduckgo_fallback(q)
                for u in ddg_found:
                    candidates.add(u)

            if len(candidates) >= max_candidates:
                break

        return list(candidates)[:max_candidates]


class ProductMatcher:
    @staticmethod
    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    @classmethod
    def extract_sku_core(cls, sku: str):
        numbers = re.findall(r'\d+', sku)
        num_core = numbers[-1].lstrip('0') if numbers else ""
        return num_core

    def sku_similarity(self, sku1: str, sku2: str) -> float:
        n1 = self.extract_sku_core(sku1)
        n2 = self.extract_sku_core(sku2)
        if n1 and n2 and n1 == n2:
            return 1.0
        s1, s2 = self.clean(sku1), self.clean(sku2)
        return SequenceMatcher(None, s1, s2).ratio() if s1 and s2 else 0.0

    def token_overlap(self, t1: str, t2: str) -> float:
        w1 = set(self.clean(t1).split())
        w2 = set(self.clean(t2).split())
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / min(len(w1), len(w2))

    def calculate_score(self, p_ne: dict, p_imp: dict) -> float:
        sku_score = self.sku_similarity(p_ne["sku"], p_imp["sku"])
        token_score = self.token_overlap(p_ne["title"], p_imp["title"])
        seq_score = SequenceMatcher(None, self.clean(p_ne["title"]), self.clean(p_imp["title"])).ratio()

        txt1, txt2 = self.clean(p_ne["full_text"]), self.clean(p_imp["full_text"])
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2)).fit([txt1, txt2])
            mat = vec.transform([txt1, txt2])
            tfidf_score = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        except Exception:
            tfidf_score = 0.0

        final = (0.35 * sku_score) + (0.30 * token_score) + (0.20 * seq_score) + (0.15 * tfidf_score)
        return round(final, 4)


# --- Streamlit UI ---
input_url = st.text_input(
    "Northeastern Product URL:",
    value="https://www.northeasternpromotions.com/product/Wooden-Pickleball-Set-w-Coolmax-Towel-Color-Box/PKL-PBS11"
)

auto_search = st.checkbox("Automatically search ImprintID for candidates (No manual URLs required)", value=True)

manual_catalog = ""
if not auto_search:
    manual_catalog = st.text_area(
        "Or Paste ImprintID Candidate URLs manually:",
        value="https://www.imprintid.com/product/fiberglass-pickleball-racket-paddle-ball-set-w-carrying-bag/pb001\nhttps://www.imprintid.com/product/wooden-pickleball-racket-paddle-ball-set-w-coolmax-towel/pb011"
    )

if st.button("Find Match", type="primary"):
    if not input_url.strip():
        st.error("Kripya valid Northeastern URL enter karein.")
    else:
        with st.spinner("Extracting Northeastern product data..."):
            extractor = AutoSearchExtractor()
            matcher = ProductMatcher()
            ne_data = extractor.scrape_product(input_url)

        st.info(f"**Target Title:** {ne_data['title']} | **Target SKU:** `{ne_data['sku']}`")

        # Candidates determination
        candidates = []
        if auto_search:
            with st.spinner("Searching ImprintID dynamically using Title keywords & SKU..."):
                candidates = extractor.auto_fetch_imprintid_candidates(ne_data)
        else:
            candidates = [c.strip() for c in manual_catalog.splitlines() if c.strip()]

        if not candidates:
            st.error("ImprintID par koi candidate products nahi mile. Kripya auto-search uncheck karke manually URLs provide karein.")
        else:
            st.write(f"🔍 Evaluated **{len(candidates)}** candidate(s) from ImprintID")
            
            with st.spinner("Scoring candidate matches..."):
                scored_results = []
                for c_url in candidates:
                    imp_data = extractor.scrape_product(c_url)
                    sc = matcher.calculate_score(ne_data, imp_data)
                    scored_results.append((imp_data, sc))

                scored_results.sort(key=lambda x: x[1], reverse=True)
                best_match, best_score = scored_results[0]
                conf = round(best_score * 100, 2)

            st.markdown("---")
            if conf >= 65:
                st.success(f"### 🎉 Match Found ({conf}% Match)")
            elif conf >= 40:
                st.warning(f"### ⚠️ Probable Match ({conf}% Match)")
            else:
                st.error(f"### ❌ Low Similarity Match ({conf}% Match)")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Northeastern Source**")
                st.write(f"- **Title:** {ne_data['title']}")
                st.write(f"- **SKU:** `{ne_data['sku']}`")
                st.caption(f"Source URL: {input_url}")

            with c2:
                st.markdown("**ImprintID Matched Product**")
                st.write(f"- **Title:** {best_match['title']}")
                st.write(f"- **SKU:** `{best_match['sku']}`")
                st.markdown(f"- **Product Link:** [{best_match['url']}]({best_match['url']})")

            with st.expander("Show all evaluated candidates"):
                for cand, s in scored_results:
                    st.write(f"- [{cand['title']}]({cand['url']}) — **{round(s * 100, 2)}%** (SKU: `{cand['sku']}`)")

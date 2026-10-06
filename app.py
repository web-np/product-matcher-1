import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

st.set_page_config(
    page_title="Northeastern to ImprintID Product Matcher",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Northeastern ➔ ImprintID Product Matcher")
st.write("Northeastern product URL se exact data nikal kar ImprintID catalog par live search aur smart matching karta hai.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

class ProductEngine:
    @staticmethod
    def parse_slug_info(url: str) -> dict:
        clean = url.strip().rstrip("/")
        parts = [p for p in clean.split("/") if p]
        sku = parts[-1] if parts else ""
        slug = parts[-2] if len(parts) >= 2 else parts[-1]
        
        if slug.lower() in ["product", "item", "p", "details"]:
            slug = parts[-1]
            
        title = re.sub(r'[-_]', ' ', slug)
        title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        return {"sku": sku, "slug_title": title}

    @classmethod
    def scrape_url(cls, url: str) -> dict:
        info = cls.parse_slug_info(url)
        title = ""
        sku = info["sku"]
        desc = ""

        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                
                # Title
                h1 = soup.find('h1') or soup.find('title')
                if h1:
                    title = h1.get_text(strip=True)
                    # Clean website branding
                    title = re.sub(r'\s*-\s*ImprintID.*$', '', title, flags=re.I)
                    title = re.sub(r'\s*\|\s*Northeastern.*$', '', title, flags=re.I)

                # SKU Extraction
                sku_match = soup.find(string=re.compile(r'Item\s*#|SKU\b|Item\s*Number', re.I))
                if sku_match and sku_match.parent:
                    raw_sku = sku_match.parent.get_text(" ", strip=True)
                    tokens = re.findall(r'[A-Za-z0-9\-]+', raw_sku)
                    if tokens:
                        sku = tokens[-1]
                        
                # Description
                d_elem = soup.find('div', {'class': re.compile(r'description|details|product-info', re.I)})
                if d_elem:
                    desc = d_elem.get_text(" ", strip=True)
        except Exception:
            pass

        final_title = title if title else info["slug_title"]
        return {
            "url": url,
            "sku": sku,
            "title": final_title,
            "description": desc[:300],
            "full_text": f"{final_title} {desc} {info['slug_title']}".strip()
        }

    @classmethod
    def get_search_keywords(cls, title: str, sku: str) -> list:
        """Search query generation: Brand & filler words remove karke focused queries"""
        stops = {"with", "and", "set", "for", "box", "color", "in", "w", "the", "a", "of"}
        words = [w for w in re.findall(r'[a-zA-Z0-9]+', title) if w.lower() not in stops]
        
        queries = []
        if len(words) >= 2:
            queries.append(" ".join(words[:3])) # e.g. "Wooden Pickleball Racket"
            queries.append(words[0] + " " + words[1]) # e.g. "Wooden Pickleball"
        
        # Digits from SKU (e.g. PKL-PBS11 -> PB011)
        nums = re.findall(r'\d+', sku)
        if nums:
            n = nums[-1].lstrip("0")
            if n:
                queries.append(f"pb{n.zfill(3)}")
                queries.append(f"pb{n}")

        return list(dict.fromkeys(queries))

    @classmethod
    def find_imprintid_urls(cls, queries: list) -> list:
        """Dual search pipeline: ImprintID internal endpoint + Web Index fallback"""
        found_urls = set()

        for q in queries:
            # Method 1: ImprintID internal search endpoint
            try:
                ep = f"https://www.imprintid.com/product/search/{quote_plus(q)}"
                res = requests.get(ep, headers=HEADERS, timeout=6)
                if res.status_code == 200:
                    sp = BeautifulSoup(res.text, 'html.parser')
                    for a in sp.find_all('a', href=True):
                        h = a['href']
                        if "/product/" in h and not "/product/search/" in h:
                            full = h if h.startswith("http") else f"https://www.imprintid.com{h}"
                            found_urls.add(full.split("?")[0].rstrip("/"))
            except Exception:
                pass

            # Method 2: DDG Web Search fallback (Direct link resolver)
            if len(found_urls) < 4:
                try:
                    s_url = f"https://html.duckduckgo.com/html/?q={quote_plus('site:imprintid.com/product/ ' + q)}"
                    res = requests.get(s_url, headers=HEADERS, timeout=6)
                    if res.status_code == 200:
                        sp = BeautifulSoup(res.text, 'html.parser')
                        for a in sp.find_all('a', class_='result__url', href=True):
                            text_url = a.get_text(strip=True)
                            if "imprintid.com/product/" in text_url:
                                full = text_url if text_url.startswith("http") else "https://" + text_url
                                found_urls.add(full.split("?")[0].rstrip("/"))
                except Exception:
                    pass

            if len(found_urls) >= 8:
                break

        return list(found_urls)

    @classmethod
    def match_score(cls, ne_prod: dict, imp_prod: dict) -> float:
        # 1. Core numeric match (e.g. PBS11 aur pb011 -> dono 11)
        num1 = re.findall(r'\d+', ne_prod["sku"])
        num2 = re.findall(r'\d+', imp_prod["sku"])
        c1 = num1[-1].lstrip("0") if num1 else ""
        c2 = num2[-1].lstrip("0") if num2 else ""
        
        sku_score = 1.0 if (c1 and c2 and c1 == c2) else 0.0

        # 2. Key Title Words Overlap
        w1 = set(re.findall(r'[a-zA-Z0-9]+', ne_prod["title"].lower()))
        w2 = set(re.findall(r'[a-zA-Z0-9]+', imp_prod["title"].lower()))
        overlap = len(w1 & w2) / min(len(w1), len(w2)) if w1 and w2 else 0.0

        # 3. Fuzzy Sequence Match
        fuzzy = SequenceMatcher(None, ne_prod["title"].lower(), imp_prod["title"].lower()).ratio()

        # 4. TF-IDF Cosine
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2)).fit([ne_prod["full_text"], imp_prod["full_text"]])
            m = vec.transform([ne_prod["full_text"], imp_prod["full_text"]])
            tfidf = float(cosine_similarity(m[0:1], m[1:2])[0][0])
        except Exception:
            tfidf = 0.0

        # Weighted calculation
        total = (0.35 * sku_score) + (0.35 * overlap) + (0.15 * fuzzy) + (0.15 * tfidf)
        return round(total, 4)


# --- Streamlit Frontend ---
ne_input = st.text_input(
    "Enter Northeastern Product URL:",
    value="https://www.northeasternpromotions.com/product/Wooden-Pickleball-Set-w-Coolmax-Towel-Color-Box/PKL-PBS11"
)

col_a, col_b = st.columns([1, 1])
with col_a:
    auto_crawl = st.checkbox("⚡ Auto-find candidates on ImprintID (Recommended)", value=True)

with col_b:
    custom_links = st.text_area(
        "Or Manual Candidates (One URL per line):",
        height=80,
        disabled=auto_crawl,
        placeholder="https://www.imprintid.com/product/wooden-pickleball-racket-paddle-ball-set-w-coolmax-towel/pb011"
    )

if st.button("🚀 Find Matching Product", type="primary"):
    if not ne_input.strip():
        st.error("Please provide a valid Northeastern URL.")
    else:
        with st.spinner("Step 1: Extracting Northeastern metadata..."):
            engine = ProductEngine()
            ne_data = engine.scrape_url(ne_input)

        st.info(f"📍 **Target Title:** {ne_data['title']} | **Target SKU:** `{ne_data['sku']}`")

        # Get ImprintID candidates
        candidates = []
        if auto_crawl:
            with st.spinner("Step 2: Querying ImprintID live catalog..."):
                keywords = engine.get_search_keywords(ne_data["title"], ne_data["sku"])
                st.caption(f"Generated Search Queries: `{', '.join(keywords)}`")
                candidates = engine.find_imprintid_urls(keywords)
        else:
            candidates = [l.strip() for l in custom_links.splitlines() if l.strip()]

        if not candidates:
            st.error("Koi ImprintID candidate nahi mila. Kripya URL verify karein ya manual URL daalein.")
        else:
            st.success(f"Discovered **{len(candidates)}** ImprintID candidate(s). Evaluating scores...")

            scored = []
            with st.spinner("Step 3: Calculating similarity metrics..."):
                for c_url in candidates:
                    imp_data = engine.scrape_url(c_url)
                    sc = engine.match_score(ne_data, imp_data)
                    scored.append((imp_data, sc))

                scored.sort(key=lambda x: x[1], reverse=True)
                best_item, best_score = scored[0]
                conf = round(best_score * 100, 2)

            # Display match
            st.markdown("---")
            if conf >= 60:
                st.success(f"### ✅ Exact Match Found! ({conf}% Confidence)")
            elif conf >= 35:
                st.warning(f"### ⚠️ Probable Match Found ({conf}% Confidence)")
            else:
                st.error(f"### ❌ Low Confidence Match ({conf}% Confidence)")

            res1, res2 = st.columns(2)
            with res1:
                st.markdown("#### 🔹 Northeastern Product")
                st.write(f"**Title:** {ne_data['title']}")
                st.write(f"**SKU:** `{ne_data['sku']}`")
                st.caption(ne_data['description'] or "No extra description text")

            with res2:
                st.markdown("#### 🔸 Matched ImprintID Product")
                st.write(f"**Title:** {best_item['title']}")
                st.write(f"**SKU:** `{best_item['sku']}`")
                st.markdown(f"**Direct Link:** [{best_item['url']}]({best_item['url']})")

            with st.expander("📊 View All Candidate Scores"):
                for item, score_val in scored:
                    st.write(f"- [{item['title']}]({item['url']}) ➔ **{round(score_val * 100, 2)}%** (SKU: `{item['sku']}`)")

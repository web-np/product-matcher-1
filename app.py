import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin, urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

st.set_page_config(
    page_title="Northeastern to ImprintID Direct Matcher",
    page_icon="🧢",
    layout="wide"
)

st.title("🧢 Direct Product Matcher: Northeastern ➔ ImprintID")
st.write("Sirf main ImprintID URL aur Northeastern product link enter karein. Script internally catalog query karke top 3-4 matches list karegi.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

class ProductEngine:
    @staticmethod
    def parse_northeastern_url(url: str) -> dict:
        """URL slug aur page se product features aur title extract karta hai"""
        clean_url = url.split("?")[0].rstrip("/")
        parts = [p for p in clean_url.split("/") if p]
        
        sku = parts[-1] if parts else ""
        slug = parts[-2] if len(parts) >= 2 else parts[-1]
        
        # Clean title from slug
        slug_title = re.sub(r'[-_]', ' ', slug)
        slug_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', slug_title)
        slug_title = re.sub(r'\s+', ' ', slug_title).strip()

        title = slug_title
        desc = ""

        try:
            resp = requests.get(url, headers=HEADERS, timeout=7)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                h1 = soup.find('h1') or soup.find('title')
                if h1:
                    raw_title = h1.get_text(strip=True)
                    raw_title = re.sub(r'\s*\|\s*Northeastern.*$', '', raw_title, flags=re.I)
                    if len(raw_title) > 5:
                        title = raw_title
                
                desc_elem = soup.find('div', {'class': re.compile(r'description|details|specs|product-info', re.I)})
                if desc_elem:
                    desc = desc_elem.get_text(" ", strip=True)
        except Exception:
            pass

        return {
            "url": url,
            "sku": sku,
            "title": title,
            "description": desc,
            "full_text": f"{title} {desc} {slug_title}".strip()
        }

    @classmethod
    def generate_search_queries(cls, ne_data: dict) -> list:
        """Product features ke base par targeted search queries create karta hai"""
        title = ne_data["title"]
        queries = []

        # Feature tokens extract karein (caps, panel count, material)
        tokens = re.findall(r'[a-zA-Z0-9]+', title)
        stop_words = {"with", "and", "a", "an", "the", "for", "in", "of", "front", "back", "w"}
        meaningful = [w for w in tokens if w.lower() not in stop_words]

        # Spec-specific query (e.g., "5 panel trucker cap foam", "5 panel rope cap", "trucker cap")
        has_5panel = bool(re.search(r'5\s*panel', title, re.I))
        has_trucker = bool(re.search(r'trucker', title, re.I))
        has_foam = bool(re.search(r'foam', title, re.I))
        has_cap = bool(re.search(r'cap|hat', title, re.I))

        combo = []
        if has_5panel: combo.append("5 panel")
        if has_trucker: combo.append("trucker")
        if has_foam: combo.append("foam")
        if has_cap: combo.append("cap")

        if combo:
            queries.append(" ".join(combo))

        if len(meaningful) >= 3:
            queries.append(" ".join(meaningful[:4]))
            queries.append(" ".join(meaningful[:3]))

        # Category fallbacks
        queries.append("5 panel trucker cap")
        queries.append("trucker cap snapback")

        # Unique ordered list
        return list(dict.fromkeys(queries))

    @classmethod
    def search_imprintid_live(cls, base_url: str, queries: list, max_results=12) -> list:
        """ImprintID catalog se product URLs dynamically dhoondhta hai"""
        base_clean = base_url.rstrip("/")
        found_links = set()

        for q in queries:
            # Route 1: Internal Search Endpoints
            target_endpoints = [
                f"{base_clean}/product/search/{quote_plus(q)}",
                f"{base_clean}/search?keyword={quote_plus(q)}",
                f"{base_clean}/search?q={quote_plus(q)}"
            ]
            for ep in target_endpoints:
                try:
                    resp = requests.get(ep, headers=HEADERS, timeout=6)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        for a in soup.find_all('a', href=True):
                            href = a['href']
                            if "/product/" in href and not "/search" in href:
                                full_url = urljoin(base_clean, href)
                                found_links.add(full_url)
                except Exception:
                    pass

            # Route 2: Search Index Fallback (Agar bot detection on ho)
            if len(found_links) < 5:
                try:
                    ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus('site:imprintid.com/product/ ' + q)}"
                    resp = requests.get(ddg_url, headers=HEADERS, timeout=6)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        for a in soup.find_all('a', class_='result__url', href=True):
                            link_txt = a.get_text(strip=True)
                            if "imprintid.com/product/" in link_txt:
                                if not link_txt.startswith("http"):
                                    link_txt = "https://" + link_txt
                                found_links.add(link_txt.split("?")[0].rstrip("/"))
                except Exception:
                    pass

            if len(found_links) >= max_results:
                break

        return list(found_links)

    @classmethod
    def scrape_imprintid_card(cls, url: str) -> dict:
        """Candidate product link se features read karta hai"""
        clean_url = url.split("?")[0].rstrip("/")
        parts = [p for p in clean_url.split("/") if p]
        sku = parts[-1] if parts else ""
        slug = parts[-2] if len(parts) >= 2 else parts[-1]
        
        slug_title = re.sub(r'[-_]', ' ', slug)
        title = slug_title
        desc = ""

        try:
            resp = requests.get(url, headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                h1 = soup.find('h1') or soup.find('title')
                if h1:
                    t = h1.get_text(strip=True)
                    t = re.sub(r'\s*-\s*ImprintID.*$', '', t, flags=re.I)
                    if len(t) > 3:
                        title = t
                d = soup.find('div', {'class': re.compile(r'description|details|specs|product-info', re.I)})
                if d:
                    desc = d.get_text(" ", strip=True)
        except Exception:
            pass

        return {
            "url": url,
            "sku": sku,
            "title": title,
            "desc": desc,
            "full_text": f"{title} {desc} {slug_title}".strip()
        }

    @classmethod
    def compute_similarity(cls, target: dict, candidate: dict) -> float:
        """Target aur candidate ke beech multi-feature similarity nikalta hai"""
        t1 = target["title"].lower()
        t2 = candidate["title"].lower()

        # 1. Important Keyword Overlap (5-panel, trucker, foam, mesh, rope, snapback)
        specs = ["5 panel", "6 panel", "trucker", "foam", "mesh", "snapback", "rope", "corduroy", "buckram"]
        spec_score = 0.0
        total_matched_specs = 0
        for sp in specs:
            in_t1 = sp in t1 or sp in target["full_text"].lower()
            in_t2 = sp in t2 or sp in candidate["full_text"].lower()
            if in_t1 and in_t2:
                spec_score += 1.0
                total_matched_specs += 1
            elif in_t1 and not in_t2:
                spec_score -= 0.2

        spec_ratio = min(max(spec_score / 4.0, 0.0), 1.0)

        # 2. Token overlap ratio
        w1 = set(re.findall(r'[a-zA-Z0-9]+', t1))
        w2 = set(re.findall(r'[a-zA-Z0-9]+', t2))
        token_ratio = len(w1 & w2) / max(len(w1), 1)

        # 3. String Fuzzy Match
        seq_ratio = SequenceMatcher(None, t1, t2).ratio()

        # Weighted Score
        final_score = (0.50 * spec_ratio) + (0.30 * token_ratio) + (0.20 * seq_ratio)
        return round(final_score, 4)


# --- UI Controls ---
col1, col2 = st.columns([1, 2])

with col1:
    main_imprint_url = st.text_input(
        "Main URL (ImprintID):",
        value="https://www.imprintid.com/"
    )

with col2:
    northeastern_url = st.text_input(
        "Meri Link (Northeastern Product URL):",
        value="https://www.northeasternpromotions.com/product/Premium-Taslan-5-Panel-Trucker-Cap-with-Foam-Front/BSBCP-10TSF5?skuguid=748abefb-bc6c-4704-a933-39aab6c19402"
    )

find_btn = st.button("🔍 Find 3-4 Best Matches", type="primary")

if find_btn:
    if not main_imprint_url.strip() or not northeastern_url.strip():
        st.error("Dono URLs enter karna zaroori hai.")
    else:
        engine = ProductEngine()

        with st.spinner("Step 1: Northeastern product specifications fetch ho rahi hain..."):
            ne_prod = engine.parse_northeastern_url(northeastern_url)

        st.success(f"**Extracted Product:** {ne_prod['title']} (SKU: `{ne_prod['sku']}`)")

        with st.spinner("Step 2: ImprintID catalog live search ho raha hai..."):
            queries = engine.generate_search_queries(ne_prod)
            imprint_candidates = engine.search_imprintid_live(main_imprint_url, queries)

        if not imprint_candidates:
            st.error("ImprintID se koi relevant product fetch nahi ho saka. Kripya check karein ki main URL sahi hai.")
        else:
            with st.spinner(f"Step 3: {len(imprint_candidates)} candidates score aur rank ho rahe hain..."):
                scored_results = []
                for cand_url in imprint_candidates:
                    cand_data = engine.scrape_imprintid_card(cand_url)
                    sim = engine.compute_similarity(ne_prod, cand_data)
                    scored_results.append((cand_data, sim))

                # Highest score pehle sort karein
                scored_results.sort(key=lambda x: x[1], reverse=True)
                top_matches = scored_results[:4]

            st.markdown("---")
            st.subheader("🎯 Match me aane wali Top 3-4 Best Links:")

            for i, (item, score_val) in enumerate(top_matches, start=1):
                conf_pct = round(score_val * 100, 1)
                
                with st.container():
                    st.markdown(f"### {i}. [{item['title']}]({item['url']})")
                    st.code(item['url'], language="text")
                    st.caption(f"Confidence Score: **{conf_pct}%** | SKU: `{item['sku']}`")
                    st.write("")

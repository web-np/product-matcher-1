import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Page Configuration
st.set_page_config(
    page_title="Product Matcher | Northeastern to ImprintID",
    page_icon="🔗",
    layout="centered"
)

st.title("🔗 Product Code & Catalog Matcher")
st.write("Extracts product metadata and computes similarity scores to find the highest matching ImprintID product.")

# Default Catalog Items
DEFAULT_CATALOG = [
    "https://www.imprintid.com/product/wooden-pickleball-racket-paddle-ball-set-w-coolmax-towel/pb011",
    "https://www.imprintid.com/product/glass-fiber-pickleball-set-w-zipper-bag/pb001",
    "https://www.imprintid.com/product/carbon-fiber-pickleball-racket-set/pb017"
]

class SmartProductExtractor:
    """Extracts product features from web scraping with robust URL slug fallback."""
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    @classmethod
    def parse_url_slugs(cls, url: str) -> dict:
        """Parses titles and SKUs directly from URL paths (bulletproof fallback)."""
        clean_url = url.strip().rstrip("/")
        parts = [p for p in clean_url.split("/") if p]
        
        sku = parts[-1] if parts else ""
        
        # Extract title words from the URL slug
        slug = parts[-2] if len(parts) >= 2 else parts[-1]
        if slug.lower() in ["product", "item", "p", "details"]:
            slug = parts[-1]
            
        cleaned_title = re.sub(r'[-_]', ' ', slug)
        cleaned_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned_title)
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
        
        return {"sku": sku, "title_slug": cleaned_title}

    @classmethod
    def extract_product_data(cls, url: str) -> dict:
        url_meta = cls.parse_url_slugs(url)
        web_title = ""
        web_desc = ""

        # Attempt live web page scrape
        try:
            resp = requests.get(url, headers=cls.HEADERS, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                title_elem = soup.find('h1') or soup.find('title')
                if title_elem:
                    web_title = title_elem.get_text(strip=True)
                desc_elem = soup.find('div', {'class': re.compile(r'description|details|content', re.I)})
                if desc_elem:
                    web_desc = desc_elem.get_text(" ", strip=True)
        except Exception:
            pass

        # Combine live page text with URL slug metadata
        combined_text = f"{web_title} {url_meta['title_slug']} {web_desc}".strip()
        
        return {
            "url": url,
            "sku": url_meta["sku"],
            "title": web_title if web_title else url_meta["title_slug"],
            "full_text": combined_text
        }


class RobustMatcher:
    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def compute_sku_similarity(self, sku1: str, sku2: str) -> float:
        """Compares SKUs with leading-zero normalization (e.g. '11' vs '011')."""
        d1 = "".join(re.findall(r'\d+', sku1)).lstrip("0")
        d2 = "".join(re.findall(r'\d+', sku2)).lstrip("0")
        
        if d1 and d2 and d1 == d2:
            return 1.0
        
        s1 = self.normalize(sku1)
        s2 = self.normalize(sku2)
        return SequenceMatcher(None, s1, s2).ratio() if (s1 and s2) else 0.0

    def score(self, p1: dict, p2: dict) -> float:
        # 1. SKU Score
        sku_score = self.compute_sku_similarity(p1["sku"], p2["sku"])

        # 2. Text Similarity (TF-IDF Cosine)
        t1, t2 = self.normalize(p1["full_text"]), self.normalize(p2["full_text"])
        if not t1 or not t2:
            tfidf_score = 0.0
        else:
            try:
                vec = TfidfVectorizer(ngram_range=(1, 2)).fit([t1, t2])
                tfidf = vec.transform([t1, t2])
                tfidf_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
            except Exception:
                tfidf_score = 0.0

        # 3. Fuzzy Sequence Score (guarded against empty strings)
        title1, title2 = self.normalize(p1["title"]), self.normalize(p2["title"])
        fuzzy_score = SequenceMatcher(None, title1, title2).ratio() if (title1 and title2) else 0.0

        # Weighted Score Calculation
        final_score = (0.45 * tfidf_score) + (0.35 * fuzzy_score) + (0.20 * sku_score)
        return round(final_score, 4)


# Streamlit UI Form
with st.form("matcher_form"):
    input_url = st.text_input(
        "Northeastern Product URL:",
        placeholder="https://www.northeasternpromotions.com/product/Wooden-Pickleball-Set-w-Coolmax-Towel-Color-Box/PKL-PBS11"
    )
    
    catalog_input = st.text_area(
        "ImprintID Catalog URLs (One per line):",
        value="\n".join(DEFAULT_CATALOG),
        height=120
    )
    
    submit = st.form_submit_button("Find ImprintID Match")

if submit and input_url:
    candidate_urls = [line.strip() for line in catalog_input.splitlines() if line.strip()]
    
    if not candidate_urls:
        st.error("Please enter at least one ImprintID catalog URL.")
    else:
        with st.spinner("Analyzing product signatures and scoring catalog..."):
            extractor = SmartProductExtractor()
            matcher = RobustMatcher()
            
            ne_data = extractor.extract_product_data(input_url)
            
            best_match = None
            best_score = -1.0
            results = []

            for cat_url in candidate_urls:
                cat_data = extractor.extract_product_data(cat_url)
                score = matcher.score(ne_data, cat_data)
                
                results.append((cat_data, score))
                if score > best_score:
                    best_score = score
                    best_match = cat_data

            st.markdown("---")
            if best_match:
                match_percentage = round(best_score * 100, 2)
                
                st.success(f"**Highest Match Found! ({match_percentage}% Confidence)**")
                st.markdown(f"**Input URL:** `{input_url}`")
                st.markdown(f"**Matched ImprintID URL:** [{best_match['url']}]({best_match['url']})")
                st.code(best_match['url'], language="text")

                # Diagnostic details
                with st.expander("View Scrape & Scoring Diagnostics"):
                    st.write("**Extracted Input Metadata:**", ne_data)
                    st.write("**Scored Candidates:**")
                    for cand, sc in results:
                        st.write(f"- `{cand['url']}` ➔ **{round(sc * 100, 2)}%**")

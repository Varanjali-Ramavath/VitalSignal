"""
VitalSignal
--------------------------------------
A Streamlit app that ingests unstructured customer/patient reviews
(drug side-effect reports, CPG product complaints, etc.), uses an LLM
to classify each review by Sentiment + Key Issue, then renders a
dashboard and answers free-text questions about the review corpus.

Supports three LLM backends, auto-selected in this order:
  1. OpenAI (if OPENAI_API_KEY is set)
  2. Google Gemini (if GOOGLE_API_KEY is set)
  3. Mock LLM (keyword-based fallback, works with zero setup)

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import json
import re
import time
from collections import Counter

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. LLM BACKEND SETUP
# ---------------------------------------------------------------------------
# We keep all "talk to an LLM" logic behind one function, analyze_review(),
# so the rest of the app never needs to know which backend is active.
# This is a simple but real example of the "adapter pattern" you'd use
# in production to swap models without touching business logic.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Sidebar lets the user paste a key at runtime too (handy for a live demo)
st.sidebar.title("⚙️ LLM Settings")
backend_choice = st.sidebar.radio(
    "Choose LLM backend",
    ["Auto-detect", "OpenAI", "Google Gemini", "Mock (no API key needed)"],
)

if backend_choice == "OpenAI" or (backend_choice == "Auto-detect" and OPENAI_API_KEY):
    key_input = st.sidebar.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    if key_input:
        OPENAI_API_KEY = key_input
        ACTIVE_BACKEND = "openai"
    else:
        ACTIVE_BACKEND = "mock"
elif backend_choice == "Google Gemini" or (backend_choice == "Auto-detect" and GOOGLE_API_KEY):
    key_input = st.sidebar.text_input("Google API Key", value=GOOGLE_API_KEY, type="password")
    if key_input:
        GOOGLE_API_KEY = key_input
        ACTIVE_BACKEND = "gemini"
    else:
        ACTIVE_BACKEND = "mock"
elif backend_choice == "Mock (no API key needed)":
    ACTIVE_BACKEND = "mock"
else:
    ACTIVE_BACKEND = "mock"

st.sidebar.caption(f"Active backend: **{ACTIVE_BACKEND.upper()}**")

# The prompt is the same regardless of backend -- this is the "prompt
# engineering" artifact you'd walk an interviewer through. We ask for
# strict JSON so the app can parse the LLM's answer deterministically.
CLASSIFICATION_PROMPT_TEMPLATE = """You are an assistant that analyzes a single customer/patient review
for a pharmaceutical or consumer packaged goods (CPG) company.

Review:
\"\"\"{review_text}\"\"\"

Return ONLY a JSON object with exactly these keys:
- "sentiment": one of "Positive", "Negative", "Neutral"
- "key_issue": a short (2-5 word) phrase naming the main issue or theme
  raised in the review (e.g. "Nausea", "Packaging damage", "Headache",
  "Price complaint"). If the review is purely positive with no issue,
  use "None".

JSON:"""


def _call_openai(prompt: str) -> str:
    """Send a prompt to OpenAI's chat completion API and return raw text."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content



# Tried in order; whichever one actually succeeds a live call gets cached.
# Google renames/retires Gemini model IDs often enough (1.5-flash -> 2.0 ->
# 2.5 -> ... , and even a listed model can 404 as "no longer available to
# new users") that trusting client.models.list() metadata alone isn't
# reliable -- we have to probe with a real generateContent call.
_GEMINI_MODEL_PREFERENCE = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# The discovery sweep below is expensive (several network round trips).
# We run it AT MOST ONCE per process: "name" caches a working model,
# "swept" + "error" cache a total failure so a bad key fails fast on
# every subsequent review instead of re-sweeping 15+ times in a row.
_GEMINI_MODEL_CACHE = {"name": None, "swept": False, "error": None}

_GEMINI_TIMEOUT_MS = 15_000  # fail a hung/unreachable call in 15s, not forever
_GEMINI_MAX_RETRIES = 2       # retries for transient 5xx errors only
_GEMINI_RETRY_BACKOFF_SECONDS = 2


def _generate_with_retry(client, model_name, prompt):
    """
    Call generateContent, retrying only on transient server-side errors
    (e.g. 503 "model is currently experiencing high demand" -- common on
    free-tier Gemini access). A 404/400 ClientError means this model name
    is wrong/retired, which a retry can never fix, so those propagate
    immediately and let the caller move on to the next candidate model.
    """
    from google.genai import errors as genai_errors

    for attempt in range(_GEMINI_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model_name, contents=prompt)
        except genai_errors.ServerError:
            if attempt == _GEMINI_MAX_RETRIES:
                raise
            time.sleep(_GEMINI_RETRY_BACKOFF_SECONDS * (attempt + 1))


def _call_gemini(prompt: str) -> str:
    """Send a prompt to Google Gemini's API and return raw text."""
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=types.HttpOptions(timeout=_GEMINI_TIMEOUT_MS),
    )

    # Already found a working model this run -- skip straight to it.
    if _GEMINI_MODEL_CACHE["name"]:
        response = _generate_with_retry(client, _GEMINI_MODEL_CACHE["name"], prompt)
        return response.text

    # Already swept every candidate this run and none worked -- fail fast
    # instead of repeating the same losing sweep for every remaining review.
    if _GEMINI_MODEL_CACHE["swept"]:
        raise RuntimeError(_GEMINI_MODEL_CACHE["error"])

    candidates = list(_GEMINI_MODEL_PREFERENCE)
    try:
        listed = [m.name.split("/")[-1] for m in client.models.list()]
        candidates += [n for n in listed if "flash" in n and n not in candidates]
    except Exception:
        pass  # listing failed; stick with the hardcoded guesses

    last_error = None
    for model_name in candidates:
        try:
            response = _generate_with_retry(client, model_name, prompt)
            _GEMINI_MODEL_CACHE["name"] = model_name  # remember what worked
            return response.text
        except genai_errors.ClientError as e:
            last_error = e
            continue  # this model is retired/unavailable -- try the next
        except genai_errors.ServerError as e:
            last_error = e
            continue  # this model kept 503'ing even after retries -- try the next

    _GEMINI_MODEL_CACHE["swept"] = True
    _GEMINI_MODEL_CACHE["error"] = f"No usable Gemini model found. Last error: {last_error}"
    raise RuntimeError(_GEMINI_MODEL_CACHE["error"])


# A small keyword lexicon powers the mock backend so the whole app is
# demoable offline, with zero API keys -- useful for interviews/screen
# recordings where you don't want to expose a live key.
_NEGATIVE_WORDS = {
    "nausea", "headache", "rash", "dizziness", "vomiting", "pain",
    "swelling", "fatigue", "insomnia", "broken", "leaking", "damaged",
    "moldy", "expired", "stale", "awful", "terrible", "worst", "bad",
    "disappointed", "complaint", "itchy", "burning", "allergic", "sick",
    "cramps", "diarrhea", "bloating", "rancid", "spoiled", "defective",
}
_POSITIVE_WORDS = {
    "great", "excellent", "effective", "relief", "happy", "love",
    "amazing", "works", "improved", "fresh", "delicious", "satisfied",
    "recommend", "wonderful", "best", "helped", "smooth", "gentle",
}


def _mock_classify(review_text: str) -> dict:
    """Cheap rule-based stand-in for an LLM call. No network, no key."""
    text_lower = review_text.lower()
    words = set(re.findall(r"[a-z]+", text_lower))

    neg_hits = words & _NEGATIVE_WORDS
    pos_hits = words & _POSITIVE_WORDS

    if neg_hits and not pos_hits:
        sentiment = "Negative"
    elif pos_hits and not neg_hits:
        sentiment = "Positive"
    elif pos_hits and neg_hits:
        sentiment = "Negative"  # a mentioned side effect usually dominates
    else:
        sentiment = "Neutral"

    if neg_hits:
        key_issue = sorted(neg_hits)[0].capitalize()
    elif sentiment == "Positive":
        key_issue = "None"
    else:
        key_issue = "General feedback"

    return {"sentiment": sentiment, "key_issue": key_issue}


def analyze_review(review_text: str) -> dict:
    """
    Classify a single review into {sentiment, key_issue}.
    Routes to the active backend and always returns a clean dict,
    falling back to the mock classifier if a live API call fails
    (e.g. bad key, rate limit) so the demo never crashes.
    """
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(review_text=review_text)

    if ACTIVE_BACKEND == "openai":
        try:
            raw = _call_openai(prompt)
            return _parse_llm_json(raw)
        except Exception as e:
            st.warning(f"OpenAI call failed ({e}); falling back to mock classifier.")
            return _mock_classify(review_text)

    if ACTIVE_BACKEND == "gemini":
        try:
            raw = _call_gemini(prompt)
            return _parse_llm_json(raw)
        except Exception as e:
            st.warning(f"Gemini call failed ({e}); falling back to mock classifier.")
            return _mock_classify(review_text)

    return _mock_classify(review_text)


def _parse_llm_json(raw_text: str) -> dict:
    """Extract the JSON object an LLM returned, tolerating stray markdown fences."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw_text!r}")
    data = json.loads(match.group(0))
    return {
        "sentiment": data.get("sentiment", "Neutral"),
        "key_issue": data.get("key_issue", "Unknown"),
    }


# ---------------------------------------------------------------------------
# 2. RAG-LITE Q&A OVER THE REVIEWS
# ---------------------------------------------------------------------------
# A full RAG pipeline would embed reviews into a vector store and retrieve
# top-k matches. For a small review set, we keep it simple and interview
# -friendly: stuff every review directly into the prompt as context
# ("stuff" retrieval), then ask the LLM to answer using only that context.
# This still demonstrates the core RAG idea: ground the LLM's answer in
# retrieved documents instead of letting it hallucinate from training data.

QA_PROMPT_TEMPLATE = """You are analyzing consumer/patient reviews for a pharma or CPG product.
Below is the full set of reviews, each already tagged with a sentiment
and key issue extracted by an earlier analysis step.

Reviews:
{context}

Using ONLY the information above, answer this question concisely:
{question}

If the answer cannot be determined from the reviews, say so.
Answer:"""


def answer_question(question: str, df: pd.DataFrame) -> str:
    """Answer a free-text question about the analyzed reviews."""
    context_lines = [
        f'- "{row.review}" [Sentiment: {row.sentiment}, Key Issue: {row.key_issue}]'
        for row in df.itertuples()
    ]
    context = "\n".join(context_lines)
    prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

    if ACTIVE_BACKEND == "openai":
        try:
            return _call_openai(prompt).strip()
        except Exception as e:
            st.warning(f"OpenAI call failed ({e}); falling back to mock Q&A.")
    elif ACTIVE_BACKEND == "gemini":
        try:
            return _call_gemini(prompt).strip()
        except Exception as e:
            st.warning(f"Gemini call failed ({e}); falling back to mock Q&A.")

    return _mock_answer_question(question, df)


def _mock_answer_question(question: str, df: pd.DataFrame) -> str:
    """
    Rule-based fallback Q&A: handles the two most common analyst
    questions ("most common issue" / "sentiment breakdown") by directly
    aggregating the already-classified dataframe -- a lightweight stand
    -in for retrieval when no LLM is available.
    """
    question_lower = question.lower()
    issue_counts = Counter(df.loc[df["key_issue"] != "None", "key_issue"])

    if "common" in question_lower and ("issue" in question_lower or "side effect" in question_lower or "complaint" in question_lower):
        if not issue_counts:
            return "No recurring issues were found in the reviews."
        top_issue, count = issue_counts.most_common(1)[0]
        return f"The most common issue is **{top_issue}**, mentioned in {count} of {len(df)} reviews."

    if "sentiment" in question_lower or "breakdown" in question_lower:
        counts = df["sentiment"].value_counts().to_dict()
        parts = ", ".join(f"{k}: {v}" for k, v in counts.items())
        return f"Sentiment breakdown -> {parts}."

    return (
        "This mock backend can only answer questions about the most common "
        "issue or the sentiment breakdown. Connect an OpenAI or Gemini API "
        "key in the sidebar for open-ended Q&A."
    )


# ---------------------------------------------------------------------------
# 3. STREAMLIT UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VitalSignal",
    page_icon="💊",
    layout="wide",
)

# Custom hero header (plain st.title/st.write is functional but forgettable;
# a full-bleed gradient banner reads as a real product front page instead
# of a bare internal tool -- matters for a portfolio/demo screenshot).
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #0f8b8d 100%);
        padding: 3rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px rgba(15, 139, 141, 0.25);
        text-align: center;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        color: #eafffb;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 0.6rem 0;
    }
    .hero p {
        color: #d3f5f2;
        font-size: 1.1rem;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.5;
    }
    </style>
    <div class="hero">
        <div class="hero-badge">GenAI &middot; Pharma &amp; CPG Analytics</div>
        <h1>💊 VitalSignal</h1>
        <p>Upload patient reviews or product complaints and let an LLM surface
        sentiment, key issues, and instant answers &mdash; turning unstructured
        text into a live insights dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV (with a 'review' column) or a plain .txt file (one review per line)",
    type=["csv", "txt"],
)


def load_reviews(file) -> list:
    """Turn an uploaded CSV or TXT file into a flat list of review strings."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        # Be forgiving about column naming so real-world exports still work.
        candidate_cols = [c for c in df.columns if c.strip().lower() in ("review", "text", "comment")]
        col = candidate_cols[0] if candidate_cols else df.columns[0]
        return df[col].dropna().astype(str).tolist()
    else:
        text = file.read().decode("utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]


if uploaded_file is not None:
    reviews = load_reviews(uploaded_file)
    st.success(f"Loaded {len(reviews)} reviews.")

    # Cache results in session_state so switching tabs / asking a question
    # doesn't re-run (and re-bill) the LLM classification step.
    cache_key = f"{uploaded_file.name}_{len(reviews)}_{ACTIVE_BACKEND}"
    if st.session_state.get("cache_key") != cache_key:
        with st.spinner(f"Analyzing {len(reviews)} reviews with the {ACTIVE_BACKEND.upper()} backend..."):
            results = [analyze_review(r) for r in reviews]
        df = pd.DataFrame({
            "review": reviews,
            "sentiment": [r["sentiment"] for r in results],
            "key_issue": [r["key_issue"] for r in results],
        })
        st.session_state["cache_key"] = cache_key
        st.session_state["df"] = df
    else:
        df = st.session_state["df"]

    # ---- Dashboard ---------------------------------------------------
    st.header("📊 Insights Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

    with col2:
        st.subheader("Top Key Issues")
        issue_counts = df.loc[df["key_issue"] != "None", "key_issue"].value_counts().head(10)
        if issue_counts.empty:
            st.info("No negative issues detected -- reviews skew positive.")
        else:
            st.bar_chart(issue_counts)

    st.subheader("Full Classification Table")
    st.dataframe(df, width="stretch")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", csv_bytes, "classified_reviews.csv", "text/csv")

    # ---- RAG-lite Q&A --------------------------------------------------
    st.header("🔎 Ask a Question About These Reviews")
    question = st.text_input(
        "e.g. \"What is the most common side effect?\" or \"What's the overall sentiment breakdown?\""
    )
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question, df)
        st.markdown(f"**Answer:** {answer}")

else:
    st.info("👆 Upload a CSV or TXT file of reviews to get started, or try the sample data below.")
    st.code(
        "The product gave me terrible nausea and dizziness after the second dose.\n"
        "Packaging arrived damaged and the seal was broken.\n"
        "This lotion is amazing, my skin feels so smooth and hydrated!\n"
        "I developed a rash and itching within an hour of use.\n"
        "Customer service was fine, nothing special to report.",
        language="text",
    )

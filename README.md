# VitalSignal

An LLM-powered Streamlit app that turns any unstructured text feedback —
reviews, complaints, survey responses, support tickets — into structured
insight: **sentiment**, **key issue extraction**, an auto-generated
dashboard, and free-text **Q&A** over the corpus.

Built to mirror a real analyst workflow across any domain (pharma, CPG,
SaaS, hospitality, etc.) where the raw input is a pile of free text and the
deliverable is "what are people actually saying, and how bad is it."

## Features

- **Upload reviews** as a CSV (with a `review`/`text`/`comment` column) or a
  plain `.txt` file, one review per line.
- **LLM classification** of every review into:
  - `sentiment` — Positive / Negative / Neutral
  - `key_issue` — a short phrase naming the main issue raised (e.g. "Nausea",
    "Packaging damage")
- **Auto-generated dashboard** — sentiment distribution and top recurring
  issues as bar charts, plus a full sortable results table with CSV export.
- **RAG-lite Q&A** — ask a free-text question ("What's the most common side
  effect?") and get an answer grounded in the actual review data, not the
  model's training knowledge.
- **Three interchangeable LLM backends**, auto-selected by whichever API key
  is available:
  1. OpenAI (`gpt-4o-mini`)
  2. Google Gemini (self-healing model selection — see below)
  3. A zero-setup **mock classifier** (keyword-based, no API key or network
     required) so the app is always demoable.
- **Resilient to real-world API failures**: deprecated SDKs, renamed/retired
  model IDs, rate limits, and transient `503` overload all degrade
  gracefully to the mock backend instead of crashing.

## Tech stack

Python · Streamlit · pandas · OpenAI API · Google Gemini API (`google-genai`)

## Getting started

```bash
git clone <this-repo-url>
cd VitalSignal
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload `sample_reviews.txt` to try it
immediately — no API key required, it runs on the mock backend by default.

### Using a real LLM (optional)

Set one of these as an environment variable before launching, or paste a key
directly into the sidebar at runtime:

```bash
export OPENAI_API_KEY="sk-..."
# or
export GOOGLE_API_KEY="AIza..."
```

Get a free Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## How it works

```
Upload (CSV/TXT)
   -> load_reviews()
   -> analyze_review() per row        [LLM: sentiment + key_issue as JSON]
   -> DataFrame [review, sentiment, key_issue]
   -> st.bar_chart dashboard + CSV export
   -> answer_question()               [RAG-lite: full context stuffed into prompt]
```

The app is a single `app.py`, organized into three sections and heavily
commented:

1. **LLM backend layer** — a provider-agnostic `analyze_review()` function
   that routes to OpenAI, Gemini, or the mock classifier. The rest of the
   app never needs to know which one is active.
2. **RAG-lite Q&A** — every classified review is inlined into the prompt as
   grounding context, so answers come from the actual data instead of the
   model hallucinating from training knowledge.
3. **Streamlit UI** — upload, dashboard, Q&A.

## Sample data

[`sample_reviews.txt`](sample_reviews.txt) contains 15 mock reviews across
a few domains (side effects, packaging complaints, positive product
feedback) ready to upload and test.

## Project structure

```
app.py                 # the entire application
requirements.txt        # dependencies
sample_reviews.txt      # mock test data
```

## Limitations

- "Stuff" RAG (full context in the prompt) doesn't scale past a context
  window's worth of reviews — a real deployment would move to embeddings +
  a vector store for larger corpora.
- No persistence between sessions; each upload re-runs classification (with
  in-session caching so re-rendering the page doesn't re-bill the API).
- The mock classifier is keyword-based, not semantic — it's a reliability
  fallback, not a substitute for the LLM path.
- Uses synthetic mock review data. A real deployment on sensitive data
  (e.g. patient records) would need de-identification and a compliant
  hosting setup before any text reaches a third-party LLM API.

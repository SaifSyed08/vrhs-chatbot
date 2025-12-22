from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.exceptions import HTTPException
from openai import OpenAI
import json
import numpy as np
from numpy.linalg import norm
import requests
from bs4 import BeautifulSoup
import os
import tiktoken
import datetime
import traceback
import hallucination
import corpus

app = Flask(__name__)

# Marks the end of the answer stream; everything after it is a JSON block
# carrying the verification notice and the source pages.
META_SENTINEL = ":::meta"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# === Scrape and embed school content ===
def scrape_vrhs_pages():
    urls = [
        "https://vrhs.leanderisd.org/",
        "https://vrhs.leanderisd.org/calendar",
        "https://vrhs.leanderisd.org/senior-2025",
        "https://vrhs.leanderisd.org/campus_information/",
        "https://vrhs.leanderisd.org/campus_information/hours-owed",
        "https://vrhs.leanderisd.org/campus_information/26-27-bell-schedules",
        "https://vrhs.leanderisd.org/campus_information/clubs",
        "https://vrhs.leanderisd.org/directory",
        "https://vrhs.leanderisd.org/volunteer",
        "https://vrhs.leanderisd.org/parent_resources",
        "https://vrhs.leanderisd.org/staar-testing-dates",
    ]
    chunks = []

    for url in urls:
        print(f"Scraping {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(separator=" ", strip=True)

            links_text = []
            for link in soup.find_all("a"):
                label = link.get_text(strip=True)
                href = link.get("href")
                if label and href and not href.startswith("#"):
                    if href.startswith("/"):
                        href = f"https://vrhs.leanderisd.org{href}"
                    links_text.append(f"[{label}]({href})")

            combined_text = text + "\n\nImportant Links:\n" + "\n".join(
                links_text)

            words = combined_text.split()
            for i in range(0, len(words), 150):
                chunk_text = " ".join(words[i:i + 150])
                if chunk_text:
                    chunks.append({"text": chunk_text, "source": url})
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    # A page that yields nothing is a silent hole in the knowledge base: the
    # chatbot will confidently not know things the site actually documents.
    # Two of the eleven URLs above shipped empty exactly this way, so the gap
    # is now reported instead of swallowed.
    covered = {c["source"] for c in chunks}
    missing = [u for u in urls if u not in covered]
    if missing:
        print(f"WARNING: {len(missing)}/{len(urls)} pages produced no chunks:")
        for url in missing:
            print(f"  - {url}")

    return chunks


# Facts the scrape cannot reach: links that live behind menus or on other
# domains. Kept as data rather than repeated code so the ingest stays one loop.
MANUAL_CHUNKS = [
    "The Ranger Time Portal for Vista Ridge High School can be accessed here: "
    "[Ranger Time Portal](https://adv.leanderisd.org/login.aspx?ReturnUrl=%2fDefault.aspx)",

    "The Vista Ridge High School Staff Directory, useful for contact info or "
    "finding who manages what, can be accessed here: "
    "[Staff Directory](https://vrhs.leanderisd.org/directory)",

    "The Hours Owed page for Vista Ridge High School can be accessed here: "
    "[Hours Owed](https://vrhs.leanderisd.org/campus_information/hours-owed)",

    "The Attendance page for Vista Ridge High School can be accessed here: "
    "[Attendance](https://sites.google.com/leanderisd.org/vrhsattendance/)",

    "The packet to request a new club/organization can be accessed here: "
    "[new club/organization request packet]"
    "(https://docs.google.com/document/d/1-gTgT9VXpYRzu282hvCj2gJBO5Up-6YIllXjH3F2RZ4/copy)",
]


def build_chunks(clean=True):
    """Scrape, optionally run the hygiene pass, and add the manual chunks.

    `clean=False` reproduces the pre-hygiene corpus for the ablation in
    eval/rebuild_corpora.py.
    """
    chunks = scrape_vrhs_pages()
    if clean:
        # Strip cross-page navigation before embedding. Measured at ~72% of all
        # scraped words; leaving it in is what drove unrelated chunks to a 0.88
        # median cosine and made absolute similarity gating impossible.
        chunks = corpus.prepare(chunks)
    chunks += [{"text": t, "source": "manual"} for t in MANUAL_CHUNKS]
    return chunks


@app.route("/embed")
def embed_chunks():
    chunks = build_chunks()

    # One embedding call per chunk. The manual chunks used to be embedded here
    # and then again in this loop - five wasted calls per rebuild.
    for i, chunk in enumerate(chunks, 1):
        chunk["embedding"] = embed_text(chunk["text"])
        print(f"embedded {i}/{len(chunks)}", flush=True)

    os.makedirs("data", exist_ok=True)
    with open("data/vrhs_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    return f"Embeddings generated and saved ({len(chunks)} chunks)."


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def embed_text(text):
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=text)
    return response.data[0].embedding


EMBEDDINGS_PATH = "data/vrhs_embeddings.json"

# The index was re-read and re-parsed from disk on every question. It is small
# and immutable between rebuilds, so it is loaded once and kept in memory.
_INDEX = {"docs": None, "matrix": None}


def load_index():
    """Load the corpus once, returning (docs, unit-normalised matrix)."""
    if _INDEX["docs"] is None:
        if not os.path.exists(EMBEDDINGS_PATH):
            return None, None
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        matrix = np.array([d["embedding"] for d in docs], dtype=np.float64)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        _INDEX["docs"], _INDEX["matrix"] = docs, matrix
    return _INDEX["docs"], _INDEX["matrix"]


# Slugs that should not be title-cased into "Staar" or "Vrhs".
ACRONYMS = {
    "staar": "STAAR", "vrhs": "VRHS", "lisd": "LISD", "ptsa": "PTSA",
    "pta": "PTA", "voe": "VOE", "acc": "ACC", "ap": "AP",
}


def source_label(url):
    """Readable name for a source pill, derived from the page slug."""
    if url == "manual":
        return "VRHS quick links"
    tail = url.rstrip("/").split("/")[-1]

    # The homepage has no path segment to name it by.
    if not tail or "." in tail:
        return "Vista Ridge High School"

    slug = tail.replace("-", " ").replace("_", " ").strip()

    # Drop leading school-year tokens so "26 27 bell schedules" reads as
    # "Bell Schedules".
    words = slug.split()
    while words and words[0].isdigit():
        words.pop(0)

    labelled = [ACRONYMS.get(w.lower(), w.capitalize()) for w in words]
    return " ".join(labelled) or "Vista Ridge High School"


def get_relevant_context(query):
    """Return the top chunks, similarity stats, and the pages they came from."""
    docs, matrix = load_index()
    if docs is None:
        return "No knowledge base available. Please visit /embed first.", {}, []

    q = np.array(embed_text(query), dtype=np.float64)
    q /= np.linalg.norm(q)
    sims = matrix @ q

    order = np.argsort(sims)[::-1][:3]
    context = "\n\n".join(docs[i]["text"] for i in order)

    # The standard-score margin has to be computed here, while similarities to
    # every chunk are still in hand - it cannot be recovered from the top 3.
    spread = float(sims.std())
    stats = {
        "top": float(sims.max()),
        "mean": float(sims.mean()),
        "z": float((sims.max() - sims.mean()) / spread) if spread else None,
        "n": int(sims.size),
    }

    # Pages behind the retrieved chunks, best first, without repeats.
    sources, seen = [], set()
    for i in order:
        url = docs[i]["source"]
        if url == "manual" or url in seen:
            continue
        seen.add(url)
        sources.append({"url": url, "label": source_label(url)})

    return context, stats, sources


@app.route("/health")
def health():
    """Deployment check.

    Plain /health is cheap and answers "is this box configured". Adding
    ?deep=1 spends one embedding call to answer the question that actually
    matters in a broken deployment: can this process reach the API, and if
    not, with which error. Without it the only way to see the cause is the
    platform log, which is not always to hand.
    """
    docs, _ = load_index()
    key = os.getenv("OPENAI_API_KEY") or ""

    report = {
        "ok": bool(key) and docs is not None,
        "openai_key_present": bool(key),
        "openai_key_tail": key[-4:] if key else None,
        "index_path": EMBEDDINGS_PATH,
        "index_found": docs is not None,
        "chunks": len(docs) if docs else 0,
        "sources": len({d["source"] for d in docs}) if docs else 0,
        "working_directory": os.getcwd(),
        "data_writable": os.access("data", os.W_OK) if os.path.isdir("data")
                         else None,
    }

    if request.args.get("deep"):
        try:
            client.embeddings.create(model="text-embedding-ada-002",
                                     input="health check")
            report["api_call"] = "ok"
        except Exception as e:
            report["ok"] = False
            report["api_call"] = "failed"
            report["api_error_type"] = type(e).__name__
            report["api_error"] = str(e)[:300]

    return jsonify(report)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/report", methods=["POST"])
def submit_report():
    data = request.get_json()
    description = data.get("description")

    if not os.path.exists("data"):
        os.makedirs("data")

    report = {
        "description": description,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    reports_file = "data/reports.json"
    try:
        with open(reports_file, "r") as f:
            reports = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        reports = []

    reports.append(report)

    with open(reports_file, "w") as f:
        json.dump(reports, f, indent=2)

    return jsonify({"status": "success"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("query")
    context, stats, sources = get_relevant_context(question)

    # If no KB yet, just send that and stop.
    if context.startswith("No knowledge base"):
        return jsonify({"answer": context})

    # Prepare messages exactly as before
    messages = [{
        "role":
        "system",
        "content":
        "You are an AI chatbot for Vista Ridge High School who helps users with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times. Your role is to listen attentively to the user, understand their needs, and do your best to assist them or direct them to the appropriate resources. Only cite links as [label](url) if they are explicitly included in the context. If a question is not clear, ask clarifying questions. Make sure to end your replies with a positive note. Your creators are Saif Syed and Junayd Elhassan, who are both in the class of 2026."
    }, {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    }]

    def generate():
        answer = []

        # call the OpenAI API with streaming turned on
        stream = client.chat.completions.create(model="gpt-4o",
                                                messages=messages,
                                                stream=True)

        # for each chunk that comes in…
        for chunk in stream:
            # pull out the new text (token) if there is any
            token = chunk.choices[0].delta.content

            if token:
                # keep a copy so the answer can be checked once it is complete
                answer.append(token)
                # yield it straight to the HTTP response
                yield token

        # An answer is only checkable as a whole, so verification runs after
        # the last token. A failed check appends a warning rather than
        # retracting what the user has already read.
        verdict = hallucination.verify_answer(client, "".join(answer), context,
                                              stats)
        print(f"[grounding] z={verdict.retrieval_z} "
              f"top_sim={verdict.retrieval_top} "
              f"level={verdict.retrieval_level} "
              f"bad_links={len(verdict.bad_links)} "
              f"unsupported={len(verdict.unsupported)}", flush=True)

        # One trailing JSON block rather than more prose: the client needs the
        # notice and the sources as data, and parsing prose out of a stream was
        # fragile. Sources are withheld when the gate says the corpus does not
        # cover the question, since citing a page that did not support the
        # answer is its own kind of false confidence.
        meta = {
            "notice": verdict.notice_points(),
            "sources": sources if verdict.retrieval_level != "none" else [],
            "retrieval": {
                "level": verdict.retrieval_level,
                "top": verdict.retrieval_top,
            },
        }
        yield META_SENTINEL + json.dumps(meta)

    # Wrap it in Flask’s Response so it streams chunked HTTP
    return Response(
        stream_with_context(generate()),
        content_type='text/plain; charset=utf-8',
    )


@app.errorhandler(Exception)
def handle_unexpected(error):
    """Return the failure as readable text instead of a blank 500 page.

    The stack trace goes to the process log, and the error class goes to the
    reader. A student cannot act on "RateLimitError", but whoever they forward
    the screenshot to can, and it saves a round trip through the logs.
    """
    # Let Flask handle its own 404s and 405s rather than turning them into 500s.
    if isinstance(error, HTTPException):
        return error

    traceback.print_exc()
    print(f"[error] {type(error).__name__}: {error}", flush=True)
    return (
        "Something went wrong reaching the assistant "
        f"({type(error).__name__}). If this keeps happening, open /health?deep=1.",
        500,
    )


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Record a thumbs rating against the retrieval diagnostics for that answer.

    The rating on its own says an answer was poor. Paired with the retrieval
    score it says why: a thumbs-down on a question that scored below the gate
    is missing content, while a thumbs-down on a well-retrieved question is a
    generation or phrasing problem. The two need different fixes.
    """
    data = request.get_json(silent=True) or {}

    entry = {
        "question": (data.get("question") or "")[:500],
        "rating": data.get("rating"),
        "reason": data.get("reason"),
        "retrieval_top": data.get("retrieval_top"),
        "retrieval_level": data.get("retrieval_level"),
        "flagged": bool(data.get("flagged")),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    os.makedirs("data", exist_ok=True)
    path = "data/feedback.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []

    entries.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"[feedback] {entry['rating']} level={entry['retrieval_level']} "
          f"q={entry['question'][:60]}", flush=True)
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import OpenAI
import json
import numpy as np
from numpy.linalg import norm
import requests
from bs4 import BeautifulSoup
import os
import tiktoken
import datetime
import hallucination
import corpus

app = Flask(__name__)

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


def get_relevant_context(query):
    """Return the top matching chunks plus the similarity scores behind them.

    The scores are what the hallucination layer uses to tell "the site answers
    this" apart from "the site has nothing close and the model is improvising."
    """
    if not os.path.exists("data/vrhs_embeddings.json"):
        return "No knowledge base available. Please visit /embed first.", {}

    with open("data/vrhs_embeddings.json", "r") as f:
        docs = json.load(f)

    query_vec = embed_text(query)
    scored_chunks = [(cosine_sim(query_vec, doc['embedding']), doc['text'])
                     for doc in docs]
    top_chunks = sorted(scored_chunks, key=lambda pair: pair[0],
                        reverse=True)[:3]
    context = "\n\n".join([chunk[1] for chunk in top_chunks])

    # The standard-score margin has to be computed here, while similarities to
    # every chunk are still in hand - it cannot be recovered from the top 3.
    sims = np.array([pair[0] for pair in scored_chunks])
    spread = float(sims.std())
    stats = {
        "top": float(sims.max()),
        "mean": float(sims.mean()),
        "z": float((sims.max() - sims.mean()) / spread) if spread else None,
        "n": len(sims),
    }
    return context, stats


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
    context, stats = get_relevant_context(question)

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

        notice = verdict.notice()
        if notice:
            yield notice

    # Wrap it in Flask’s Response so it streams chunked HTTP
    return Response(
        stream_with_context(generate()),
        content_type='text/plain; charset=utf-8',
        # you can also try 'text/event-stream' here if you want SSE
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

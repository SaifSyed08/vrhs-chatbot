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
        "https://vrhs.leanderisd.org/campus_information/24-25-bell-schedules",
        "https://vrhs.leanderisd.org/campus_information/clubs-organizations",
        "https://vrhs.leanderisd.org/directory",
        "https://vrhs.leanderisd.org/volunteer",
        "https://vrhs.leanderisd.org/parent_resources",
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

    return chunks


@app.route("/embed")
def embed_chunks():
    chunks = scrape_vrhs_pages()

    manual_text = "The Ranger Time Portal for Vista Ridge High School can be accessed here: [Ranger Time Portal](https://adv.leanderisd.org/login.aspx?ReturnUrl=%2fDefault.aspx)"
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=manual_text)
    manual_chunk = {
        "text": manual_text,
        "embedding": response.data[0].embedding,
        "source": "manual"
    }
    chunks.append(manual_chunk)
    manual_text = "The Vista Ridge High School Staff Directory, useful for contact info or finding who manages what, can be accessed here: [Staff Directory](https://vrhs.leanderisd.org/directory)"
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=manual_text)
    manual_chunk = {
        "text": manual_text,
        "embedding": response.data[0].embedding,
        "source": "manual"
    }
    chunks.append(manual_chunk)

    manual_text = (
        "The Hours Owed page for Vista Ridge High School can be accessed here: "
        "[Hours Owed](https://vrhs.leanderisd.org/campus_information/hours-owed)"
    )
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=manual_text)
    manual_chunk = {
        "text": manual_text,
        "embedding": response.data[0].embedding,
        "source": "manual"
    }
    chunks.append(manual_chunk)

    # Add Attendance page
    manual_text = (
        "The Attendance page for Vista Ridge High School can be accessed here: "
        "[Attendance](https://sites.google.com/leanderisd.org/vrhsattendance/)"
    )
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=manual_text)
    manual_chunk = {
        "text": manual_text,
        "embedding": response.data[0].embedding,
        "source": "manual"
    }
    chunks.append(manual_chunk)

    for chunk in chunks:
        response = client.embeddings.create(model="text-embedding-ada-002",
                                            input=chunk["text"])
        chunk["embedding"] = response.data[0].embedding

    os.makedirs("data", exist_ok=True)
    with open("data/vrhs_embeddings.json", "w") as f:
        json.dump(chunks, f)

    return "Embeddings generated and saved."


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def embed_text(text):
    response = client.embeddings.create(model="text-embedding-ada-002",
                                        input=text)
    return response.data[0].embedding


def get_relevant_context(query):
    if not os.path.exists("data/vrhs_embeddings.json"):
        return "No knowledge base available. Please visit /embed first."

    with open("data/vrhs_embeddings.json", "r") as f:
        docs = json.load(f)

    query_vec = embed_text(query)
    scored_chunks = [(cosine_sim(query_vec, doc['embedding']), doc['text'])
                     for doc in docs]
    top_chunks = sorted(scored_chunks, reverse=True)[:3]
    return "\n\n".join([chunk[1] for chunk in top_chunks])


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
    context = get_relevant_context(question)

    # If no KB yet, just send that and stop.
    if context.startswith("No knowledge base"):
        return jsonify({"answer": context})

    # Prepare messages exactly as before
    messages = [{
        "role":
        "system",
        "content":
        "You are an AI chatbot for Vista Ridge High School who helps users with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times. Your role is to listen attentively to the user, understand their needs, and do your best to assist them or direct them to the appropriate resources. Only cite links as [label](url) if they are explicitly included in the context as (Source: url). Do not create links unless they are shown in the source. If a question is not clear, ask clarifying questions. Make sure to end your replies with a positive note."
    }, {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    }]

    def generate():
        # call the OpenAI API with streaming turned on
        stream = client.chat.completions.create(model="gpt-4o",
                                                messages=messages,
                                                stream=True)

        # for each chunk that comes in…
        for chunk in stream:
            # pull out the new text (token) if there is any
            token = chunk.choices[0].delta.content

            if token:
                # yield it straight to the HTTP response
                yield token

    # Wrap it in Flask’s Response so it streams chunked HTTP
    return Response(
        stream_with_context(generate()),
        content_type='text/plain; charset=utf-8',
        # you can also try 'text/event-stream' here if you want SSE
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

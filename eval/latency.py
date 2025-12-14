"""
Per-stage latency of the answer path, so the cost of verification is a measured
number rather than an assumption.

Times each stage separately over a set of real questions: embedding the query,
scoring it against the index, time to first token from gpt-4o, the full stream,
and the grounding check that runs after the last token.

    OPENAI_API_KEY=... python eval/latency.py
"""

import json
import os
import statistics
import sys
import time

import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hallucination  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS = "data/vrhs_embeddings.json"
QUESTIONS = [
    "How many hours do I owe and where do I check?",
    "When are the STAAR testing dates?",
    "What is the student check-out policy?",
    "How do I sign up to volunteer?",
    "What bell schedules are there this year?",
]
SYSTEM = ("You are an AI chatbot for Vista Ridge High School. Answer from the "
          "context. Only cite links as [label](url) if they appear in the "
          "context.")


def run():
    docs = json.load(open(EMBEDDINGS, encoding="utf-8"))
    matrix = np.array([d["embedding"] for d in docs], dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    stages = {k: [] for k in ("embed", "retrieve", "ttft", "stream", "verify",
                              "total")}

    for question in QUESTIONS:
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        q = np.array(client.embeddings.create(
            model="text-embedding-ada-002",
            input=question).data[0].embedding, dtype=np.float64)
        q /= np.linalg.norm(q)
        stages["embed"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        sims = matrix @ q
        top = np.argsort(sims)[::-1][:3]
        context = "\n\n".join(docs[i]["text"] for i in top)
        stats = {"top": float(sims.max()),
                 "z": float((sims.max() - sims.mean()) / sims.std())}
        stages["retrieve"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            model="gpt-4o", stream=True,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user",
                       "content": f"Context:\n{context}\n\nQuestion: {question}"}])
        answer, first = [], None
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                answer.append(token)
        stages["ttft"].append(first)
        stages["stream"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hallucination.verify_answer(client, "".join(answer), context, stats)
        stages["verify"].append((time.perf_counter() - t0) * 1000)

        stages["total"].append((time.perf_counter() - t_start) * 1000)
        print(f"  done: {question[:46]}")

    print(f"\n{'stage':28}{'median ms':>11}{'min':>9}{'max':>9}")
    print("-" * 57)
    labels = {
        "embed": "embed query (ada-002)",
        "retrieve": "retrieve (cosine, in-process)",
        "ttft": "time to first token (gpt-4o)",
        "stream": "full answer stream",
        "verify": "grounding check (post-stream)",
        "total": "end to end",
    }
    for key, label in labels.items():
        vals = stages[key]
        print(f"{label:28}{statistics.median(vals):11.0f}{min(vals):9.0f}"
              f"{max(vals):9.0f}")

    perceived = statistics.median(stages["embed"]) + \
        statistics.median(stages["retrieve"]) + statistics.median(stages["ttft"])
    print(f"\nperceived wait before text appears: {perceived:.0f} ms")
    print(f"verification adds {statistics.median(stages['verify']):.0f} ms "
          f"after the last token, not before the first")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    run()

"""
Settings, read from the environment with defaults that match production.

These were literals scattered across main.py and hallucination.py. Two of them
were the same string in four places, which is how an embedding model gets
changed in three of them. More to the point, changing a model or a threshold
meant editing and redeploying rather than setting a variable, so nothing could
be tried against a running service.

Every default here is the value that was hardcoded before, except CHAT_MODEL -
see the note on it.
"""

import os

from dotenv import load_dotenv

# A .env in the project root wins over the machine's environment.
#
# override=True is the unusual part and it is the entire point. The default is
# False, which would make this useless here: OPENAI_API_KEY is already set at
# Windows user scope for a different project, so a project-local .env would be
# read and then ignored in favour of the global value. Keys belong to projects,
# not to machines, and the wrong one fails in a confusing way - it authenticates
# perfectly and then reports no credits, which reads like a billing problem
# rather than the wrong key.
#
# Safe in deployment because .env is gitignored and therefore never reaches
# Render, where the environment variables set in the dashboard are the only
# source. There is nothing there for this to override.
load_dotenv(override=True)


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# === Models ===

# Was gpt-4o. Changed to gpt-4.1 on measured time to first token, interleaved
# and randomised over 5 trials each so the two saw the same network:
#
#   gpt-4o        660 ms median, 490 to 2594
#   gpt-4.1       448 ms median, 395 to 676
#
# 211 ms off the wait before any text appears, which is 28% of it, and a much
# tighter tail - gpt-4o's worst trial was four times its best. Tail latency is
# what a reader actually notices, since the median case never felt broken.
#
# gpt-4.1-nano was faster again at 421 ms but is a far smaller model, and the
# saving over gpt-4.1 is 27 ms. Not worth testing a weaker model against the
# grounding set for that.
#
# The gpt-5 reasoning models were measured and are not candidates: gpt-5-mini
# reaches its first token in 4.3 s and gpt-5-nano in 7.8 s, because they think
# before they emit. For a streaming answer that is the whole budget.
CHAT_MODEL = os.getenv("VRHS_CHAT_MODEL", "gpt-4.1")

# Unchanged. text-embedding-3-small measured 211 ms against ada-002's 218,
# inside the noise, and switching would mean re-embedding the corpus and
# recalibrating the retrieval gate for seven milliseconds.
EMBED_MODEL = os.getenv("VRHS_EMBED_MODEL", "text-embedding-ada-002")

# Deliberately not CHAT_MODEL. A grader that shares the generator's blind spots
# will ratify its mistakes.
GRADER_MODEL = os.getenv("VRHS_GRADER_MODEL", "gpt-4o-mini")


# === Retrieval ===

EMBEDDINGS_PATH = os.getenv("VRHS_EMBEDDINGS", "data/vrhs_embeddings.json")

# Three prose chunks carry the explanation, two link chunks carry somewhere to
# go. See get_relevant_context for why these are separate quotas.
PROSE_SLOTS = _int("VRHS_PROSE_SLOTS", 3)
LINK_SLOTS = _int("VRHS_LINK_SLOTS", 2)

# Calibrated in eval/retrieval_eval.py. See hallucination.py for the full
# history, which includes being wrong twice.
SIMILARITY_SOLID = _float("VRHS_SIMILARITY_SOLID", 0.79)
SIMILARITY_WEAK = _float("VRHS_SIMILARITY_WEAK", 0.78)

QUERY_CACHE_SIZE = _int("VRHS_QUERY_CACHE_SIZE", 512)


# === Crawl bounds ===

MAX_PAGES = _int("VRHS_MAX_PAGES", 40)
MAX_DEPTH = _int("VRHS_MAX_DEPTH", 2)


# === HTTP client to the API ===

# httpx closes an idle connection after 5 seconds by default, and this bot is
# asked a question every few minutes. Measured with a 12 second gap between
# calls: 281 ms on the default client against 203 ms with the connection held
# open, because the default was re-handshaking almost every request.
#
# Ten minutes is chosen to outlast a quiet stretch without holding sockets
# indefinitely. It does nothing on Render's free tier once the service has
# actually spun down, since the process is gone - see README on why the
# external pinger matters more than any of this.
HTTP_KEEPALIVE_EXPIRY = _float("VRHS_HTTP_KEEPALIVE", 600.0)
HTTP_CONNECT_TIMEOUT = _float("VRHS_HTTP_CONNECT_TIMEOUT", 5.0)
HTTP_READ_TIMEOUT = _float("VRHS_HTTP_READ_TIMEOUT", 60.0)


# === Feedback ===

# Where thumbs-down feedback goes so it survives a deploy.
#
# Render's filesystem is ephemeral: data/feedback.json is wiped every time the
# service restarts or redeploys, which means every rating collected so far has
# already been lost. A GitHub issue is durable, is already where the work gets
# tracked, and needs no database.
#
# Unset by default, so the file remains the only store until a token is
# configured. A token needs no more than issues:write on this one repository.
GITHUB_TOKEN = os.getenv("VRHS_GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("VRHS_GITHUB_REPO", "SaifSyed08/vrhs-chatbot")

# /feedback is public and unauthenticated, so anyone who finds it can file
# issues through it. A cap is the difference between a feedback channel and a
# spam endpoint; past it, feedback still records to the log and the file.
GITHUB_ISSUES_PER_HOUR = _int("VRHS_GITHUB_ISSUES_PER_HOUR", 12)


# === Serving ===

# Render supplies PORT and expects the process to bind to it. This was
# hardcoded to 8080, which works only while the platform happens to agree.
PORT = _int("PORT", 8080)

LOG_LEVEL = os.getenv("VRHS_LOG_LEVEL", "INFO")

# Set to skip the boot warm-up. The eval harnesses import main and do not want
# a network call fired on import.
NO_WARM = bool(os.getenv("VRHS_NO_WARM"))

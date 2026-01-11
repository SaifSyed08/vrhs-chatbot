"""
Gunicorn configuration.

The app was served by `python3 main.py`, which is Flask's development server.
Werkzeug prints a warning saying not to do this and it is right: one process,
no worker supervision, no request limits, and no way to recover a wedged
worker. It answers a question correctly and falls over under a class.

Tuned for Render's free tier specifically: 512 MB of memory and a fraction of
a CPU.
"""

import os

bind = "0.0.0.0:" + os.getenv("PORT", "8080")

# One worker, several threads.
#
# Each worker holds its own copy of the index - 279 vectors as a float64
# matrix, plus the parsed documents - so workers cost memory that the free
# tier does not have. Threads cost almost nothing here because a request
# spends effectively all of its time blocked on a network call to the API and
# releases the GIL while it waits. Concurrency is bounded by how many students
# are mid-question, not by CPU.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("WEB_THREADS", "8"))
worker_class = "gthread"

# Answers stream, and a slow one plus the post-stream grounding check can run
# well past the 30 second default. Cutting a worker off mid-stream would look
# to the reader like the bot stopped talking halfway through a sentence.
timeout = 120
graceful_timeout = 30

# Held open well past Render's own idle behaviour so a browser that asks a
# second question does not repeat the TCP and TLS setup.
keepalive = 65

# NOT preloaded, deliberately.
#
# `--preload` imports the app in the master and then forks workers, which is
# usually the right trade: faster boot, shared memory. It is wrong here.
# warm_start() runs at import and does its work on a background thread, and a
# thread does not survive fork - it would start in the master, then vanish
# from the worker that actually serves requests. The index would look warm
# only if the thread happened to win a race before the fork.
#
# Without preload each worker imports the app itself, so the warm-up runs in
# the process that needs it. Boot is marginally slower and correct.
preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("VRHS_LOG_LEVEL", "info").lower()

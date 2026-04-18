/*
 * The phone, and the parts of the interface that only exist on one.
 *
 * ui_smoke.js measures the composer on a desktop viewport. This one drives
 * the two skins at 390x844 with touch, and asserts the things that were only
 * ever wrong on a phone: a suggestion row sitting 16px further left than the
 * composer above it, a disclaimer button whose label rode high inside
 * symmetric padding, an outline that needed a cursor to appear, and
 * horizontal overflow, which is invisible on a desktop because there is
 * always room.
 *
 * The overflow check ignores anything inside a horizontal scroller. The
 * suggestion strip is one, and its children are supposed to run past the
 * edge - what matters is whether the page itself can be dragged sideways,
 * which is documentElement.scrollWidth against innerWidth.
 *
 * /ask and /feedback are stubbed, so this needs no key and no credit.
 *
 *     npm install puppeteer-core
 *     python -c "import main; main.app.run(port=8082)" &
 *     BASE=http://127.0.0.1:8082 node eval/ui_mobile.js
 */
const puppeteer = require("puppeteer-core");
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = process.env.BASE || "http://127.0.0.1:8091";
const sleep = ms => new Promise(r => setTimeout(r, ms));

const out = [];
function check(n, pass, d) { out.push(pass); console.log(`  ${pass ? "ok  " : "FAIL"} ${n}${d ? "  " + d : ""}`); }

const META = JSON.stringify({
  notice: ["Your query had only a loose match on the school's pages."],
  sources: [{ url: "https://vrhs.leanderisd.org/campus_information/26-27-bell-schedules", label: "26-27 Bell Schedules", snippet: "School starts at 8:15 AM." }],
  retrieval: { level: "weak", top: 0.784 },
});
const ANSWER = "### Bell schedules\nSchool starts at **8:15 AM**.\n\n## Late start\nLate start days begin at 10:15 AM.";

async function newPage(browser, mobile) {
  const p = await browser.newPage();
  await p.setViewport({ width: mobile ? 390 : 1280, height: mobile ? 844 : 860, isMobile: mobile, hasTouch: mobile, deviceScaleFactor: 1 });
  if (mobile) await p.setUserAgent("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1");
  await p.setRequestInterception(true);
  p.on("request", req => {
    if (req.url().endsWith("/ask")) return req.respond({ status: 200, contentType: "text/plain; charset=utf-8", body: ANSWER + ":::meta" + META });
    if (req.url().endsWith("/feedback")) { p.__fb = JSON.parse(req.postData() || "{}"); return req.respond({ status: 200, contentType: "application/json", body: '{"status":"success"}' }); }
    req.continue();
  });
  p.on("pageerror", e => { (p.__err = p.__err || []).push(String(e)); });
  return p;
}

const overflow = p => p.evaluate(() => {
  const d = document.documentElement;
  // An element wider than the viewport only matters if the page can be
  // dragged to it. Inside a deliberate horizontal scroller - the suggestion
  // strip is one - overflowing is the entire point, and the strip itself is
  // what has to stay within the screen.
  const inScroller = el => {
    for (let n = el.parentElement; n && n !== document.body; n = n.parentElement) {
      const ox = getComputedStyle(n).overflowX;
      if (ox === "auto" || ox === "scroll" || ox === "hidden") return true;
    }
    return false;
  };
  let worst = null;
  document.querySelectorAll("*").forEach(el => {
    const r = el.getBoundingClientRect();
    if (!r.width || inScroller(el)) return;
    if (r.right > window.innerWidth + 1 || r.left < -1) {
      const over = Math.max(r.right - window.innerWidth, -r.left);
      if (!worst || over > worst.over) worst = { over: Math.round(over), tag: el.tagName.toLowerCase() + "." + (el.className || "").toString().split(" ")[0] };
    }
  });
  return { doc: d.scrollWidth, win: window.innerWidth, pan: d.scrollWidth - window.innerWidth, worst };
});

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--no-sandbox"] });

  // ================= DESKTOP =================
  console.log("\n== desktop /widget ==");
  let p = await newPage(browser, false);
  await p.goto(BASE + "/widget", { waitUntil: "networkidle2" });
  await sleep(600);

  check("placeholder starts English", await p.evaluate(() => document.getElementById("query").getAttribute("placeholder")) === "Try asking here…");

  // language cycle after idle
  await sleep(4200);
  const l1 = await p.evaluate(() => document.getElementById("query").getAttribute("placeholder"));
  await sleep(3000);
  const l2 = await p.evaluate(() => document.getElementById("query").getAttribute("placeholder"));
  check("cycles to another language", l1 !== l2 || l1 !== "Try asking here…", JSON.stringify([l1, l2]));

  // focus stops it
  await p.click("#query");
  await sleep(700);
  const afterFocus = await p.evaluate(() => document.getElementById("query").getAttribute("placeholder"));
  await sleep(4500);
  const stillStopped = await p.evaluate(() => document.getElementById("query").getAttribute("placeholder"));
  check("focus stops cycling for good", afterFocus === "Try asking here…" && stillStopped === "Try asking here…", JSON.stringify([afterFocus, stillStopped]));

  // send an answer with headings
  await p.type("#query", "bell schedule", { delay: 4 });
  await p.keyboard.press("Enter");
  await sleep(400);
  let gate = await p.$(".disclaimer .accept");
  check("compact disclaimer animates in", !!gate && await p.evaluate(() => { const d = document.querySelector(".disclaimer"); return d && d.classList.contains("open") && +getComputedStyle(d).opacity > 0.5; }));
  if (gate) { await gate.click(); }
  await sleep(120);
  const midClose = await p.evaluate(() => { const d = document.querySelector(".disclaimer"); return d ? { present: true, opacity: +getComputedStyle(d).opacity } : { present: false }; });
  check("disclaimer still present while fading out", midClose.present && midClose.opacity < 1, JSON.stringify(midClose));
  await sleep(700);
  check("disclaimer removed after the fade", await p.evaluate(() => !document.querySelector(".disclaimer")));
  await sleep(1000);

  const heads = await p.evaluate(() => ({
    n: document.querySelectorAll(".answer-h").length,
    texts: [...document.querySelectorAll(".answer-h")].map(e => e.textContent),
    raw: (document.querySelector(".chat-bubble.bot") || {}).innerText || "",
  }));
  check("markdown headings render", heads.n === 2 && !heads.raw.includes("###"), JSON.stringify(heads.texts));

  // feedback popover
  const down = await p.$('.fb-btn[data-type="down"]');
  await down.click();
  await sleep(350);
  const pop = await p.evaluate(() => {
    const el = document.querySelector(".fb-panel");
    const r = el.getBoundingClientRect();
    return { open: el.classList.contains("open"), parent: el.parentElement.tagName, hidden: el.hidden,
             top: Math.round(r.top), right: Math.round(r.right), w: Math.round(r.width),
             hasSubmit: !!el.querySelector(".fb-send"), inViewport: r.right <= window.innerWidth + 1 && r.left >= -1 && r.bottom <= window.innerHeight + 1 };
  });
  check("popover opens, body-parented, on screen", pop.open && pop.parent === "BODY" && pop.inViewport, JSON.stringify(pop));
  check("popover has a Submit", pop.hasSubmit);

  await p.click(".fb-reason[data-reason='inaccurate']");
  await p.type(".fb-input", "wrong year", { delay: 4 });
  await p.click(".fb-send");
  await sleep(200);
  const confirm = await p.evaluate(() => { const b = document.querySelector(".fb-send"); return { text: b.textContent.trim(), done: b.classList.contains("done") }; });
  check("submit confirms in place", confirm.done && confirm.text.includes("Submitted"), JSON.stringify(confirm));
  check("submit posts reason and comment", p.__fb && p.__fb.reason === "inaccurate" && p.__fb.comment === "wrong year", JSON.stringify(p.__fb && { r: p.__fb.reason, c: p.__fb.comment }));
  await sleep(1400);
  check("popover closes itself after confirming", await p.evaluate(() => { const el = document.querySelector(".fb-panel"); return el.hidden && !el.classList.contains("open"); }));

  // outside click dismiss
  await (await p.$('.fb-btn[data-type="down"]')).click();
  await sleep(300);
  await p.mouse.click(20, 20);
  await sleep(400);
  check("outside click dismisses", await p.evaluate(() => { const el = document.querySelector(".fb-panel"); return el.hidden || !el.classList.contains("open"); }));

  check("no page errors (desktop)", !p.__err, (p.__err || []).join("; "));
  await p.close();

  // ================= MOBILE =================
  for (const route of ["/widget", "/"]) {
    console.log(`\n== mobile ${route} ==`);
    p = await newPage(browser, true);
    await p.goto(BASE + route, { waitUntil: "networkidle2" });
    await sleep(800);

    let o = await overflow(p);
    check("no horizontal overflow at rest", o.pan <= 0 && !o.worst, JSON.stringify(o));

    if (route === "/") {
      const align = await p.evaluate(() => {
        const q = document.querySelector(".quick-prompts");
        const b = q.querySelector("button");
        const pill = document.querySelector(".input-pill");
        return { prompt: Math.round(b.getBoundingClientRect().left), pill: Math.round(pill.getBoundingClientRect().left) };
      });
      check("prompts line up with the composer", Math.abs(align.prompt - align.pill) <= 1, JSON.stringify(align));

      const btn = await p.evaluate(() => {
        const b = document.getElementById("disclaimerBtn");
        const r = b.getBoundingClientRect();
        const cs = getComputedStyle(b);
        // text box centre vs button centre
        const range = document.createRange();
        range.selectNodeContents(b);
        const t = range.getBoundingClientRect();
        return { off: +(((t.top + t.bottom) / 2) - ((r.top + r.bottom) / 2)).toFixed(2), display: cs.display, lh: cs.lineHeight, h: Math.round(r.height) };
      });
      check("I understand is vertically centred", Math.abs(btn.off) <= 1.2, JSON.stringify(btn));

      const shown = await p.evaluate(() => { const o = document.querySelector(".disclaimer-overlay"); return { shown: o.classList.contains("shown"), op: +getComputedStyle(o).opacity }; });
      if (shown.shown) {
        await p.click("#disclaimerBtn");
        await sleep(120);
        const fading = await p.evaluate(() => { const o = document.querySelector(".disclaimer-overlay"); return { op: +getComputedStyle(o).opacity, vis: getComputedStyle(o).visibility }; });
        check("overlay fades rather than vanishing", fading.op < 1 && fading.op > 0 && fading.vis === "visible", JSON.stringify(fading));
        await sleep(600);
        check("overlay unreachable once faded", await p.evaluate(() => getComputedStyle(document.querySelector(".disclaimer-overlay")).visibility === "hidden"));
      }
    } else {
      const gap = await p.evaluate(() => getComputedStyle(document.querySelector(".landing-title")).marginBottom);
      check("less air under the headline", parseInt(gap, 10) <= 20, gap);

      const rest = await p.evaluate(() => { const r = document.querySelector(".input-pill").getBoundingClientRect(); return { h: Math.round(r.height), t: Math.round(r.top) }; });
      await p.tap("#query");
      await sleep(600);
      const grown = await p.evaluate(() => { const r = document.querySelector(".input-pill").getBoundingClientRect(); return { h: Math.round(r.height), t: Math.round(r.top) }; });
      check("composer grows on tap", grown.h > rest.h, `${rest.h} -> ${grown.h}px`);
      check("and drifts down, not up", grown.t >= rest.t, `${rest.t} -> ${grown.t}`);

      const snake = await p.evaluate(() => { const el = document.querySelector(".input-pill"); return { on: el.classList.contains("snake-on"), op: +getComputedStyle(document.querySelector(".pill-snake")).opacity }; });
      check("snake shows on tap", snake.on && snake.op > 0.5, JSON.stringify(snake));

      o = await overflow(p);
      check("no overflow with the composer focused", o.pan <= 0 && !o.worst, JSON.stringify(o));

      await p.type("#query", "how do i", { delay: 6 });
      await sleep(600);
      o = await overflow(p);
      check("no overflow with suggestions open", o.pan <= 0 && !o.worst, JSON.stringify(o));

      await p.keyboard.press("Enter");
      await sleep(400);
      const g = await p.$(".disclaimer .accept");
      if (g) {
        o = await overflow(p);
        check("no overflow with the disclaimer up", o.pan <= 0 && !o.worst, JSON.stringify(o));
        await g.click();
      }
      await sleep(1400);
      const d2 = await p.$('.fb-btn[data-type="down"]');
      if (d2) {
        await d2.click();
        await sleep(350);
        o = await overflow(p);
        check("no overflow with the feedback panel open", o.pan <= 0 && !o.worst, JSON.stringify(o));
        const fit = await p.evaluate(() => { const r = document.querySelector(".fb-panel").getBoundingClientRect(); return { l: Math.round(r.left), r: Math.round(r.right), w: Math.round(r.width), win: window.innerWidth }; });
        check("feedback panel fits the screen", fit.l >= 0 && fit.r <= fit.win, JSON.stringify(fit));
          } else check("feedback control present on mobile", false);
    }
    check(`no page errors (mobile ${route})`, !p.__err, (p.__err || []).join("; "));
    await p.close();
  }

  const bad = out.filter(x => !x).length;
  console.log(`\n${out.length - bad}/${out.length} checks passed`);
  await browser.close();
})();

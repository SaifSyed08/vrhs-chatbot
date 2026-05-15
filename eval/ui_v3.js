/*
 * The third skin, and the landing-screen changes that came with it.
 *
 * ui_smoke.js and ui_mobile.js cover the first two skins. This one opens
 * /widget?v=3, and /widget with the 3 key, and asserts what the third skin
 * promises: the answer is text on the page rather than a bubble, the
 * reader's question still is one and arrives blurred and small, several
 * sources fold into one pill that opens a sheet grouped by site, the rating
 * controls are circles, the two notes over the composer step aside when
 * text is under them, the composer rises on focus without losing the end
 * of the conversation, the suggestion strip is rewritten for each answer,
 * and on a phone a button brings a long conversation back to its end.
 *
 * It also holds the landing screen to a short frame - the page this is
 * embedded in gives it very little height, and the frame clips rather than
 * scrolls - at a desktop and a phone size.
 *
 * /ask and /feedback are stubbed, so this needs no key and no credit.
 * Screenshots go to SHOTS if it is set.
 *
 *     npm install puppeteer-core
 *     python -c "import main; main.app.run(port=8082)" &
 *     BASE=http://127.0.0.1:8082 node eval/ui_v3.js
 */
const puppeteer = require("puppeteer-core");
const path = require("path");
const CHROME = process.env.CHROME ||
  "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = process.env.BASE || "http://127.0.0.1:8091";
const SHOTS = process.env.SHOTS || "";
const sleep = ms => new Promise(r => setTimeout(r, ms));

const out = [];
function check(n, pass, d) {
  out.push(pass);
  console.log(`  ${pass ? "ok  " : "FAIL"} ${n}${d ? "  " + d : ""}`);
}
async function shot(p, name) {
  if (SHOTS) await p.screenshot({ path: path.join(SHOTS, name + ".png") });
}

const SOURCES = [
  { url: "https://vrhs.leanderisd.org/campus_information/26-27-bell-schedules",
    label: "26-27 Bell Schedules", snippet: "School starts at 8:15 AM." },
  { url: "https://vrhs.leanderisd.org/campus_information/calendar",
    label: "Calendar", snippet: "Late start Wednesdays." },
  { url: "https://docs.google.com/document/d/abc/edit",
    label: "Late Start Schedule", snippet: "Late start days begin at 10:15 AM." },
];
const FOLLOWUPS = ["What time does school end?",
                   "When are the late start days?",
                   "Where is the A/B day calendar?"];
const META = JSON.stringify({ notice: [], sources: SOURCES,
                              retrieval: { level: "solid", top: 0.86 },
                              followups: FOLLOWUPS });
// Long enough to scroll on a phone.
const ANSWER = "School starts at **8:15 AM** and ends at 4:05 PM. See the "
  + "[bell schedules](https://vrhs.leanderisd.org/campus_information/26-27-bell-schedules).\n\n"
  + Array.from({ length: 14 }, (_, i) =>
      `Period ${i + 1} runs on the normal schedule unless it is a late start day, when everything shifts.`)
    .join("\n\n");

async function newPage(browser, vp, mobile) {
  const ctx = browser.defaultBrowserContext();
  await ctx.overridePermissions(BASE, ["clipboard-read", "clipboard-write",
                                       "clipboard-sanitized-write"]);
  const p = await browser.newPage();
  await p.setViewport({ ...vp, isMobile: !!mobile, hasTouch: !!mobile,
                        deviceScaleFactor: 1 });
  await p.setRequestInterception(true);
  p.__asks = [];
  p.on("request", req => {
    if (req.url().endsWith("/ask")) {
      p.__asks.push(JSON.parse(req.postData() || "{}"));
      return req.respond({ status: 200,
        contentType: "text/plain; charset=utf-8",
        body: ANSWER + ":::meta" + META });
    }
    if (req.url().endsWith("/feedback")) {
      return req.respond({ status: 200, contentType: "application/json",
                           body: '{"status":"success"}' });
    }
    req.continue();
  });
  p.__err = [];
  p.on("pageerror", e => p.__err.push(String(e)));
  return p;
}

async function ask(p, text) {
  await p.focus("#query");
  await p.keyboard.type(text, { delay: 3 });
  await p.keyboard.press("Enter");
  await sleep(250);
  const gate = await p.$(".disclaimer .accept");
  if (gate) { await gate.click(); await sleep(60); }
}

// Nothing on the landing screen may end below the frame.
const landingFits = p => p.evaluate(() => {
  const H = window.innerHeight;
  const sel = [".landing-title", ".input-pill", ".ai-notes-landing"];
  const bottoms = sel.map(s => {
    const el = document.querySelector(s);
    return el ? Math.round(el.getBoundingClientRect().bottom) : null;
  });
  return { H, bottoms, fits: bottoms.every(b => b === null || b <= H),
           scroll: document.documentElement.scrollHeight - H };
});

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME,
    headless: "new", args: ["--no-sandbox"] });

  // ================= switching =================
  console.log("\n== switching ==");
  {
    const p = await newPage(browser, { width: 1280, height: 860 });
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    const cls = await p.evaluate(() => document.body.className);
    check("?v=3 opens the third skin", /\bv3\b/.test(cls) && /compact/.test(cls), cls);

    await p.goto(BASE + "/widget", { waitUntil: "networkidle2" });
    await p.evaluate(() => document.activeElement && document.activeElement.blur());
    await p.keyboard.press("3");
    const on = await p.evaluate(() => document.body.className);
    check("3 switches it on", /\bv3\b/.test(on) && /awaiting/.test(on), on);
    const typed = await p.evaluate(() => document.getElementById("query").value);
    check("and does not type a 3", typed === "", JSON.stringify(typed));
    await p.keyboard.press("3");
    const off = await p.evaluate(() => document.body.className);
    check("3 again goes back to compact", !/\bv3\b/.test(off) && /compact/.test(off), off);
    await p.keyboard.press("3");
    await p.keyboard.press("2");
    const two = await p.evaluate(() => document.body.className);
    check("2 from the third skin lands in compact", !/\bv3\b/.test(two) && /compact/.test(two), two);
    await p.close();
  }

  // ================= landing, desktop =================
  console.log("\n== landing, desktop ==");
  {
    const p = await newPage(browser, { width: 1280, height: 860 });
    await p.goto(BASE + "/widget", { waitUntil: "networkidle2" });
    await sleep(300);
    const restColour = await p.evaluate(() =>
      getComputedStyle(document.getElementById("query"), "::placeholder").color);
    const before = await p.evaluate(() => {
      const r = document.querySelector(".input-pill").getBoundingClientRect();
      return { w: Math.round(r.width), h: Math.round(r.height), t: Math.round(r.top) };
    });
    await p.click("#query");
    await sleep(600);
    const focusColour = await p.evaluate(() =>
      getComputedStyle(document.getElementById("query"), "::placeholder").color);
    const after = await p.evaluate(() => {
      const r = document.querySelector(".input-pill").getBoundingClientRect();
      return { w: Math.round(r.width), h: Math.round(r.height), t: Math.round(r.top) };
    });
    check("placeholder is darker at rest", restColour === "rgb(95, 99, 104)", restColour);
    check("and light again once focused", focusColour === "rgb(168, 173, 179)", focusColour);
    check("composer does not widen on focus", after.w === before.w, `${before.w} -> ${after.w}`);
    check("or grow or drop", after.h === before.h && after.t === before.t,
          JSON.stringify({ before, after }));
    await p.close();
  }

  // ================= landing, short frames =================
  console.log("\n== landing, short frames ==");
  for (const [label, vp, mobile, v] of [
    ["desktop 1280x240", { width: 1280, height: 240 }, false, ""],
    ["desktop 1280x240, third skin", { width: 1280, height: 240 }, false, "?v=3"],
    ["phone 390x300", { width: 390, height: 300 }, true, ""],
    ["phone 390x300, third skin", { width: 390, height: 300 }, true, "?v=3"],
  ]) {
    const p = await newPage(browser, vp, mobile);
    await p.goto(BASE + "/widget" + v, { waitUntil: "networkidle2" });
    await sleep(500);
    const fit = await landingFits(p);
    check(`${label}: everything fits`, fit.fits && fit.scroll <= 0, JSON.stringify(fit));
    await p.focus("#query");
    await p.keyboard.type("sch", { delay: 5 });
    await sleep(450);
    const sug = await p.evaluate(() => {
      const panel = document.querySelector(".suggest");
      if (!panel) return { rows: 0, bottom: 0 };
      return { rows: panel.querySelectorAll(".suggest-item").length,
               bottom: Math.round(panel.getBoundingClientRect().bottom) };
    });
    if (mobile) {
      check(`${label}: no suggestions while typing`, sug.rows === 0, JSON.stringify(sug));
    } else {
      check(`${label}: suggestions stop at the frame`, sug.bottom <= vp.height,
            JSON.stringify({ ...sug, H: vp.height }));
    }
    await shot(p, "landing-" + label.replace(/[^a-z0-9]+/gi, "-"));
    await p.close();
  }

  // ================= the conversation, desktop =================
  console.log("\n== conversation, desktop ==");
  {
    const p = await newPage(browser, { width: 1100, height: 760 });
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    await sleep(300);
    await shot(p, "v3-landing-desktop");
    await ask(p, "when does school start");

    // The launch, caught mid-flight.
    const flight = await p.evaluate(() => {
      const b = document.querySelector(".chat-bubble.user");
      const t = b && b.querySelector(".ub-text");
      return b && { launch: b.classList.contains("launch"),
                    anim: getComputedStyle(b).animationName,
                    textAnim: t && getComputedStyle(t).animationName };
    });
    check("the question rises in, blurred and small",
          flight && flight.anim === "bubbleLaunch" && flight.textAnim === "textFocus",
          JSON.stringify(flight));
    await sleep(1400);

    const asked = p.__asks[0] || {};
    check("the third skin asks for follow-ups", asked.followups === true,
          JSON.stringify({ followups: asked.followups }));

    const look = await p.evaluate(() => {
      const bot = document.querySelector(".chat-bubble.bot");
      const user = document.querySelector(".chat-bubble.user");
      const cs = el => getComputedStyle(el);
      return { botBg: cs(bot).backgroundColor, userBg: cs(user).backgroundColor,
               botW: Math.round(bot.getBoundingClientRect().width),
               colW: Math.round(document.getElementById("messages").clientWidth),
               launchGone: !user.classList.contains("launch") };
    });
    check("the answer is not in a bubble", look.botBg === "rgba(0, 0, 0, 0)", look.botBg);
    check("the question still is", look.userBg === "rgb(230, 0, 35)", look.userBg);
    check("the launch class is taken off after", look.launchGone);

    const src = await p.evaluate(() => {
      const f = document.querySelector(".answer-footer");
      const stack = f.querySelector(".source-stack");
      const row = f.querySelector(".source-pills");
      return { stack: stack && getComputedStyle(stack).display,
               row: row && getComputedStyle(row).display,
               favs: stack ? stack.querySelectorAll(".fav").length : 0,
               text: stack && stack.textContent.trim() };
    });
    check("three sources fold into one pill", /flex/.test(src.stack || "") && src.row === "none",
          JSON.stringify(src));
    check("carrying one icon per site", src.favs === 2, `${src.favs} icons`);

    const btn = await p.evaluate(() => {
      const r = s => { const el = document.querySelector(s); return el && getComputedStyle(el).borderRadius; };
      return { copy: r(".fb-copy"), pair: r(".fb-rate"),
               up: r('.fb-btn[data-type="up"]'),
               shown: getComputedStyle(document.querySelector(".feedback-inline")).opacity };
    });
    check("copy is a circle", btn.copy === "50%", btn.copy);
    check("the thumbs are circles in a pill", btn.up === "50%" && btn.pair === "999px",
          JSON.stringify(btn));
    check("and they are visible without a hover", btn.shown === "1", btn.shown);

    await p.click(".fb-copy");
    await sleep(250);
    const copied = await p.evaluate(async () => ({
      cls: document.querySelector(".fb-copy").classList.contains("copied"),
      text: await navigator.clipboard.readText().catch(e => "ERR " + e) }));
    check("copy puts the answer on the clipboard, links kept",
          copied.cls && /8:15 AM/.test(copied.text)
          && /bell schedules \(https:\/\/vrhs\.leanderisd\.org/.test(copied.text),
          JSON.stringify({ cls: copied.cls, head: copied.text.slice(0, 120) }));

    const chips = await p.evaluate(() => {
      const strip = document.querySelector(".quick-prompts");
      const shown = [...strip.querySelectorAll("button")]
        .filter(b => getComputedStyle(b).display !== "none")
        .map(b => b.textContent.trim());
      return shown;
    });
    check("the strip is rewritten for the answer",
          JSON.stringify(chips) === JSON.stringify(FOLLOWUPS), JSON.stringify(chips));

    // Notes: over white at the end, gone once text is under them.
    await p.evaluate(() => { const m = document.getElementById("messages");
      m.scrollTo({ top: m.scrollHeight, behavior: "instant" }); });
    await sleep(400);
    const atEnd = await p.evaluate(() => [...document.querySelectorAll(".ai-notes-row .ai-note")]
      .map(n => ({ covered: n.classList.contains("covered"),
                   op: getComputedStyle(n).opacity })));
    check("the notes show over white space", atEnd.every(n => !n.covered), JSON.stringify(atEnd));
    await shot(p, "v3-answer-desktop");

    await p.evaluate(() => { const m = document.getElementById("messages");
      m.scrollTo({ top: m.scrollHeight - m.clientHeight - 160, behavior: "instant" }); });
    await sleep(400);
    const over = await p.evaluate(() => [...document.querySelectorAll(".ai-notes-row .ai-note")]
      .map(n => n.classList.contains("covered")));
    check("and step aside when text is under them", over.some(Boolean), JSON.stringify(over));
    await shot(p, "v3-notes-covered-desktop");
    await p.evaluate(() => { const m = document.getElementById("messages");
      m.scrollTo({ top: m.scrollHeight, behavior: "instant" }); });
    await sleep(300);

    // The sources sheet.
    await p.click(".source-stack");
    await sleep(600);
    const sheet = await p.evaluate(() => {
      const s = document.querySelector(".sheet");
      if (!s) return null;
      const panel = s.querySelector(".sheet-panel");
      const r = panel.getBoundingClientRect();
      return { title: s.querySelector("h3").textContent,
               groups: [...s.querySelectorAll(".src-group summary span:nth-child(2)")].map(x => x.textContent),
               counts: [...s.querySelectorAll(".src-group")].map(g => g.querySelectorAll(".src-item").length),
               blur: getComputedStyle(s).backdropFilter,
               bottom: Math.round(r.bottom), H: window.innerHeight,
               close: !!s.querySelector(".sheet-close") };
    });
    check("the pill opens a sheet from the bottom",
          sheet && sheet.title === "Sources" && Math.abs(sheet.bottom - sheet.H) <= 1,
          JSON.stringify(sheet));
    check("over a blurred page", sheet && /blur/.test(sheet.blur), sheet && sheet.blur);
    check("grouped by site", sheet && sheet.groups.length === 2
          && sheet.counts.join() === "2,1", JSON.stringify(sheet && { g: sheet.groups, c: sheet.counts }));
    await shot(p, "v3-sources-sheet");
    await p.click(".src-group summary");
    await sleep(200);
    const folded = await p.evaluate(() => document.querySelector(".src-group").open);
    check("each group folds", folded === false);
    await p.click(".sheet-close");
    await sleep(600);
    check("the X closes it", await p.evaluate(() => !document.querySelector(".sheet")));

    // The notes' own sheets.
    await p.click('.ai-notes-row .ai-note[data-sheet="ai"]');
    await sleep(600);
    const ai = await p.evaluate(() => {
      const s = document.querySelector(".sheet");
      return s && { title: s.querySelector("h3").textContent,
                    lead: s.querySelector(".sheet-lead").textContent };
    });
    check("How AI works opens its sheet",
          ai && ai.title === "How AI works" && /official sources/.test(ai.lead),
          JSON.stringify(ai));
    await shot(p, "v3-how-ai-works");
    await p.keyboard.press("Escape");
    await sleep(600);
    check("Escape closes it", await p.evaluate(() => !document.querySelector(".sheet")));

    await p.click('.ai-notes-row .ai-note[data-sheet="privacy"]');
    await sleep(600);
    const priv = await p.evaluate(() => [...document.querySelectorAll(".sheet .sheet-points strong")]
      .map(x => x.textContent));
    check("Your privacy lists its points", priv.includes("Limited retention"), JSON.stringify(priv));
    await shot(p, "v3-privacy");
    await p.mouse.click(10, 10);
    await sleep(600);
    check("a click outside closes it", await p.evaluate(() => !document.querySelector(".sheet")));

    // The composer rises and the end stays in view.
    const rise0 = await p.evaluate(() => ({
      pad: getComputedStyle(document.querySelector(".chat-input")).paddingBottom,
      pill: Math.round(document.querySelector(".input-pill").getBoundingClientRect().top) }));
    await p.click("#query");
    await sleep(700);
    const rise1 = await p.evaluate(() => {
      const m = document.getElementById("messages");
      return { pad: getComputedStyle(document.querySelector(".chat-input")).paddingBottom,
               pill: Math.round(document.querySelector(".input-pill").getBoundingClientRect().top),
               behind: Math.round(m.scrollHeight - m.scrollTop - m.clientHeight),
               bg: getComputedStyle(document.querySelector(".input-pill")).backgroundColor,
               shadow: getComputedStyle(document.querySelector(".input-pill")).boxShadow !== "none" };
    });
    check("the composer rises on focus", rise1.pill < rise0.pill,
          `${rise0.pill} -> ${rise1.pill} (padding ${rise0.pad} -> ${rise1.pad})`);
    check("and the conversation keeps its end in view", rise1.behind <= 2, `${rise1.behind}px behind`);
    check("the composer is white with a shadow", rise1.bg === "rgb(255, 255, 255)" && rise1.shadow,
          JSON.stringify({ bg: rise1.bg, shadow: rise1.shadow }));
    check("no page errors (desktop)", p.__err.length === 0, p.__err.join(" | "));
    await p.close();
  }

  // ================= the conversation, phone =================
  console.log("\n== conversation, phone ==");
  {
    const p = await newPage(browser, { width: 390, height: 700 }, true);
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    await sleep(300);
    await shot(p, "v3-landing-phone");
    await ask(p, "when does school start");
    await sleep(1500);
    await p.evaluate(() => document.activeElement && document.activeElement.blur());
    await sleep(500);
    await shot(p, "v3-answer-phone");

    const hidden = await p.evaluate(() => document.querySelector(".to-end").classList.contains("show"));
    check("no scroll button at the end", hidden === false);
    await p.evaluate(() => document.getElementById("messages").scrollTo({ top: 0, behavior: "instant" }));
    await sleep(500);
    const btn = await p.evaluate(() => {
      const b = document.querySelector(".to-end");
      const cs = getComputedStyle(b);
      return { show: b.classList.contains("show"), op: cs.opacity,
               blur: cs.backdropFilter, radius: cs.borderRadius };
    });
    check("scrolled up, the button appears", btn.show && btn.op === "1", JSON.stringify(btn));
    check("a frosted circle", /blur/.test(btn.blur) && btn.radius === "50%", JSON.stringify(btn));
    await shot(p, "v3-scroll-button-phone");
    await p.tap(".to-end");
    await sleep(1200);
    const end = await p.evaluate(() => {
      const m = document.getElementById("messages");
      return { behind: Math.round(m.scrollHeight - m.scrollTop - m.clientHeight),
               show: document.querySelector(".to-end").classList.contains("show") };
    });
    check("and brings the conversation back to its end", end.behind <= 2 && !end.show,
          JSON.stringify(end));
    const overflow = await p.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    check("no horizontal overflow", overflow <= 0, `${overflow}px`);
    check("no page errors (phone)", p.__err.length === 0, p.__err.join(" | "));
    await p.close();
  }

  // The scroll button is a phone control.
  {
    const p = await newPage(browser, { width: 1100, height: 700 });
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    await ask(p, "when does school start");
    await sleep(1500);
    await p.evaluate(() => document.getElementById("messages").scrollTo({ top: 0, behavior: "instant" }));
    await sleep(400);
    const d = await p.evaluate(() => getComputedStyle(document.querySelector(".to-end")).display);
    check("no scroll button on a desktop", d === "none", d);
    await p.close();
  }

  await browser.close();
  const passed = out.filter(Boolean).length;
  console.log(`\n${passed}/${out.length} checks passed`);
  process.exit(passed === out.length ? 0 : 1);
})();

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

async function newPage(browser, vp, mobile, keyboard) {
  const ctx = browser.defaultBrowserContext();
  await ctx.overridePermissions(BASE, ["clipboard-read", "clipboard-write",
                                       "clipboard-sanitized-write"]);
  const p = await browser.newPage();
  await p.setViewport({ ...vp, isMobile: !!mobile, hasTouch: !!mobile,
                        deviceScaleFactor: 1 });
  if (keyboard) {
    // Headless Chrome has no on-screen keyboard, so the visual viewport is
    // replaced with one whose height the test can take a keyboard out of.
    await p.evaluateOnNewDocument(() => {
      const vv = new EventTarget();
      window.__kb = 0;
      Object.defineProperties(vv, {
        height: { get: () => window.innerHeight - window.__kb },
        width: { get: () => window.innerWidth },
        scale: { get: () => 1 },
        offsetTop: { get: () => 0 },
        offsetLeft: { get: () => 0 },
      });
      Object.defineProperty(window, "visualViewport", { get: () => vv });
    });
  }
  p.__delay = 0;
  await p.setRequestInterception(true);
  p.__asks = [];
  p.on("request", req => {
    if (req.url().endsWith("/ask")) {
      p.__asks.push(JSON.parse(req.postData() || "{}"));
      const send = () => req.respond({ status: 200,
        contentType: "text/plain; charset=utf-8",
        body: ANSWER + ":::meta" + META });
      // A queue of per-request delays when a test needs answers to finish
      // out of order; otherwise the one delay for every request.
      const wait = (p.__delays && p.__delays.length) ? p.__delays.shift() : p.__delay;
      if (wait) { setTimeout(send, wait); return; }
      return send();
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
  const sel = [".landing-title", ".input-pill"];
  const bottoms = sel.map(s => {
    const el = document.querySelector(s);
    return el ? Math.round(el.getBoundingClientRect().bottom) : null;
  });
  // The body too, not only the document: a phone lets an embedded frame be
  // scrolled to anything below it even with overflow hidden, and a stray
  // character after a script tag once hung a 19px line of text down there.
  const title = document.querySelector(".landing-title");
  return { H, bottoms, fits: bottoms.every(b => b === null || b <= H),
           scroll: Math.max(document.documentElement.scrollHeight,
                            document.body.scrollHeight) - H,
           size: Math.round(parseFloat(getComputedStyle(title).fontSize)) };
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
    const plain = await p.evaluate(() => document.body.className);
    check("and so does a plain /widget, the default now", /\bv3\b/.test(plain), plain);
    await p.goto(BASE + "/widget?v=2", { waitUntil: "networkidle2" });
    const second = await p.evaluate(() => document.body.className);
    check("?v=2 opens the second", !/\bv3\b/.test(second) && /compact/.test(second), second);

    await p.goto(BASE + "/widget?v=2", { waitUntil: "networkidle2" });
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
    await p.goto(BASE + "/widget?v=2", { waitUntil: "networkidle2" });
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
    await p.evaluate(() => document.body.classList.add("v3"));
    const landingNotes = await p.evaluate(() => [...document.querySelectorAll(".ai-note")]
      .filter(n => n.getBoundingClientRect().width > 0).length);
    check("the notes are not on the landing screen", landingNotes === 0, `${landingNotes} shown`);
    await p.evaluate(() => document.body.classList.remove("v3"));
    check("and light again once focused", focusColour === "rgb(168, 173, 179)", focusColour);
    check("composer does not widen on focus", after.w === before.w, `${before.w} -> ${after.w}`);
    check("or grow or drop", after.h === before.h && after.t === before.t,
          JSON.stringify({ before, after }));
    await p.close();
  }

  // ================= landing, short frames =================
  console.log("\n== landing, short frames ==");
  for (const [label, vp, mobile, v] of [
    ["desktop 1280x240", { width: 1280, height: 240 }, false, "?v=2"],
    ["desktop 1280x240, third skin", { width: 1280, height: 240 }, false, "?v=3"],
    ["phone 390x300", { width: 390, height: 300 }, true, "?v=2"],
    ["phone 390x300, third skin", { width: 390, height: 300 }, true, "?v=3"],
    ["phone 390x220, third skin", { width: 390, height: 220 }, true, "?v=3"],
    ["phone 390x150, third skin", { width: 390, height: 150 }, true, "?v=3"],
    ["desktop 1280x180, third skin", { width: 1280, height: 180 }, false, "?v=3"],
    ["desktop 800x150, third skin", { width: 800, height: 150 }, false, "?v=3"],
    ["desktop 800x110, third skin", { width: 800, height: 110 }, false, "?v=3"],
  ]) {
    const p = await newPage(browser, vp, mobile);
    await p.goto(BASE + "/widget" + v, { waitUntil: "networkidle2" });
    await sleep(500);
    const fit = await landingFits(p);
    check(`${label}: everything fits`, fit.fits && fit.scroll <= 0, JSON.stringify(fit));
    // Full size wherever the frame has room for it, and where it has not,
    // stepped down only as far as it must. Which case applies is measured,
    // not guessed: the headline wraps differently at different sizes, so
    // whether full size fits is a question for the layout.
    const sizing = await p.evaluate(() => {
      const t = document.querySelector(".landing-title");
      const row = document.querySelector(".chat-input");
      const pill = document.querySelector(".input-pill");
      const fitsAt = (px) => {
        const was = t.style.fontSize;
        t.style.fontSize = px ? px + "px" : "";
        const gap = parseFloat(getComputedStyle(t).marginBottom) || 0;
        const ok = t.offsetHeight <= row.clientHeight - pill.offsetHeight - gap - 8;
        t.style.fontSize = was;
        return ok;
      };
      const size = parseFloat(getComputedStyle(t).fontSize);
      const full = (() => { const was = t.style.fontSize; t.style.fontSize = "";
        const f = parseFloat(getComputedStyle(t).fontSize); t.style.fontSize = was; return f; })();
      return { size, full, fullFits: fitsAt(null), biggerFits: size < full && fitsAt(size + 3) };
    });
    check(`${label}: the headline is as large as the frame allows`,
          sizing.fullFits ? sizing.size === sizing.full : !sizing.biggerFits && sizing.size >= 16,
          JSON.stringify(sizing));
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
    const arrive = await p.evaluate(() => {
      const a = s => { const el = document.querySelector(s); const cs = getComputedStyle(el);
        return cs.animationName + "@" + cs.getPropertyValue("--i").trim(); };
      return [a(".source-stack"), a(".fb-copy"), a(".fb-rate")];
    });
    check("the footer arrives in order rather than appearing",
          arrive.join() === "chipIn@0,chipIn@1,chipIn@2", JSON.stringify(arrive));
    check("three sources fold into one pill", /flex/.test(src.stack || "") && src.row === "none",
          JSON.stringify(src));
    check("carrying one icon per site", src.favs === 2, `${src.favs} icons`);
    const icons = await p.evaluate(() => [...document.querySelectorAll(".source-stack .fav img")]
      .map(i => i.getAttribute("src").replace(/^https:\/\/[^/]+/, "")));
    check("a Google Doc gets the Docs icon, not Google's G",
          icons.some(s => /docs_2020q4/.test(s)) && icons.some(s => /s2\/favicons/.test(s)),
          JSON.stringify(icons));

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
    const copiedBg = await p.evaluate(() =>
      getComputedStyle(document.querySelector(".fb-copy")).backgroundColor);
    check("copied is red", copiedBg === "rgb(230, 0, 35)", copiedBg);
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

    await p.click('.fb-btn[data-type="down"]');
    await sleep(300);
    const downBg = await p.evaluate(() => {
      const cs = getComputedStyle(document.querySelector('.fb-btn[data-type="down"]'));
      return cs.backgroundColor + " / " + cs.color;
    });
    check("thumbs-down is red when chosen", downBg === "rgb(230, 0, 35) / rgb(255, 255, 255)", downBg);
    const fills = await p.evaluate(() => [...document.querySelectorAll('.fb-btn[data-type="down"] svg path')]
      .map(x => getComputedStyle(x).fill));
    check("and its thumb stays an outline", fills.every(f => f === "none"), JSON.stringify(fills));
    await p.keyboard.press("Escape");
    await sleep(300);

    const noteColour = await p.evaluate(() =>
      getComputedStyle(document.querySelector(".ai-notes-row .ai-note")).color);
    check("the notes are light", noteColour === "rgb(176, 181, 186)", noteColour);

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
    const aiIcons = await p.evaluate(() => document.querySelectorAll(".sheet svg").length);
    check("with no icons but the close button", aiIcons === 1, `${aiIcons} svg`);
    check("How AI works opens its sheet",
          ai && ai.title === "How AI works" && /official sources/.test(ai.lead),
          JSON.stringify(ai));
    const card = await p.evaluate(() => {
      const panel = document.querySelector(".sheet-panel");
      const r = panel.getBoundingClientRect();
      const b = panel.querySelector(".sheet-body");
      return { mid: Math.round(r.top + r.height / 2), H: window.innerHeight,
               w: Math.round(r.width), scroll: b.scrollHeight - b.clientHeight };
    });
    check("on a desktop it is a card in the middle",
          Math.abs(card.mid - card.H / 2) <= 4 && card.w <= 380, JSON.stringify(card));
    check("with nothing to scroll", card.scroll <= 0, JSON.stringify(card));
    await shot(p, "v3-how-ai-works");
    await p.keyboard.press("Escape");
    await sleep(600);
    check("Escape closes it", await p.evaluate(() => !document.querySelector(".sheet")));

    await p.click('.ai-notes-row .ai-note[data-sheet="privacy"]');
    await sleep(600);
    const priv = await p.evaluate(() => [...document.querySelectorAll(".sheet .sheet-points strong")]
      .map(x => x.textContent));
    check("Your privacy lists its points", priv.includes("Limited retention") && priv.length === 2,
          JSON.stringify(priv));
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
    const edge = await p.evaluate(() =>
      getComputedStyle(document.querySelector(".input-pill")).borderTopColor);
    check("the composer's outline is darker", edge === "rgb(154, 160, 166)" ||
          edge === "rgb(230, 0, 35)", edge);
    check("the composer is white with a shadow", rise1.bg === "rgb(255, 255, 255)" && rise1.shadow,
          JSON.stringify({ bg: rise1.bg, shadow: rise1.shadow }));
    await p.evaluate(() => document.activeElement.blur());
    await sleep(500);
    const reach = await p.evaluate(() => {
      const pill = document.querySelector(".input-pill").getBoundingClientRect();
      const box = document.querySelector(".chat-container").getBoundingClientRect();
      return { gap: Math.round(box.bottom - pill.bottom),
               shadow: getComputedStyle(document.querySelector(".input-pill")).boxShadow };
    });
    // The shadow's reach below the pill is its offset plus its blur.
    check("the shadow stays inside the frame", /0px 2px 6px/.test(reach.shadow) && reach.gap >= 8,
          JSON.stringify(reach));

    // A second question: the strip leaves while it is pending and comes
    // back rewritten.
    p.__delay = 1500;
    await p.click(".quick-prompts .qp-tailored");
    await sleep(450);
    const pending = await p.evaluate(() => {
      const strip = document.querySelector(".quick-prompts");
      const b = strip.querySelector("button:not([style*='display: none'])");
      return { away: strip.classList.contains("qp-away"),
               op: [...strip.querySelectorAll(".qp-tailored")].map(x => getComputedStyle(x).opacity) };
    });
    check("the suggestions leave while the answer is pending",
          pending.away && pending.op.every(o => +o < 0.05), JSON.stringify(pending));
    await sleep(2600);
    const back = await p.evaluate(() => {
      const strip = document.querySelector(".quick-prompts");
      return { away: strip.classList.contains("qp-away"),
               chips: [...strip.querySelectorAll(".qp-tailored")].map(x => x.textContent) };
    });
    check("and come back, rewritten, once it is complete",
          !back.away && back.chips.length > 0 && !back.chips.some(c => /time does (the )?school( day)? end/i.test(c)),
          JSON.stringify(back));
    p.__delay = 0;
    check("no page errors (desktop)", p.__err.length === 0, p.__err.join(" | "));
    await p.close();
  }

  // ================= the headline at its own size =================
  console.log("\n== the landing headline ==");
  for (const [label, vp, mobile, want] of [
    ["phone 390x844", { width: 390, height: 844 }, true, 32],
    ["desktop 1280x860", { width: 1280, height: 860 }, false, 56],
  ]) {
    const p = await newPage(browser, vp, mobile);
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    const px = await p.evaluate(() =>
      Math.round(parseFloat(getComputedStyle(document.querySelector(".landing-title")).fontSize)));
    check(`${label}: the headline is its normal size`, px === want, `${px}px`);
    await p.close();
  }

  // ================= the feedback panel on a phone =================
  console.log("\n== the feedback panel on a phone ==");
  {
    const p = await newPage(browser, { width: 390, height: 760 }, true);
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    await sleep(300);
    await ask(p, "when does school start");
    await sleep(1500);
    await p.tap('.fb-btn[data-type="down"]');
    await sleep(500);
    const fp = await p.evaluate(() => {
      const panel = document.querySelector(".fb-panel:not([hidden])");
      const ta = panel.querySelector(".fb-input");
      const pr = panel.getBoundingClientRect(), tr = ta.getBoundingClientRect();
      const send = panel.querySelector(".fb-send").getBoundingClientRect();
      return { w: Math.round(pr.width), h: Math.round(pr.height),
               font: parseFloat(getComputedStyle(ta).fontSize),
               drawn: Math.round(parseFloat(getComputedStyle(ta).fontSize) * tr.width / ta.offsetWidth * 10) / 10,
               taRight: Math.round(tr.right), inner: Math.round(pr.right - 10),
               gapToSend: Math.round(send.top - tr.bottom) };
    });
    check("it is compact", fp.w <= 232 && fp.h <= 190, JSON.stringify(fp));
    check("its comment box is 16px to iOS and drawn smaller",
          fp.font === 16 && fp.drawn < 13.5, JSON.stringify({ font: fp.font, drawn: fp.drawn }));
    check("and fills the panel without spilling or leaving a gap",
          Math.abs(fp.taRight - fp.inner) <= 2 && fp.gapToSend >= 0 && fp.gapToSend <= 26,
          JSON.stringify(fp));
    await shot(p, "v3-feedback-phone");
    await p.close();
  }

  // ================= the notes wait for the first answer =================
  console.log("\n== the notes wait for the first answer ==");
  {
    const p = await newPage(browser, { width: 1100, height: 760 });
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    await sleep(300);
    p.__delay = 1500;
    await ask(p, "when does school start");
    await sleep(500);
    const during = await p.evaluate(() =>
      getComputedStyle(document.querySelector(".ai-notes-row .ai-notes")).display);
    check("the notes are hidden while the first answer is pending", during === "none", during);
    await sleep(2600);
    const after = await p.evaluate(() => {
      const n = document.querySelector(".ai-notes-row .ai-notes");
      return { display: getComputedStyle(n).display,
               copy: (() => { const cs = getComputedStyle(document.querySelector(".fb-copy"));
                              return cs.backgroundColor + " / " + cs.borderTopStyle; })(),
               pair: (() => { const cs = getComputedStyle(document.querySelector(".fb-rate"));
                              return cs.backgroundColor + " / " + cs.borderTopStyle; })() };
    });
    check("and appear once it has finished", after.display === "flex", after.display);
    check("the copy and rating controls are grey discs with no outline",
          after.copy === "rgb(241, 243, 244) / none" && after.pair === "rgb(241, 243, 244) / none",
          JSON.stringify(after));
    await p.click('.ai-notes-row .ai-note[data-sheet="privacy"]');
    await sleep(600);
    const priv = await p.evaluate(() => ({
      text: document.querySelector(".sheet").textContent,
      icons: [...document.querySelectorAll(".sheet .pt-icon")].map(i => getComputedStyle(i).backgroundColor) }));
    check("the privacy note names no provider", !/openai/i.test(priv.text), priv.text.slice(0, 160));
    check("and its icons sit on light grey", priv.icons.every(c => c === "rgb(241, 243, 244)"),
          JSON.stringify(priv.icons));
    await p.close();
  }

  // ================= a question sent before the last one finished ========
  console.log("\n== asking again before the last answer finishes ==");
  {
    const p = await newPage(browser, { width: 1100, height: 760 });
    await p.goto(BASE + "/widget", { waitUntil: "networkidle2" });
    await sleep(300);
    // The first answer is held back; the second arrives at once and finishes
    // first. The first one's sources and rating must still land under it.
    p.__delays = [1800, 0];
    await ask(p, "when does school start");
    await sleep(200);
    await ask(p, "where is the library website");
    await sleep(3200);
    const order = await p.evaluate(() => [...document.getElementById("messages").children]
      .map(el => el.classList.contains("user") ? "Q"
               : el.classList.contains("bot") ? "A"
               : el.classList.contains("answer-footer") ? "F"
               : el.classList.contains("verify-card") ? "C"
               : el.className).join(" "));
    check("each answer keeps its own row of sources under it",
          order === "Q A F Q A F", order);
    const rows = await p.evaluate(() =>
      [...document.querySelectorAll(".answer-footer")].map(f =>
        f.previousElementSibling && f.previousElementSibling.classList.contains("bot")));
    check("no answer has two rows", rows.length === 2 && rows.every(Boolean), JSON.stringify(rows));
    await p.close();
  }

  // ================= the hover outline leaving =================
  console.log("\n== the hover outline ==");
  {
    const p = await newPage(browser, { width: 1100, height: 760 });
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    await sleep(300);
    const ring = () => p.evaluate(() => {
      const svg = document.querySelector(".pill-snake");
      const r = svg.querySelector("rect");
      const cs = getComputedStyle(r);
      return { op: +getComputedStyle(svg).opacity,
               off: parseFloat(cs.strokeDashoffset),
               w: parseFloat(cs.strokeWidth) };
    });
    await p.hover(".input-pill");
    await sleep(800);
    const drawn = await ring();
    check("hovered, the ring draws on", drawn.op === 1 && drawn.off < 1, JSON.stringify(drawn));

    await p.mouse.move(5, 5);
    await sleep(200);
    const mid = await ring();
    check("pointer off, it undraws rather than vanishing",
          mid.op === 1 && mid.off > 1 && mid.off < 99, JSON.stringify(mid));
    await sleep(900);
    const gone = await ring();
    check("and is gone once it has", gone.op === 0, JSON.stringify(gone));

    await p.hover(".input-pill");
    await sleep(800);
    await p.click("#query");
    await sleep(140);
    const thinning = await ring();
    check("clicked, it thins into the red outline",
          thinning.op === 1 && thinning.off < 1 && thinning.w > 0 && thinning.w < 1.1,
          JSON.stringify(thinning));
    await sleep(700);
    const thin = await ring();
    const border = await p.evaluate(() =>
      getComputedStyle(document.querySelector(".input-pill")).borderTopColor);
    check("leaving the red outline standing", thin.op === 0 && thin.w === 0
          && border === "rgb(230, 0, 35)", JSON.stringify({ ...thin, border }));
    await p.close();
  }

  // ================= a short frame, on a phone =================
  console.log("\n== a short frame, on a phone ==");
  for (const [h, want, zoom] of [[700, "", "1"], [520, "fit-1", "0.92"],
                                 [420, "fit-2", "0.84"], [300, "fit-3", "0.77"]]) {
    const p = await newPage(browser, { width: 390, height: h }, true);
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    await sleep(300);
    const landingZoom = await p.evaluate(() =>
      getComputedStyle(document.querySelector(".chat-container")).zoom);
    check(`${h}px tall: the landing screen is not scaled`, landingZoom === "1", landingZoom);

    // The first-question notice takes the same step.
    await p.focus("#query");
    await p.keyboard.type("when does school start", { delay: 3 });
    await p.keyboard.press("Enter");
    await sleep(450);
    const noticeZoom = await p.evaluate(() => {
      const c = document.querySelector(".disclaimer .disclaimer-card");
      return c ? getComputedStyle(c).zoom : "none";
    });
    check(`${h}px tall: the notice takes the step too`, noticeZoom === zoom, noticeZoom);
    await p.click(".disclaimer .accept");
    await sleep(1500);
    await p.evaluate(() => document.activeElement && document.activeElement.blur());
    await sleep(500);
    const fit = await p.evaluate(() => {
      const cls = ["fit-1", "fit-2", "fit-3"].filter(c => document.body.classList.contains(c));
      const pill = document.querySelector(".input-pill").getBoundingClientRect();
      const input = document.getElementById("query");
      return { cls: cls.join(), zoom: getComputedStyle(document.querySelector(".chat-container")).zoom,
               pillBottom: Math.round(pill.bottom), H: window.innerHeight,
               inputPx: Math.round(parseFloat(getComputedStyle(input).fontSize)
                        * parseFloat(getComputedStyle(document.querySelector(".chat-container")).zoom)),
               pan: document.documentElement.scrollWidth - window.innerWidth };
    });
    check(`${h}px tall: ${want || "full size"}`,
          fit.cls === want && fit.zoom === zoom && fit.pillBottom <= fit.H && fit.pan <= 0,
          JSON.stringify(fit));
    check(`${h}px tall: the composer's text is still 16px on screen`,
          fit.inputPx === 16, `${fit.inputPx}px`);
    const corner = await p.evaluate(() => {
      const r = document.querySelector(".pill-snake rect");
      return { rx: +r.getAttribute("rx"), half: r.getBBox().height / 2 };
    });
    check(`${h}px tall: the hover ring's corners are still a pill`,
          Math.abs(corner.rx - corner.half) <= 0.5, JSON.stringify(corner));
    if (h === 300) await shot(p, "v3-short-phone");
    {
      // The scroll-to-end button sits lower the shorter the frame.
      await p.evaluate(() => document.getElementById("messages").scrollTo({ top: 0, behavior: "instant" }));
      await sleep(400);
      const te = await p.evaluate(() => {
        const b = document.querySelector(".to-end");
        return { show: b.classList.contains("show"), bottom: getComputedStyle(b).bottom };
      });
      const wantBottom = { "": "34px", "fit-1": "26px", "fit-2": "18px", "fit-3": "10px" }[want];
      check(`${h}px tall: the scroll button sits ${wantBottom} up`,
            te.show && te.bottom === wantBottom, JSON.stringify(te));
      await p.evaluate(() => document.getElementById("messages")
        .scrollTo({ top: document.getElementById("messages").scrollHeight, behavior: "instant" }));
      await sleep(300);
    }
    if (h === 300) {
      // The "what went wrong" panel takes the step too, and still opens
      // against the button that opened it.
      await p.tap('.fb-btn[data-type="down"]');
      await sleep(500);
      const fbp = await p.evaluate(() => {
        const panel = document.querySelector(".fb-panel:not([hidden])");
        const down = document.querySelector('.fb-btn[data-type="down"]').getBoundingClientRect();
        const r = panel.getBoundingClientRect();
        return { zoom: getComputedStyle(panel).zoom,
                 left: Math.round(r.left), right: Math.round(r.right), downRight: Math.round(down.right),
                 top: Math.round(r.top), bottom: Math.round(r.bottom),
                 downTop: Math.round(down.top), downBottom: Math.round(down.bottom),
                 W: window.innerWidth, H: window.innerHeight,
                 inputPx: Math.round(parseFloat(getComputedStyle(panel.querySelector(".fb-input")).fontSize)
                          * parseFloat(getComputedStyle(panel).zoom)),
                 // What the eye sees: the font, the zoom, and the drawing
                 // scale on top.
                 drawnPx: Math.round(parseFloat(getComputedStyle(panel.querySelector(".fb-input")).fontSize)
                          * parseFloat(getComputedStyle(panel).zoom)
                          * panel.querySelector(".fb-input").getBoundingClientRect().width
                          / (panel.querySelector(".fb-input").offsetWidth
                             * parseFloat(getComputedStyle(panel).zoom)) * 10) / 10 };
      });
      // Under the button, or flipped above it, or held against the top
      // edge when neither fits; and right-aligned to it unless that would
      // run off the screen. Unscaled positions would land at 0.77 of these.
      // A few pixels of slack vertically: the panel is placed while its
      // opening scale is still in flight, so its measured height is a
      // little short of the settled one. Unscaled positions are off by
      // tens of pixels, which this still catches.
      const vertical = Math.abs(fbp.top - (fbp.downBottom + 8)) <= 5
                    || Math.abs(fbp.bottom - (fbp.downTop - 8)) <= 5
                    || Math.abs(fbp.top - 12) <= 5;
      const horizontal = Math.abs(fbp.right - fbp.downRight) <= 2
                      || Math.abs(fbp.right - (fbp.W - 12)) <= 2
                      || Math.abs(fbp.left - 12) <= 2;
      const beside = vertical && horizontal;
      check("300px tall: the what-went-wrong panel takes the step",
            fbp.zoom === "0.77" && fbp.right <= fbp.W && fbp.bottom <= fbp.H && fbp.top >= 0 && beside,
            JSON.stringify(fbp));
      check("300px tall: its comment box is still 16px to iOS", fbp.inputPx === 16, `${fbp.inputPx}px`);
      // 12.8px at full size, times the 0.77 step: shrunk with the panel.
      check("300px tall: and drawn as small as the rest of the panel",
            Math.abs(fbp.drawnPx - 12.8 * 0.77) <= 0.3, `${fbp.drawnPx}px`);
      await shot(p, "v3-short-phone-feedback");
    }
    await p.close();
  }
  {
    const p = await newPage(browser, { width: 1280, height: 300 });
    await p.goto(BASE + "/widget?v=3", { waitUntil: "networkidle2" });
    const cls = await p.evaluate(() => ["fit-1", "fit-2", "fit-3"]
      .filter(c => document.body.classList.contains(c)).join());
    check("a short desktop frame is not scaled", cls === "", JSON.stringify(cls));
    await p.close();
  }

  // ================= the conversation, phone =================
  console.log("\n== conversation, phone ==");
  {
    const p = await newPage(browser, { width: 390, height: 700 }, true, true);
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

    // On a phone too, the notes open as a card in the middle.
    await p.evaluate(() => document.getElementById("messages")
      .scrollTo({ top: document.getElementById("messages").scrollHeight, behavior: "instant" }));
    await sleep(400);
    await p.tap('.ai-notes-row .ai-note[data-sheet="privacy"]');
    await sleep(600);
    const phoneSheet = await p.evaluate(() => {
      const panel = document.querySelector(".sheet-panel");
      const r = panel.getBoundingClientRect();
      const b = panel.querySelector(".sheet-body");
      const wrap = document.querySelector(".sheet");
      const cs = getComputedStyle(wrap);
      return { mid: Math.round(r.top + r.height / 2), H: window.innerHeight,
               scroll: b.scrollHeight - b.clientHeight,
               bg: cs.backgroundColor, glow: cs.boxShadow, blur: cs.backdropFilter };
    });
    check("on a phone the privacy note is a card in the middle",
          Math.abs(phoneSheet.mid - phoneSheet.H / 2) <= 4 && phoneSheet.scroll <= 0,
          JSON.stringify({ mid: phoneSheet.mid, H: phoneSheet.H, scroll: phoneSheet.scroll }));
    check("over a blur that lightens rather than darkens",
          /^rgba\(255, 255, 255/.test(phoneSheet.bg) && /blur/.test(phoneSheet.blur),
          phoneSheet.bg + " / " + phoneSheet.blur);
    check("with a white glow on the window's inside edges",
          /rgba\(255, 255, 255[^)]*\)[^,]*inset/.test(phoneSheet.glow), phoneSheet.glow);
    await shot(p, "v3-privacy-phone");
    await p.tap(".sheet-close");
    await sleep(600);
    await p.tap(".to-end");
    await sleep(1200);
    const end = await p.evaluate(() => {
      const m = document.getElementById("messages");
      return { behind: Math.round(m.scrollHeight - m.scrollTop - m.clientHeight),
               show: document.querySelector(".to-end").classList.contains("show") };
    });
    check("and brings the conversation back to its end", end.behind <= 2 && !end.show,
          JSON.stringify(end));
    // The keyboard: the frame shrinks to what is above it, the composer
    // sits on it and rises, and the conversation keeps its end.
    await p.tap("#query");
    await p.evaluate(() => { window.__kb = 320; window.visualViewport.dispatchEvent(new Event("resize")); });
    await sleep(700);
    const kb = await p.evaluate(() => {
      const m = document.getElementById("messages");
      return { html: document.documentElement.style.height,
               pill: Math.round(document.querySelector(".input-pill").getBoundingClientRect().bottom),
               pad: getComputedStyle(document.querySelector(".chat-input")).paddingBottom,
               behind: Math.round(m.scrollHeight - m.scrollTop - m.clientHeight) };
    });
    check("with the keyboard up, the frame fits above it", kb.html === "380px" && kb.pill <= 380 - 30,
          JSON.stringify(kb));
    check("the composer rises further on a phone", kb.pad === "40px", kb.pad);
    check("and the end of the conversation stays in view", kb.behind <= 2, `${kb.behind}px behind`);
    await shot(p, "v3-keyboard-phone");
    await p.evaluate(() => { window.__kb = 0; window.visualViewport.dispatchEvent(new Event("resize")); });
    await sleep(300);
    const kbDown = await p.evaluate(() => document.documentElement.style.height);
    check("and the frame is given back when it goes", kbDown === "", JSON.stringify(kbDown));

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

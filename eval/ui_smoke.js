/*
 * Look at the page, in a browser, and assert what it does.
 *
 * Every interface bug in this repository has been found by somebody opening
 * it and telling me. Not one was visible in the source: a percentage that
 * resolved against the wrong box, an <svg> falling back to its intrinsic
 * 300x150 because inset cannot size a replaced element, a descendant selector
 * reaching further than intended, scrollHeight refusing to report less than
 * the height already set. Reading CSS finds none of those. Rendering it finds
 * all of them in a second.
 *
 * Needs Chrome, which is already on the machine, and puppeteer-core, which is
 * not a project dependency because nothing in production imports it:
 *
 *     npm install puppeteer-core
 *     python -c "import main; main.app.run(port=8082)" &
 *     node eval/ui_smoke.js
 *
 * CHROME and URL can be overridden by environment variable.
 */
const puppeteer = require("puppeteer-core");

const CHROME = process.env.CHROME ||
  "C:/Program Files/Google/Chrome/Application/chrome.exe";
const URL = process.env.URL || "http://127.0.0.1:8082/widget";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const results = [];
function check(name, pass, detail) {
  results.push({ name, pass, detail });
  console.log(`  ${pass ? "ok  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: "new",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });

  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });

  // /ask is stubbed so this needs no API key and no credit. The point is the
  // interface, not the answer.
  await page.setRequestInterception(true);
  let feedback = null;
  page.on("request", (req) => {
    if (req.url().endsWith("/ask")) {
      const meta = JSON.stringify({
        notice: ["Your query had only a loose match on the school's pages."],
        sources: [{
          url: "https://vrhs.leanderisd.org/campus_information/26-27-bell-schedules",
          label: "26-27 Bell Schedules",
          snippet: "School starts at 8:15 AM.",
        }],
        retrieval: { level: "weak", top: 0.784 },
      });
      return req.respond({
        status: 200, contentType: "text/plain; charset=utf-8",
        body: "School starts at 8:15 AM. Late start days begin at 10:15 AM.:::meta" + meta,
      });
    }
    if (req.url().endsWith("/feedback")) {
      feedback = JSON.parse(req.postData() || "{}");
      return req.respond({
        status: 200, contentType: "application/json",
        body: '{"status":"success"}',
      });
    }
    req.continue();
  });

  await page.goto(URL, { waitUntil: "networkidle2" });

  // === the suggestion panel tracks its content ===========================
  console.log("\nsuggestion panel");
  const measure = async (text) => {
    await page.evaluate(() => {
      const b = document.getElementById("query");
      b.value = "";
      b.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.focus("#query");
    await page.type("#query", text, { delay: 6 });
    await sleep(420);
    return page.evaluate(() => {
      const p = document.querySelector(".suggest");
      if (!p) return { rows: 0, height: 0, content: 0, width: 0 };
      let content = 0;
      p.querySelectorAll(".suggest-item").forEach(
        (r) => { content += r.getBoundingClientRect().height; });
      return {
        rows: p.querySelectorAll(".suggest-item").length,
        height: Math.round(p.getBoundingClientRect().height),
        content: Math.round(content),
        width: Math.round(p.getBoundingClientRect().width),
      };
    });
  };
  const three = await measure("how do i");
  const two = await measure("how do i s");
  const one = await measure("how do i si");
  check("three matches render as rows", three.rows === 3, `${three.rows} rows`);
  check("rows are full width, not circles", three.width > 300, `${three.width}px`);
  check("panel matches its content at 3",
        Math.abs(three.height - three.content) <= 3,
        `${three.height} vs ${three.content}`);
  check("panel shrinks 3 -> 2",
        Math.abs((three.height - two.height) - (three.content - two.content)) <= 3,
        `-${three.height - two.height}px`);
  check("panel shrinks 2 -> 1",
        Math.abs((two.height - one.height) - (two.content - one.content)) <= 3,
        `-${two.height - one.height}px`);

  // === the hover outline is a pill, and sized to the composer ============
  console.log("\nhover outline");
  const snake = await page.evaluate(() => {
    const pill = document.querySelector(".input-pill");
    const svg = document.querySelector(".pill-snake");
    const rect = document.querySelector(".pill-snake rect");
    const p = pill.getBoundingClientRect();
    const s = svg.getBoundingClientRect();
    return {
      pillW: Math.round(p.width), pillH: Math.round(p.height),
      svgW: Math.round(s.width), svgH: Math.round(s.height),
      rx: +rect.getAttribute("rx"), ry: +rect.getAttribute("ry"),
    };
  });
  check("outline box wraps the composer",
        Math.abs(snake.svgW - (snake.pillW + 6)) <= 2 &&
        Math.abs(snake.svgH - (snake.pillH + 6)) <= 2,
        `svg ${snake.svgW}x${snake.svgH}, pill ${snake.pillW}x${snake.pillH}`);
  check("corners are a pill, not an ellipse",
        snake.rx === snake.ry &&
        Math.abs(snake.rx - (snake.pillH + 4) / 2) <= 1,
        `rx=${snake.rx} ry=${snake.ry}`);

  // === the placeholder crossfades rather than snapping ===================
  console.log("\nplaceholder");
  await page.evaluate(() => {
    const b = document.getElementById("query");
    b.value = "";
    b.blur();
    b.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.hover(".input-pill");
  await sleep(1500);
  const cycling = await page.evaluate(() =>
    document.getElementById("query").getAttribute("placeholder"));
  await page.click("#query");
  await sleep(45);
  const mid = await page.evaluate(() => {
    const q = document.getElementById("query");
    return {
      faded: q.classList.contains("ph-fade"),
      colour: getComputedStyle(q, "::placeholder").color,
    };
  });
  await sleep(320);
  const settled = await page.evaluate(() =>
    document.getElementById("query").getAttribute("placeholder"));
  check("a question types itself in on hover",
        cycling && cycling !== "Ask about VRHS...", JSON.stringify(cycling));
  check("it fades rather than snapping", mid.faded, mid.colour);
  check("it settles back to the resting prompt",
        settled === "Ask about VRHS...", JSON.stringify(settled));

  // === the send button ===================================================
  console.log("\ncomposer");
  const send = await page.evaluate(() => {
    const svg = document.querySelector("#submitButton svg");
    return { d: svg.querySelector("path").getAttribute("d"),
             stroke: svg.getAttribute("stroke") };
  });
  check("send icon is a stroked arrow",
        send.stroke === "white" && send.d.startsWith("M12 19V5"), send.d);

  // === one answer, its footer, and the rating it files ===================
  console.log("\nanswer and feedback");
  await page.evaluate(() => {
    const b = document.getElementById("query");
    b.value = "";
    b.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.click("#query");
  await page.type("#query", "when is the bell schedule", { delay: 5 });
  await page.keyboard.press("Enter");
  await sleep(400);
  const gate = await page.$(".disclaimer .accept");
  if (gate) { await gate.click(); await sleep(700); }
  await sleep(900);

  const shown = await page.evaluate(() => ({
    fbButtons: document.querySelectorAll(".fb-btn").length,
    pills: document.querySelectorAll(".source-pill").length,
    cards: document.querySelectorAll(".verify-card").length,
  }));
  check("the FIRST answer carries a rating control",
        shown.fbButtons === 2, `${shown.fbButtons} buttons`);
  check("sources and the caution render",
        shown.pills === 1 && shown.cards === 1,
        `${shown.pills} pill, ${shown.cards} card`);

  const down = await page.$('.fb-btn[data-type="down"]');
  if (down) {
    await down.click();
    await sleep(500);
  }
  check("the rating files the question and the answer",
        !!feedback && feedback.rating === "down" &&
        (feedback.answer || "").includes("8:15"),
        feedback ? JSON.stringify(feedback.answer || "").slice(0, 44) : "no post");

  console.log("\npage errors: " + (errors.length ? errors.join("; ") : "none"));
  const failed = results.filter((r) => !r.pass).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  await browser.close();
  process.exit(failed || errors.length ? 1 : 0);
})();

#!/usr/bin/env node
/*
Capture key Streamlit screens for the AIDEOM-VN report.

This script uses Chrome DevTools Protocol directly so the project does not need
Playwright or Selenium as extra dependencies. It assumes the Streamlit server is
already running locally at http://localhost:8501.
*/

const { spawn } = require("node:child_process");
const fs = require("node:fs/promises");
const path = require("node:path");

const BASE_URL = process.env.AIDEOM_STREAMLIT_URL || "http://localhost:8501";
const OUT_DIR = path.resolve("reports/figures/screenshots");
const DEBUG_PORT = Number(process.env.AIDEOM_CHROME_DEBUG_PORT || 9222);
const CHROME_PATH =
  process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHttp(url, timeoutMs = 15000) {
  const start = Date.now();
  let lastError;
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await sleep(300);
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError?.message || "no response"}`);
}

async function launchChrome() {
  await fs.rm(path.join("/tmp", "aideom-cdp-profile"), { recursive: true, force: true });
  const args = [
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
    `--remote-debugging-port=${DEBUG_PORT}`,
    "--window-size=1600,1200",
    "--user-data-dir=/tmp/aideom-cdp-profile",
    "about:blank",
  ];

  const chrome = spawn(CHROME_PATH, args, {
    stdio: ["ignore", "pipe", "pipe"],
  });
  chrome.stderr.on("data", () => {});
  chrome.stdout.on("data", () => {});

  await waitForHttp(`http://127.0.0.1:${DEBUG_PORT}/json/version`);
  return chrome;
}

async function openTarget() {
  const response = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/new?about:blank`, {
    method: "PUT",
  });
  if (!response.ok) {
    throw new Error(`Cannot create Chrome target: ${response.status} ${response.statusText}`);
  }
  const target = await response.json();
  return target.webSocketDebuggerUrl;
}

class CdpClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
  }

  async connect() {
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    this.ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(message.error.message));
        else resolve(message.result || {});
      } else if (message.method) {
        this.events.push(message);
      }
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    const payload = JSON.stringify({ id, method, params });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(payload);
    });
  }

  close() {
    this.ws.close();
  }
}

async function waitForStreamlit(client, timeoutMs = 25000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const result = await client.send("Runtime.evaluate", {
      expression: `
        (() => {
          const bodyText = document.body ? document.body.innerText : "";
          const hasToolbar = Boolean(document.querySelector('[data-testid="stToolbar"]'));
          const hasApp = Boolean(document.querySelector('.stApp'));
          const loading = bodyText.includes("Please wait") || bodyText.includes("Running...");
          return { ready: hasApp && hasToolbar && !loading && bodyText.length > 250, bodyText };
        })()
      `,
      returnByValue: true,
    });
    if (result.result.value?.ready) return result.result.value.bodyText;
    await sleep(500);
  }
  throw new Error("Streamlit page did not finish rendering in time.");
}

async function goto(client, url) {
  await client.send("Page.navigate", { url });
  await waitForStreamlit(client);
  await sleep(1500);
  await client.send("Runtime.evaluate", { expression: "window.scrollTo(0, 0)" });
}

async function clickByText(client, text, selector = "a, button, [role='tab']", timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const result = await client.send("Runtime.evaluate", {
      expression: `
        (() => {
          const wanted = ${JSON.stringify(text)};
          const nodes = [...document.querySelectorAll(${JSON.stringify(selector)})];
          const node = nodes.find((el) => (el.innerText || el.textContent || "").includes(wanted));
          if (!node) return { clicked: false, count: nodes.length };
          node.scrollIntoView({ block: "center", inline: "center" });
          node.click();
          return { clicked: true, text: node.innerText || node.textContent };
        })()
      `,
      returnByValue: true,
    });
    if (result.result.value?.clicked) {
      await sleep(2500);
      return;
    }
    await sleep(400);
  }
  const debug = await client.send("Runtime.evaluate", {
    expression: `
      (() => {
        const nodes = [...document.querySelectorAll(${JSON.stringify(selector)})]
          .slice(0, 80)
          .map((el) => (el.innerText || el.textContent || "").trim())
          .filter(Boolean);
        return {
          url: location.href,
          title: document.title,
          bodyStart: (document.body?.innerText || "").slice(0, 1200),
          candidates: nodes
        };
      })()
    `,
    returnByValue: true,
  });
  throw new Error(`Cannot find clickable text: ${text}\n${JSON.stringify(debug.result.value, null, 2)}`);
}

async function clickPageLink(client, pageText) {
  await clickByText(client, pageText, "a, button, [role='link']");
  await waitForStreamlit(client);
  await sleep(2500);
  await client.send("Runtime.evaluate", { expression: "window.scrollTo(0, 0)" });
}

async function revealHiddenPages(client) {
  const result = await client.send("Runtime.evaluate", {
    expression: `
      (() => {
        const node = [...document.querySelectorAll("a, button")]
          .find((el) => (el.innerText || el.textContent || "").includes("View 3 more"));
        if (!node) return false;
        node.click();
        return true;
      })()
    `,
    returnByValue: true,
  });
  if (result.result.value) await sleep(1000);
}

async function capture(client, filename) {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await client.send("Runtime.evaluate", { expression: "window.scrollTo(0, 0)" });
  await sleep(500);
  const result = await client.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false,
  });
  const targetPath = path.join(OUT_DIR, filename);
  await fs.writeFile(targetPath, Buffer.from(result.data, "base64"));
  console.log(`saved ${targetPath}`);
}

async function main() {
  await waitForHttp(BASE_URL);
  const chrome = await launchChrome();
  const client = new CdpClient(await openTarget());

  try {
    await client.connect();
    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("Emulation.setDeviceMetricsOverride", {
      width: 1600,
      height: 1200,
      deviceScaleFactor: 1,
      mobile: false,
    });

    await goto(client, BASE_URL);
    await revealHiddenPages(client);
    await clickPageLink(client, "Bai 12");
    await capture(client, "01_home.png");

    await goto(client, BASE_URL);
    await clickPageLink(client, "Bai 1");
    await clickByText(client, "Chạy Bài 1");
    await sleep(3000);
    await capture(client, "02_macro_forecast.png");

    await goto(client, BASE_URL);
    await clickPageLink(client, "Bai 2");
    await clickByText(client, "Chạy SciPy");
    await sleep(3000);
    await capture(client, "03_budget_allocation.png");

    await goto(client, BASE_URL);
    await revealHiddenPages(client);
    await clickPageLink(client, "Bai 12");
    await clickByText(client, "5 kịch bản", "[role='tab'], button");
    await capture(client, "04_scenario_comparison.png");

    await clickByText(client, "Rủi ro", "[role='tab'], button");
    await capture(client, "05_risk_warning.png");
  } finally {
    client.close();
    chrome.kill("SIGTERM");
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});

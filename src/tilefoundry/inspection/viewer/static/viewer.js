/* tilefoundry HIR viewer — client.
 *
 * Fetches DOT from /api/dot and renders it client-side via @hpcc-js/wasm
 * (WASM layout) + innerHTML (no server-side `dot`). Vanilla interaction
 * (no d3-graphviz / jquery plugin needed):
 *   - click a node title → detail panel + highlight its upstream (blue) /
 *     downstream (amber) cone, dim the rest
 *   - click the ▼/▶ toggle → collapse/expand that function region
 *   - search box → highlight nodes whose id/label matches
 *   - wheel = zoom, drag = pan (d3-zoom)
 * Adjacency is built from the rendered SVG: g.node <title> = node id,
 * g.edge <title> = "src:port->dst:port" (ports stripped to bare ids).
 */
(function () {
  "use strict";

  const statusEl = document.getElementById("status");
  const detailEl = document.getElementById("detail");
  const graphEl = document.getElementById("graph");
  const searchEl = document.getElementById("search");
  const hlmodeEl = document.getElementById("hlmode");
  let palette = {};
  let graphviz = null;  // @hpcc-js/wasm
  let index = null;     // {up, down, pairs, nodeEls}
  let lastNode = null;  // last click-selected node id (for mode re-apply)
  const collapsedIds = new Set();

  function setStatus(msg, isErr) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("err", !!isErr);
  }
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ---- detail panel ---------------------------------------------------
  // djb2, mirrors palette.stable_hash so unknown storage colours match the graph.
  function stableHash(s) {
    let h = 5381;
    for (let i = 0; i < s.length; i++) h = (h * 33 + s.charCodeAt(i)) >>> 0;
    return h;
  }
  function storageColor(name) {
    const cls = palette.storage_classes || [];
    const aliases = palette.storage_aliases || {};
    const pool = palette.storage_pool || [];
    const fallback = palette.muted || "#3a4640";
    const canon = aliases[name] || name;
    const ki = cls.indexOf(canon);
    if (ki !== -1) return pool[ki] || fallback;
    const spare = pool.length - cls.length;
    if (spare <= 0) return fallback;
    return pool[cls.length + (stableHash(name) % spare)] || fallback;
  }
  function colorizeType(text) {
    let s = esc(text);
    s = s.replace(/@(\w+)/g, (_m, name) => {
      return '@<span style="color:' + storageColor(name) + ';font-weight:600">' + name + "</span>";
    });
    const dc = palette.dimvar || "#5a3a6e";
    s = s.replace(/\b([A-Z][A-Z0-9_]+)\b/g,
      '<span style="color:' + dc + ';font-weight:600">$1</span>');
    return s;
  }
  function kvTable(rows, colorizeVal) {
    let h = '<table class="kv">';
    for (const [k, v] of rows) {
      h += '<tr><td class="k">' + esc(k) + '</td><td class="v">' +
        (colorizeVal ? colorizeType(v) : esc(v)) + "</td></tr>";
    }
    return h + "</table>";
  }
  function renderDetail(d) {
    let h = "<h2>" + esc(d.kind) + " · " + esc(d.name) + "</h2>";
    if (d.params && d.params.length) h += "<h3>params</h3>" + kvTable(d.params.map((p) => [p.name, p.type]), true);
    if (d.returns && d.returns.length) h += "<h3>returns</h3>" + kvTable(d.returns.map((r) => [r.idx, r.type]), true);
    if (d.attrs && d.attrs.length) h += "<h3>attrs</h3>" + kvTable(d.attrs.map((a) => [a.key, a.value]), false);
    detailEl.innerHTML = h;
  }
  async function showDetail(vid) {
    try {
      const r = await fetch("/api/expr/" + encodeURIComponent(vid));
      if (!r.ok) { detailEl.innerHTML = '<p class="empty">no detail for ' + esc(vid) + "</p>"; return; }
      renderDetail(await r.json());
    } catch (err) { console.error(err); }
  }

  // ---- adjacency index over the rendered SVG --------------------------
  function titleOf(g) {
    const t = g.querySelector("title");
    return t ? t.textContent : (g.getAttribute("data-name") || "");
  }
  function buildIndex() {
    const up = {}, down = {}, pairs = [], nodeEls = {};
    graphEl.querySelectorAll("g.node").forEach((n) => {
      const id = titleOf(n);
      if (id) nodeEls[id] = n;
    });
    graphEl.querySelectorAll("g.edge").forEach((e) => {
      const txt = titleOf(e);
      const a = txt.indexOf("->");
      if (a < 0) return;
      const s = txt.slice(0, a).split(":")[0];
      const d = txt.slice(a + 2).split(":")[0];
      (down[s] = down[s] || []).push(d);
      (up[d] = up[d] || []).push(s);
      pairs.push({ s: s, d: d, el: e });
    });
    return { up: up, down: down, pairs: pairs, nodeEls: nodeEls };
  }
  function reach(start, adj) {
    const seen = new Set();
    const stack = (adj[start] || []).slice();
    while (stack.length) {
      const n = stack.pop();
      if (seen.has(n)) continue;
      seen.add(n);
      (adj[n] || []).forEach((m) => stack.push(m));
    }
    return seen;
  }

  // ---- highlight (3-way: current / upstream / downstream) -------------
  function clearHighlight() {
    graphEl.classList.remove("dimmed");
    graphEl.querySelectorAll(".hl-on, .hl-up, .hl-down, .hl-cur").forEach((el) =>
      el.classList.remove("hl-on", "hl-up", "hl-down", "hl-cur"));
  }
  function highlightConnected(id) {
    if (!index || !index.nodeEls[id]) return;
    clearHighlight();
    lastNode = id;
    const mode = hlmodeEl.value;  // bidirectional | single | upstream | downstream
    const wantUp = mode === "bidirectional" || mode === "upstream";
    const wantDown = mode === "bidirectional" || mode === "downstream";
    const U = wantUp ? reach(id, index.up) : new Set();
    const D = wantDown ? reach(id, index.down) : new Set();
    graphEl.classList.add("dimmed");
    const onNode = (nid, cls) => { const el = index.nodeEls[nid]; if (el) el.classList.add("hl-on", cls); };
    onNode(id, "hl-cur");
    U.forEach((n) => onNode(n, "hl-up"));
    D.forEach((n) => onNode(n, "hl-down"));
    index.pairs.forEach((p) => {
      if (wantUp && (p.d === id || U.has(p.d)) && U.has(p.s)) p.el.classList.add("hl-on", "hl-up");
      else if (wantDown && (p.s === id || D.has(p.s)) && D.has(p.d)) p.el.classList.add("hl-on", "hl-down");
    });
  }

  function runSearch(q) {
    if (!index) return;
    q = (q || "").trim().toLowerCase();
    clearHighlight();
    if (!q) return;
    graphEl.classList.add("dimmed");
    let any = false;
    for (const id in index.nodeEls) {
      const el = index.nodeEls[id];
      if (id.toLowerCase().indexOf(q) !== -1 || (el.textContent || "").toLowerCase().indexOf(q) !== -1) {
        el.classList.add("hl-on", "hl-cur");
        any = true;
      }
    }
    if (!any) graphEl.classList.remove("dimmed");
  }

  // ---- pan / zoom -----------------------------------------------------
  function setupZoom() {
    const svg = d3.select(graphEl).select("svg");
    if (svg.empty()) return;
    const g = svg.select("g");
    const base = g.attr("transform") || "";
    const zoom = d3.zoom()
      .scaleExtent([0.05, 20])
      .on("zoom", (ev) => g.attr("transform", ev.transform.toString() + " " + base));
    svg.call(zoom).on("dblclick.zoom", null);
  }

  // ---- clicks (bound once on the persistent container) ----------------
  function linkTitle(a) {
    return (
      a.getAttribute("xlink:title") ||
      a.getAttributeNS("http://www.w3.org/1999/xlink", "title") ||
      a.getAttribute("title") || ""
    );
  }
  function bindClicks() {
    graphEl.addEventListener("click", (e) => {
      const a = e.target.closest && e.target.closest("a");
      if (!a) { lastNode = null; clearHighlight(); return; }
      e.preventDefault();
      const title = linkTitle(a);
      if (title.indexOf("toggle:") === 0) { toggleRegion(title.slice(7)); return; }
      if (title.indexOf("expr:") === 0) {
        showDetail(title.slice(5));
        const node = a.closest("g.node");
        if (node) highlightConnected(titleOf(node));
      }
    });
    searchEl.addEventListener("input", () => { lastNode = null; runSearch(searchEl.value); });
    // Changing the direction re-applies to the currently selected node.
    hlmodeEl.addEventListener("change", () => { if (lastNode) highlightConnected(lastNode); });
  }

  // ---- collapse / URL-hash state -------------------------------------
  function collapsedCsv() { return Array.from(collapsedIds).join(","); }
  function loadHashState() {
    const m = /(?:^#?|&)collapsed=([^&]*)/.exec(location.hash);
    if (m && m[1]) for (const id of decodeURIComponent(m[1]).split(",")) if (id) collapsedIds.add(id);
  }
  function syncHash() {
    const csv = collapsedCsv();
    history.replaceState(null, "", csv ? "#collapsed=" + encodeURIComponent(csv) : "#");
  }
  function toggleRegion(regionId) {
    if (collapsedIds.has(regionId)) collapsedIds.delete(regionId);
    else collapsedIds.add(regionId);
    syncHash();
    render();
  }

  async function render() {
    try {
      setStatus("rendering…");
      const t0 = performance.now();
      const dot = await (await fetch("/api/dot?collapsed=" + encodeURIComponent(collapsedCsv()))).text();
      const tFetch = performance.now() - t0;
      const t1 = performance.now();
      const svg = graphviz.layout(dot, "svg", "dot");
      const i = svg.indexOf("<svg");
      graphEl.innerHTML = i >= 0 ? svg.slice(i) : svg;
      const tLayout = performance.now() - t1;
      index = buildIndex();
      setupZoom();
      // Re-apply any active selection/search to the freshly rendered SVG.
      if (lastNode && index.nodeEls[lastNode]) highlightConnected(lastNode);
      else if (searchEl.value) runSearch(searchEl.value);
      setStatus("rendered (fetch " + (tFetch | 0) + "ms, layout " + (tLayout | 0) + "ms)");
    } catch (e) {
      setStatus("FATAL: " + e.message, true);
      console.error(e);
    }
  }

  async function loadPalette() {
    try { palette = await (await fetch("/api/palette")).json(); }
    catch (e) { console.warn("palette fetch failed:", e); }
  }

  async function main() {
    await loadPalette();
    bindClicks();
    loadHashState();
    const hpcc = window["@hpcc-js/wasm/graphviz"] || window["@hpcc-js/wasm"];
    graphviz = await hpcc.Graphviz.load();
    render();
  }

  main();
})();

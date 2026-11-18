// Renders data/history.json (SPEC 8.3): per device, GFLOPS per kernel vs commit at the
// largest common size, plus TTFT and decode tokens/s vs commit. All DOM text is set with
// textContent (history data is never interpreted as HTML).
(function () {
  "use strict";

  var LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
  var DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"];
  var KERNEL_ORDER = ["v0", "v1", "v2", "v3", "v4", "hipblas", "v0_fp16", "v4_fp16", "v5", "hipblas_fp16"];

  var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var css = getComputedStyle(document.documentElement);
  function cv(name) { return css.getPropertyValue(name).trim(); }
  var palette = dark ? DARK : LIGHT;

  // Color follows the kernel (stable index), never its rank in a given chart.
  function kernelColor(name, extra) {
    var i = KERNEL_ORDER.indexOf(name);
    if (i < 0) i = KERNEL_ORDER.length + extra;
    return palette[i % palette.length];
  }

  function byTime(a, b) { return a.timestamp < b.timestamp ? -1 : a.timestamp > b.timestamp ? 1 : 0; }
  function shapeKey(g) { return g.M + "x" + g.N + "x" + g.K; }
  function flops(key) { var p = key.split("x").map(Number); return p[0] * p[1] * p[2]; }

  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  function card(parent, title) {
    var c = el("div", null, "card");
    c.appendChild(el("h3", title));
    var wrap = el("div", null, "chart");
    var canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    c.appendChild(wrap);
    parent.appendChild(c);
    return canvas;
  }

  function lineChart(canvas, labels, datasets, yTitle) {
    var ink2 = cv("--ink2"), grid = cv("--grid"), muted = cv("--muted"), surface = cv("--surface");
    datasets.forEach(function (d) {
      d.borderWidth = 2; d.tension = 0; d.spanGaps = true;
      d.pointRadius = 4; d.pointHoverRadius = 6; d.pointBorderColor = surface; d.pointBorderWidth = 2;
      d.pointBackgroundColor = d.borderColor;
    });
    new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: datasets.length > 1, position: "top", align: "start",
                             labels: { color: ink2, usePointStyle: true, boxWidth: 8 } } },
        scales: {
          x: { ticks: { color: muted }, grid: { display: false }, title: { display: true, text: "commit", color: ink2 } },
          y: { beginAtZero: true, ticks: { color: muted }, grid: { color: grid },
               title: { display: true, text: yTitle, color: ink2 } }
        }
      }
    });
  }

  function renderDevice(parent, device, entries) {
    entries.sort(byTime);
    parent.appendChild(el("h2", device));
    var labels = entries.map(function (e) { return String(e.git_sha).slice(0, 7); });

    // GEMM: largest shape present in every entry that has GEMM data (fallback: latest entry's largest).
    var withGemm = entries.filter(function (e) { return e.gemm && e.gemm.length; });
    if (withGemm.length) {
      var common = null;
      withGemm.forEach(function (e) {
        var keys = {};
        e.gemm.forEach(function (g) { keys[shapeKey(g)] = true; });
        if (common === null) common = keys;
        else Object.keys(common).forEach(function (k) { if (!keys[k]) delete common[k]; });
      });
      var cand = Object.keys(common);
      if (!cand.length) {
        cand = withGemm[withGemm.length - 1].gemm.map(shapeKey);
      }
      cand.sort(function (a, b) { return flops(b) - flops(a); });
      var size = cand[0];
      var kernels = [];
      withGemm.forEach(function (e) {
        e.gemm.forEach(function (g) {
          if (shapeKey(g) === size && kernels.indexOf(g.kernel) < 0) kernels.push(g.kernel);
        });
      });
      kernels.sort(function (a, b) {
        var ia = KERNEL_ORDER.indexOf(a), ib = KERNEL_ORDER.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || (a < b ? -1 : 1);
      });
      var extra = 0;
      var sets = kernels.map(function (k) {
        var color = kernelColor(k, extra);
        if (KERNEL_ORDER.indexOf(k) < 0) extra++;
        return {
          label: k, borderColor: color,
          data: entries.map(function (e) {
            var row = (e.gemm || []).filter(function (g) { return g.kernel === k && shapeKey(g) === size; })[0];
            return row ? row.gflops : null;
          })
        };
      });
      lineChart(card(parent, "GEMM GFLOPS at " + size + " per commit"), labels, sets, "GFLOPS");
    } else {
      parent.appendChild(el("p", "No GEMM data for this device yet.", "sub"));
    }

    var withLlm = entries.filter(function (e) { return e.llm; });
    if (withLlm.length) {
      lineChart(card(parent, "Time to first token (lower is better)"), labels,
        [{ label: "ttft_ms", borderColor: palette[0], data: entries.map(function (e) { return e.llm ? e.llm.ttft_ms : null; }) }],
        "ms");
      lineChart(card(parent, "Decode tokens per second (higher is better)"), labels,
        [{ label: "decode_tok_per_s", borderColor: palette[0], data: entries.map(function (e) { return e.llm ? e.llm.decode_tok_per_s : null; }) }],
        "tokens/s");
    }
  }

  var status = document.getElementById("status");
  var root = document.getElementById("devices");

  if (typeof Chart === "undefined") {
    status.textContent = "Chart.js failed to load (CDN unreachable?).";
    return;
  }

  fetch("data/history.json", { cache: "no-store" })
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (history) {
      if (!Array.isArray(history) || !history.length) {
        status.textContent = "History is empty.";
        return;
      }
      var perDevice = {};
      history.forEach(function (e) { (perDevice[e.device] = perDevice[e.device] || []).push(e); });
      status.textContent = history.length + " entries across " + Object.keys(perDevice).length + " device(s).";
      Object.keys(perDevice).sort().forEach(function (d) { renderDevice(root, d, perDevice[d]); });
    })
    .catch(function (err) { status.textContent = "Could not load data/history.json: " + err.message; });
})();

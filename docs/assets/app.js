const state = { data: null, technology: "all", framework: "all", model: "all", selected: null };
const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

async function loadSummary() {
  const response = await fetch("data/summary.json");
  if (!response.ok) throw new Error(`Failed to load summary data (${response.status})`);
  return response.json();
}

function comparisons() { return state.data?.comparisons ?? []; }
function visibleComparisons() {
  return comparisons().filter((item) =>
    (state.technology === "all" || item.technology === state.technology) &&
    (state.framework === "all" || item.framework === state.framework) &&
    (state.model === "all" || item.model === state.model)
  );
}
function optionMarkup(items, label) {
  return `<option value="all">All ${label}</option>${items.map((item) => `<option value="${item}">${item}</option>`).join("")}`;
}
function renderFilters() {
  const technologies = [...new Set(comparisons().map((item) => item.technology))].sort();
  const frameworks = [...new Set(comparisons().map((item) => item.framework))].sort();
  const models = [...new Set(comparisons().map((item) => item.model))].sort((a, b) => Number(a.match(/\d+/)[0]) - Number(b.match(/\d+/)[0]));
  const filters = [
    ["#technology-filter", technologies, "methods", "technology"],
    ["#framework-filter", frameworks, "frameworks", "framework"],
    ["#model-filter", models, "models", "model"]
  ];
  filters.forEach(([selector, items, label, property]) => {
    const target = document.querySelector(selector);
    target.innerHTML = optionMarkup(items, label);
    target.value = state[property];
  });
}
function latency(value) { return `${decimal.format(value)} s`; }
function deltaPercent(ratio) { return `${ratio < 1 ? "↓" : "↑"} ${Math.abs((ratio - 1) * 100).toFixed(1)}%`; }
function renderCards() {
  const target = document.querySelector("#comparison-grid");
  const entries = visibleComparisons();
  target.innerHTML = entries.length ? entries.map((item) => {
    const isSelected = state.selected === item.id;
    const p95Class = item.latency_ratio_p95 > 1 ? "regression" : "improvement";
    return `<article class="comparison-card ${isSelected ? "selected" : ""}" data-id="${item.id}" tabindex="0">
      <div class="card-topline"><span class="tag ${item.technology}">${item.technology}</span><span>${item.framework}</span></div>
      <h3>${item.model}</h3>
      <p class="card-context">${item.hardware} · ${item.prompts} prompts · c${item.concurrency}</p>
      <div class="speedup"><strong>${decimal.format(item.speedup)}×</strong><span>output throughput</span></div>
      <div class="throughput-pair"><div><span>Baseline</span><strong>${fmt.format(item.baseline.output_tokens_per_second)}</strong><small>tok/s</small></div><div><span>${item.technology}</span><strong>${fmt.format(item.speculative.output_tokens_per_second)}</strong><small>tok/s</small></div></div>
      <div class="latency-row"><span>p50 ${latency(item.baseline.p50_latency_s)} → ${latency(item.speculative.p50_latency_s)} <b class="improvement">${deltaPercent(item.latency_ratio_p50)}</b></span><span>p95 ${latency(item.baseline.p95_latency_s)} → ${latency(item.speculative.p95_latency_s)} <b class="${p95Class}">${deltaPercent(item.latency_ratio_p95)}</b></span></div>
    </article>`;
  }).join("") : "<p class=\"empty\">No comparisons match these filters.</p>";
  target.querySelectorAll(".comparison-card").forEach((card) => {
    const choose = () => { state.selected = card.dataset.id; renderCards(); renderChart(); };
    card.addEventListener("click", choose);
    card.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") choose(); });
  });
}
function renderChart() {
  const target = document.querySelector("#speedup-chart");
  const entries = visibleComparisons();
  const max = Math.max(...entries.map((item) => item.speedup), 1.75);
  target.innerHTML = entries.length ? entries.map((item) => `<button class="chart-row ${state.selected === item.id ? "selected" : ""}" data-id="${item.id}" aria-label="Focus ${item.framework} ${item.model} ${item.technology}: ${decimal.format(item.speedup)} times baseline"><span class="chart-label"><b>${item.framework} · ${item.model}</b><small class="method ${item.technology}">${item.technology}</small></span><i><b class="${item.technology}" style="width:${(item.speedup / max) * 100}%"></b></i><strong>${decimal.format(item.speedup)}×</strong></button>`).join("") : "";
  target.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => { state.selected = button.dataset.id; renderCards(); renderChart(); document.querySelector(`[data-id="${state.selected}"]`)?.focus(); }));
}
function renderSummary() {
  const best = [...comparisons()].sort((a, b) => b.speedup - a.speedup)[0];
  document.querySelector("#updated").textContent = state.data.updated;
  document.querySelector("#headline").textContent = state.data.headline;
  document.querySelector("#metric-speedup").textContent = `${decimal.format(best.speedup)}×`;
  document.querySelector("#metric-speedup-label").textContent = `${best.framework} · ${best.model} · ${best.technology}`;
  document.querySelector("#metric-comparisons").textContent = comparisons().length;
  document.querySelector("#metric-frameworks").textContent = new Set(comparisons().map((item) => item.framework)).size;
  document.querySelector("#metric-runs").textContent = comparisons().length * 2;
}
function bindEvents() {
  [["#technology-filter", "technology"], ["#framework-filter", "framework"], ["#model-filter", "model"]].forEach(([selector, property]) => {
    document.querySelector(selector).addEventListener("change", (event) => { state[property] = event.target.value; state.selected = null; renderCards(); renderChart(); });
  });
}
loadSummary().then((data) => { state.data = data; renderSummary(); renderFilters(); bindEvents(); renderCards(); renderChart(); }).catch((error) => { document.querySelector("#headline").textContent = error.message; });

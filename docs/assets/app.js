let state = {
  data: null,
  framework: "all",
  mode: "all",
  query: ""
};

const fmt = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2
});

async function loadSummary() {
  const response = await fetch("data/summary.json");
  if (!response.ok) {
    throw new Error(`Failed to load summary.json: ${response.status}`);
  }
  return response.json();
}

function asNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function displayNumber(value, suffix = "") {
  const number = asNumber(value);
  return number === null ? "n/a" : `${fmt.format(number)}${suffix}`;
}

function runs() {
  return state.data?.runs ?? [];
}

function filteredRuns() {
  const query = state.query.trim().toLowerCase();
  return runs().filter((run) => {
    const frameworkMatch = state.framework === "all" || run.framework === state.framework;
    const modeMatch = state.mode === "all" || run.mode === state.mode;
    const haystack = [
      run.label,
      run.mode,
      run.framework,
      run.profile,
      run.model,
      run.draft_model,
      run.hardware,
      run.status,
      run.notes
    ].join(" ").toLowerCase();
    return frameworkMatch && modeMatch && (!query || haystack.includes(query));
  });
}

function bestRun() {
  return [...runs()]
    .filter((run) => asNumber(run.output_tokens_per_second) !== null)
    .sort((a, b) => b.output_tokens_per_second - a.output_tokens_per_second)[0];
}

function bestSpeedupRun() {
  return [...runs()]
    .filter((run) => run.mode === "dflash" && asNumber(run.speedup) !== null)
    .sort((a, b) => b.speedup - a.speedup)[0];
}

function renderMetrics() {
  const best = bestRun();
  const bestSpeedup = bestSpeedupRun();
  const okCount = runs().filter((run) => run.status === "ok" || run.status === "success").length;
  const failedCount = runs().filter((run) => run.status === "failed" || run.status === "error").length;

  document.querySelector("#metric-speedup").textContent = bestSpeedup
    ? `${displayNumber(bestSpeedup.speedup)}x`
    : "n/a";
  document.querySelector("#metric-speedup-label").textContent = bestSpeedup
    ? bestSpeedup.label
    : "Best DFlash speedup";
  document.querySelector("#metric-best").textContent = best
    ? displayNumber(best.output_tokens_per_second, " tok/s")
    : "n/a";
  document.querySelector("#metric-best-label").textContent = best
    ? best.label
    : "No completed benchmark yet";
  document.querySelector("#metric-runs").textContent = runs().length;
  document.querySelector("#metric-ok").textContent = `${okCount}/${failedCount}`;
}

function renderFilters() {
  const frameworkSelect = document.querySelector("#framework-filter");
  const modeSelect = document.querySelector("#mode-filter");

  const frameworks = [...new Set(runs().map((run) => run.framework).filter(Boolean))].sort();
  const modes = [...new Set(runs().map((run) => run.mode).filter(Boolean))].sort();

  frameworkSelect.innerHTML = `<option value="all">All frameworks</option>${frameworks
    .map((framework) => `<option value="${framework}">${framework}</option>`)
    .join("")}`;
  modeSelect.innerHTML = `<option value="all">All modes</option>${modes
    .map((mode) => `<option value="${mode}">${mode}</option>`)
    .join("")}`;

  frameworkSelect.value = state.framework;
  modeSelect.value = state.mode;
}

function renderChart() {
  const chart = document.querySelector("#throughput-chart");
  const chartRuns = filteredRuns()
    .filter((run) => asNumber(run.output_tokens_per_second) !== null)
    .sort((a, b) => b.output_tokens_per_second - a.output_tokens_per_second);
  const max = Math.max(...chartRuns.map((run) => run.output_tokens_per_second), 1);

  chart.innerHTML = chartRuns.length
    ? chartRuns
        .map((run) => {
          const width = Math.max(1, (run.output_tokens_per_second / max) * 100);
          return `
            <div class="bar-row">
              <div class="bar-label">${run.label}</div>
              <div class="bar-track" title="${run.notes}">
                <div class="bar-fill ${run.mode}" style="width: ${width}%"></div>
              </div>
              <div class="bar-value">${displayNumber(run.output_tokens_per_second)}</div>
            </div>
          `;
        })
        .join("")
    : `<p class="small">No numeric throughput in the current filter.</p>`;
}

function statusClass(status) {
  if (status === "ok") return "success";
  if (status === "error") return "failed";
  return status || "planned";
}

function renderRuns() {
  const list = document.querySelector("#run-list");
  const visibleRuns = filteredRuns();

  list.innerHTML = visibleRuns.length
    ? visibleRuns
        .map((run) => {
          const draft = run.draft_model ? ` Draft: ${run.draft_model}.` : "";
          const p50 = displayNumber(run.p50_latency_s, "s p50");
          const p95 = displayNumber(run.p95_latency_s, "s p95");
          const speedup = asNumber(run.speedup) === null ? "" : ` · ${displayNumber(run.speedup)}x speedup`;
          const shape = [
            run.hardware,
            run.prompts ? `${run.prompts} prompts` : null,
            run.max_new_tokens ? `${run.max_new_tokens} max tokens` : null,
            run.concurrency ? `c${run.concurrency}` : null
          ].filter(Boolean).join(" · ");
          const wandb = run.wandb_url
            ? `<a class="inline-link" href="${run.wandb_url}">W&B run</a>`
            : "";
          return `
            <article class="run-row">
              <div class="run-head">
                <div>
                  <h3 class="run-title">${run.label}</h3>
                  <p class="run-meta">${run.framework ?? "unknown"} · ${run.profile ?? "profile n/a"} · ${run.model}.${draft}</p>
                </div>
                <span class="pill ${statusClass(run.status)}">${run.status}</span>
              </div>
              <p class="run-notes">${run.notes}</p>
              <p class="run-meta">
                ${displayNumber(run.output_tokens_per_second, " tok/s")}${speedup} ·
                ${p50} · ${p95} ·
                ${displayNumber(run.total_output_tokens, " tokens")}
              </p>
              <p class="run-meta">${shape}${wandb ? ` · ${wandb}` : ""}</p>
            </article>
          `;
        })
        .join("")
    : `<p class="small">No runs match the current filters.</p>`;
}

function renderTimeline() {
  const target = document.querySelector("#timeline");
  target.innerHTML = (state.data?.milestones ?? [])
    .map((item) => `
      <div class="timeline-item">
        <span class="pill ${item.state}">${item.state}</span>
        <div>
          <h3>${item.title}</h3>
          <p>${item.body}</p>
        </div>
      </div>
    `)
    .join("");
}

function renderDecisions() {
  const target = document.querySelector("#decisions");
  target.innerHTML = (state.data?.decisions ?? [])
    .map((decision) => `<li>${decision}</li>`)
    .join("");
}

function renderAll() {
  document.querySelector("#updated").textContent = state.data?.updated ?? "unknown";
  document.querySelector("#headline").textContent = state.data?.headline ?? "";
  renderMetrics();
  renderFilters();
  renderChart();
  renderRuns();
  renderTimeline();
  renderDecisions();
}

function normalizeBenchmarkResult(raw, index = 0) {
  const config = raw.config ?? {};
  const summary = raw.summary ?? {};
  const requests = raw.requests ?? [];
  const failedRequests = requests.filter((request) => request.status && request.status !== "ok").length;
  const mode = config.mode ?? raw.mode ?? "imported";
  const status = failedRequests > 0 ? "error" : "ok";
  const suffix = index ? ` ${index + 1}` : "";

  return {
    id: `import-${Date.now()}-${index}`,
    label: raw.label ?? `${mode}${suffix}`,
    mode,
    framework: config.framework ?? raw.framework ?? "imported",
    profile: config.profile ?? raw.profile ?? "unknown",
    model: config.model ?? raw.model ?? "unknown",
    draft_model: config.draft_model ?? config.draftModel ?? raw.draft_model ?? null,
    hardware: summary.gpu_name ?? raw.hardware ?? "imported",
    status,
    prompts: summary.num_prompts ?? null,
    max_new_tokens: config.max_new_tokens ?? null,
    concurrency: config.concurrency ?? null,
    output_tokens_per_second: summary.output_tokens_per_second ?? null,
    p50_latency_s: summary.p50_latency_s ?? null,
    p95_latency_s: summary.p95_latency_s ?? null,
    total_output_tokens: summary.total_output_tokens ?? null,
    notes: failedRequests
      ? `${failedRequests} request(s) failed.`
      : `Imported ${requests.length || "no"} request row(s) from bench_openai_server.py output.`
  };
}

function importJson() {
  const input = document.querySelector("#json-input");
  const message = document.querySelector("#import-message");
  try {
    const parsed = JSON.parse(input.value);
    const rawItems = Array.isArray(parsed) ? parsed : [parsed];
    const imported = rawItems.map(normalizeBenchmarkResult);
    state.data.runs = [...imported, ...runs()];
    message.textContent = `Imported ${imported.length} run${imported.length === 1 ? "" : "s"}.`;
    renderAll();
  } catch (error) {
    message.textContent = `Import failed: ${error.message}`;
  }
}

function bindEvents() {
  document.querySelector("#framework-filter").addEventListener("change", (event) => {
    state.framework = event.target.value;
    renderChart();
    renderRuns();
  });
  document.querySelector("#mode-filter").addEventListener("change", (event) => {
    state.mode = event.target.value;
    renderChart();
    renderRuns();
  });
  document.querySelector("#search").addEventListener("input", (event) => {
    state.query = event.target.value;
    renderChart();
    renderRuns();
  });
  document.querySelector("#import-json").addEventListener("click", importJson);
  document.querySelector("#clear-import").addEventListener("click", () => {
    document.querySelector("#json-input").value = "";
    document.querySelector("#import-message").textContent = "";
  });
}

loadSummary()
  .then((data) => {
    state.data = data;
    bindEvents();
    renderAll();
  })
  .catch((error) => {
    document.querySelector("#headline").textContent = error.message;
  });

const state = {
  data: null,
  technology: "all",
  framework: "all",
  model: "all",
  selected: null
};

const integerFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0
});

const decimalFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
});

const selectors = {
  headline: "#headline",
  updated: "#updated",
  speedupMetric: "#metric-speedup",
  speedupLabel: "#metric-speedup-label",
  comparisonsMetric: "#metric-comparisons",
  frameworksMetric: "#metric-frameworks",
  runsMetric: "#metric-runs",
  technologyFilter: "#technology-filter",
  frameworkFilter: "#framework-filter",
  modelFilter: "#model-filter",
  comparisonGrid: "#comparison-grid",
  speedupChart: "#speedup-chart"
};

async function loadSummary() {
  const response = await fetch("data/summary.json", {
    headers: {
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    throw new Error(
      `Unable to load benchmark data. Server returned ${response.status}.`
    );
  }

  return response.json();
}

function comparisons() {
  return Array.isArray(state.data?.comparisons)
    ? state.data.comparisons
    : [];
}

function visibleComparisons() {
  return comparisons().filter((item) => {
    const technologyMatches =
      state.technology === "all" ||
      item.technology === state.technology;

    const frameworkMatches =
      state.framework === "all" ||
      item.framework === state.framework;

    const modelMatches =
      state.model === "all" ||
      item.model === state.model;

    return (
      technologyMatches &&
      frameworkMatches &&
      modelMatches
    );
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeClassName(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-");
}

function getModelSize(modelName) {
  const match = String(modelName).match(
    /(\d+(?:\.\d+)?)\s*[bB]/
  );

  return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
}

function sortModels(a, b) {
  const sizeDifference = getModelSize(a) - getModelSize(b);

  if (sizeDifference !== 0) {
    return sizeDifference;
  }

  return String(a).localeCompare(String(b));
}

function formatInteger(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? integerFormatter.format(numericValue)
    : "—";
}

function formatDecimal(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? decimalFormatter.format(numericValue)
    : "—";
}

function formatLatency(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue)
    ? `${decimalFormatter.format(numericValue)} s`
    : "—";
}

function formatDelta(ratio) {
  const numericRatio = Number(ratio);

  if (!Number.isFinite(numericRatio)) {
    return "—";
  }

  const percentage = Math.abs(
    (numericRatio - 1) * 100
  ).toFixed(1);

  if (numericRatio < 1) {
    return `↓ ${percentage}%`;
  }

  if (numericRatio > 1) {
    return `↑ ${percentage}%`;
  }

  return "No change";
}

function optionMarkup(items, pluralLabel) {
  const options = items
    .map((item) => {
      const escapedItem = escapeHtml(item);

      return `
        <option value="${escapedItem}">
          ${escapedItem}
        </option>
      `;
    })
    .join("");

  return `
    <option value="all">All ${escapeHtml(pluralLabel)}</option>
    ${options}
  `;
}

function renderFilters() {
  const technologies = [
    ...new Set(
      comparisons().map((item) => item.technology)
    )
  ]
    .filter(Boolean)
    .sort();

  const frameworks = [
    ...new Set(
      comparisons().map((item) => item.framework)
    )
  ]
    .filter(Boolean)
    .sort();

  const models = [
    ...new Set(
      comparisons().map((item) => item.model)
    )
  ]
    .filter(Boolean)
    .sort(sortModels);

  const filters = [
    {
      selector: selectors.technologyFilter,
      items: technologies,
      label: "methods",
      property: "technology"
    },
    {
      selector: selectors.frameworkFilter,
      items: frameworks,
      label: "frameworks",
      property: "framework"
    },
    {
      selector: selectors.modelFilter,
      items: models,
      label: "models",
      property: "model"
    }
  ];

  filters.forEach(
    ({ selector, items, label, property }) => {
      const target = document.querySelector(selector);

      if (!target) {
        return;
      }

      target.innerHTML = optionMarkup(items, label);
      target.value = state[property];
    }
  );
}

function renderCards() {
  const target = document.querySelector(
    selectors.comparisonGrid
  );

  if (!target) {
    return;
  }

  const entries = visibleComparisons();

  if (!entries.length) {
    target.innerHTML = `
      <div class="empty-state" role="status">
        <h3>No matching comparisons</h3>
        <p>
          Adjust the method, framework, or model filters
          to view additional benchmark results.
        </p>
      </div>
    `;

    return;
  }

  target.innerHTML = entries
    .map((item) => {
      const itemId = escapeHtml(item.id);
      const technology = escapeHtml(item.technology);
      const framework = escapeHtml(item.framework);
      const model = escapeHtml(item.model);
      const hardware = escapeHtml(item.hardware);
      const technologyClass = normalizeClassName(
        item.technology
      );

      const isSelected = state.selected === item.id;

      const p50Class =
        Number(item.latency_ratio_p50) > 1
          ? "regression"
          : "improvement";

      const p95Class =
        Number(item.latency_ratio_p95) > 1
          ? "regression"
          : "improvement";

      const baselineThroughput =
        item.baseline?.output_tokens_per_second;

      const speculativeThroughput =
        item.speculative?.output_tokens_per_second;

      return `
        <article
          class="comparison-card ${
            isSelected ? "selected" : ""
          }"
          data-card-id="${itemId}"
          tabindex="0"
          role="button"
          aria-pressed="${isSelected}"
          aria-label="Select ${framework} ${model} ${technology} comparison"
        >
          <div class="card-topline">
            <span class="tag ${technologyClass}">
              ${technology}
            </span>

            <span class="framework-label">
              ${framework}
            </span>
          </div>

          <h3>${model}</h3>

          <p class="card-context">
            ${hardware}
            <span aria-hidden="true">•</span>
            ${formatInteger(item.prompts)} prompts
            <span aria-hidden="true">•</span>
            concurrency ${formatInteger(item.concurrency)}
          </p>

          <div class="speedup">
            <strong>${formatDecimal(item.speedup)}×</strong>
            <span>Baseline throughput</span>
          </div>

          <div class="throughput-pair">
            <div>
              <span>Baseline</span>
              <strong>
                ${formatInteger(baselineThroughput)}
              </strong>
              <small>output tok/s</small>
            </div>

            <div>
              <span>${technology}</span>
              <strong>
                ${formatInteger(speculativeThroughput)}
              </strong>
              <small>output tok/s</small>
            </div>
          </div>

          <div class="latency-row">
            <div>
              <span class="latency-label">Median latency</span>

              <span class="latency-values">
                ${formatLatency(item.baseline?.p50_latency_s)}
                <span aria-hidden="true">→</span>
                ${formatLatency(item.speculative?.p50_latency_s)}
              </span>

              <b class="${p50Class}">
                ${formatDelta(item.latency_ratio_p50)}
              </b>
            </div>

            <div>
              <span class="latency-label">p95 latency</span>

              <span class="latency-values">
                ${formatLatency(item.baseline?.p95_latency_s)}
                <span aria-hidden="true">→</span>
                ${formatLatency(item.speculative?.p95_latency_s)}
              </span>

              <b class="${p95Class}">
                ${formatDelta(item.latency_ratio_p95)}
              </b>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  target
    .querySelectorAll(".comparison-card")
    .forEach((card) => {
      const chooseCard = () => {
        state.selected = card.dataset.cardId;
        renderCards();
        renderChart();
      };

      card.addEventListener("click", chooseCard);

      card.addEventListener("keydown", (event) => {
        if (
          event.key === "Enter" ||
          event.key === " "
        ) {
          event.preventDefault();
          chooseCard();
        }
      });
    });
}

function renderChart() {
  const target = document.querySelector(
    selectors.speedupChart
  );

  if (!target) {
    return;
  }

  const entries = visibleComparisons();

  if (!entries.length) {
    target.innerHTML = "";
    return;
  }

  const speedups = entries
    .map((item) => Number(item.speedup))
    .filter(Number.isFinite);

  const maximumSpeedup = Math.max(
    ...speedups,
    1.25
  );

  target.innerHTML = entries
    .map((item) => {
      const itemId = escapeHtml(item.id);
      const technology = escapeHtml(item.technology);
      const framework = escapeHtml(item.framework);
      const model = escapeHtml(item.model);

      const technologyClass = normalizeClassName(
        item.technology
      );

      const numericSpeedup = Number(item.speedup);
      const safeSpeedup = Number.isFinite(numericSpeedup)
        ? numericSpeedup
        : 0;

      const width = Math.min(
        (safeSpeedup / maximumSpeedup) * 100,
        100
      );

      const isSelected = state.selected === item.id;

      return `
        <button
          type="button"
          class="chart-row ${
            isSelected ? "selected" : ""
          }"
          data-chart-id="${itemId}"
          aria-pressed="${isSelected}"
          aria-label="Select ${framework} ${model} ${technology}, ${formatDecimal(
            item.speedup
          )} times baseline throughput"
        >
          <span class="chart-label">
            <b>${framework} · ${model}</b>

            <small class="method ${technologyClass}">
              ${technology}
            </small>
          </span>

          <span class="chart-track" aria-hidden="true">
            <span
              class="chart-fill ${technologyClass}"
              style="width: ${width}%"
            ></span>
          </span>

          <strong>
            ${formatDecimal(item.speedup)}×
          </strong>
        </button>
      `;
    })
    .join("");

  target
    .querySelectorAll(".chart-row")
    .forEach((button) => {
      button.addEventListener("click", () => {
        state.selected = button.dataset.chartId;

        renderCards();
        renderChart();

        const selectedCard = document.querySelector(
          `[data-card-id="${CSS.escape(state.selected)}"]`
        );

        selectedCard?.scrollIntoView({
          behavior: "smooth",
          block: "center"
        });

        selectedCard?.focus({
          preventScroll: true
        });
      });
    });
}

function renderSummary() {
  const entries = comparisons();

  const best = [...entries]
    .filter((item) =>
      Number.isFinite(Number(item.speedup))
    )
    .sort(
      (a, b) =>
        Number(b.speedup) - Number(a.speedup)
    )[0];

  const updatedTarget = document.querySelector(
    selectors.updated
  );

  const headlineTarget = document.querySelector(
    selectors.headline
  );

  if (updatedTarget) {
    updatedTarget.textContent =
      state.data?.updated ?? "Unavailable";
  }

  if (headlineTarget) {
    headlineTarget.textContent =
      state.data?.headline ??
      "Benchmark results are ready to explore.";
  }

  const speedupTarget = document.querySelector(
    selectors.speedupMetric
  );

  const speedupLabelTarget = document.querySelector(
    selectors.speedupLabel
  );

  if (best) {
    if (speedupTarget) {
      speedupTarget.textContent =
        `${formatDecimal(best.speedup)}×`;
    }

    if (speedupLabelTarget) {
      speedupLabelTarget.textContent =
        `${best.framework} · ${best.model} · ${best.technology}`;
    }
  } else {
    if (speedupTarget) {
      speedupTarget.textContent = "—";
    }

    if (speedupLabelTarget) {
      speedupLabelTarget.textContent =
        "No comparison data available";
    }
  }

  const comparisonsTarget = document.querySelector(
    selectors.comparisonsMetric
  );

  const frameworksTarget = document.querySelector(
    selectors.frameworksMetric
  );

  const runsTarget = document.querySelector(
    selectors.runsMetric
  );

  if (comparisonsTarget) {
    comparisonsTarget.textContent = formatInteger(
      entries.length
    );
  }

  if (frameworksTarget) {
    frameworksTarget.textContent = formatInteger(
      new Set(
        entries.map((item) => item.framework)
      ).size
    );
  }

  if (runsTarget) {
    runsTarget.textContent = formatInteger(
      entries.length * 2
    );
  }
}

function bindEvents() {
  const filters = [
    {
      selector: selectors.technologyFilter,
      property: "technology"
    },
    {
      selector: selectors.frameworkFilter,
      property: "framework"
    },
    {
      selector: selectors.modelFilter,
      property: "model"
    }
  ];

  filters.forEach(({ selector, property }) => {
    const element = document.querySelector(selector);

    if (!element) {
      return;
    }

    element.addEventListener("change", (event) => {
      state[property] = event.target.value;
      state.selected = null;

      renderCards();
      renderChart();
    });
  });
}

function showLoadingState() {
  const comparisonTarget = document.querySelector(
    selectors.comparisonGrid
  );

  const chartTarget = document.querySelector(
    selectors.speedupChart
  );

  if (comparisonTarget) {
    comparisonTarget.innerHTML = `
      <div class="loading-state" role="status">
        <span class="loading-indicator" aria-hidden="true"></span>
        <p>Loading benchmark comparisons…</p>
      </div>
    `;
  }

  if (chartTarget) {
    chartTarget.innerHTML = "";
  }
}

function showErrorState(error) {
  const message =
    error instanceof Error
      ? error.message
      : "An unexpected error occurred.";

  const headlineTarget = document.querySelector(
    selectors.headline
  );

  const comparisonTarget = document.querySelector(
    selectors.comparisonGrid
  );

  if (headlineTarget) {
    headlineTarget.textContent =
      "Benchmark data could not be loaded.";
  }

  if (comparisonTarget) {
    comparisonTarget.innerHTML = `
      <div class="error-state" role="alert">
        <h3>Unable to display results</h3>
        <p>${escapeHtml(message)}</p>
        <button type="button" id="retry-load">
          Retry
        </button>
      </div>
    `;

    comparisonTarget
      .querySelector("#retry-load")
      ?.addEventListener("click", initialize);
  }
}

async function initialize() {
  showLoadingState();

  try {
    state.data = await loadSummary();
    state.selected = null;

    renderSummary();
    renderFilters();
    bindEvents();
    renderCards();
    renderChart();
  } catch (error) {
    console.error("Benchmark initialization failed:", error);
    showErrorState(error);
  }
}

document.addEventListener("DOMContentLoaded", initialize);

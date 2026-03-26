import { useState, useRef, useCallback, useEffect } from "react";

// ── CONFIG ───────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:5000";

const SCENARIO_META = {
  A: {
    label: "Big-Bang + Chatbot",
    color: "#C0392B",
    bg: "#C0392B0F",
    border: "#C0392B50",
  },
  B: {
    label: "Phased + Human + Training",
    color: "#1A6FB5",
    bg: "#1A6FB50F",
    border: "#1A6FB550",
  },
  C: {
    label: "Pilot + Strong Management",
    color: "#15845D",
    bg: "#15845D0F",
    border: "#15845D50",
  },
};

const PERSONA_COLORS = {
  "Tech Pioneer": "#15845D",
  "Power User": "#1A6FB5",
  "Pragmatic Adopter": "#B07A10",
  "Reluctant User": "#C0392B",
  "Remote-First Worker": "#6B5CE7",
};

const PERSONA_SHORT = {
  "Tech Pioneer": "Pioneer",
  "Power User": "Power",
  "Pragmatic Adopter": "Pragmatic",
  "Reluctant User": "Reluctant",
  "Remote-First Worker": "Remote",
};

const DEFAULT_SCENARIO_CONFIGS = {
  A: {
    tool_complexity: 0.65,
    training_intensity: 0.1,
    support_model: "chatbot",
    manager_signal: 0.4,
  },
  B: {
    tool_complexity: 0.65,
    training_intensity: 0.7,
    support_model: "human",
    manager_signal: 0.6,
  },
  C: {
    tool_complexity: 0.65,
    training_intensity: 0.45,
    support_model: "hybrid",
    manager_signal: 0.8,
  },
};

// ── CSS (injected once) ───────────────────────────────────────────────────────
const GLOBAL_CSS = `
  :root {
    --bg-page:       #F4F5F7;
    --bg-card:       #FFFFFF;
    --bg-input:      #F9FAFB;
    --border-light:  #E2E6EA;
    --border-mid:    #CBD2DA;
    --text-primary:  #1A202C;
    --text-secondary:#52606D;
    --text-muted:    #8896A5;
    --accent-green:  #15845D;
    --accent-blue:   #1A6FB5;
    --accent-red:    #C0392B;
    --accent-amber:  #B07A10;
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:     0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     14px;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg-page); color: var(--text-primary); margin: 0; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
  input[type=range] {
    -webkit-appearance: none; height: 4px;
    background: var(--border-light); border-radius: 2px; outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: var(--accent-green);
    cursor: pointer; border: 2px solid #fff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.18);
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
  }
  .btn-ghost {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm);
    padding: 6px 14px;
    cursor: pointer;
    font-size: 12px;
    color: var(--text-secondary);
    transition: all 0.12s;
  }
  .btn-ghost:hover { border-color: var(--border-mid); color: var(--text-primary); }
`;

function useGlobalCss() {
  useEffect(() => {
    const id = "dtw-global-css";
    if (!document.getElementById(id)) {
      const s = document.createElement("style");
      s.id = id;
      s.textContent = GLOBAL_CSS;
      document.head.appendChild(s);
    }
  }, []);
}

// ── PLOTLY LOADER ─────────────────────────────────────────────────────────────
let plotlyLoaded = false;
let plotlyCallbacks = [];
function loadPlotly(cb) {
  if (plotlyLoaded) {
    cb();
    return;
  }
  plotlyCallbacks.push(cb);
  if (plotlyCallbacks.length > 1) return;
  const s = document.createElement("script");
  s.src = "https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.0/plotly.min.js";
  s.onload = () => {
    plotlyLoaded = true;
    plotlyCallbacks.forEach((f) => f());
    plotlyCallbacks = [];
  };
  document.head.appendChild(s);
}

function usePlotly() {
  const [ready, setReady] = useState(plotlyLoaded);
  useEffect(() => {
    if (!plotlyLoaded) loadPlotly(() => setReady(true));
  }, []);
  return ready;
}

// ── CHART HELPERS ─────────────────────────────────────────────────────────────

const LIGHT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {
    family: "'IBM Plex Sans', system-ui, sans-serif",
    color: "#52606D",
    size: 11,
  },
};

const GRID_COLOR = "rgba(0,0,0,0.07)";
const ZERO_COLOR = "rgba(0,0,0,0.18)";

// ── CHART COMPONENTS ─────────────────────────────────────────────────────────

function AdoptionChart({ data, activeScenarios }) {
  const ref = useRef();
  const plotlyReady = usePlotly();

  useEffect(() => {
    if (!plotlyReady || !ref.current || !data) return;
    const weeks = Array.from({ length: 52 }, (_, i) => i);
    const traces = activeScenarios
      .map((key) => {
        const s = data.scenarios[key];
        const m = SCENARIO_META[key];
        return [
          {
            x: weeks,
            y: s.weekly.adoption_p05.map((v) => +(v * 100).toFixed(1)),
            fill: "tozeroy",
            fillcolor: m.color + "10",
            line: { color: "transparent" },
            showlegend: false,
            hoverinfo: "skip",
            type: "scatter",
            mode: "lines",
          },
          {
            x: weeks,
            y: s.weekly.adoption_p95.map((v) => +(v * 100).toFixed(1)),
            fill: "tonexty",
            fillcolor: m.color + "18",
            line: { color: "transparent" },
            showlegend: false,
            hoverinfo: "skip",
            type: "scatter",
            mode: "lines",
          },
          {
            x: weeks,
            y: s.weekly.adoption_mean.map((v) => +(v * 100).toFixed(1)),
            name: `${key}: ${s.label.split(" ")[0]}`,
            line: { color: m.color, width: 2.5 },
            mode: "lines",
            type: "scatter",
            hovertemplate: `<b>${s.label}</b><br>Week %{x}<br>Adoption: %{y:.1f}%<extra></extra>`,
          },
        ];
      })
      .flat();

    window.Plotly.newPlot(
      ref.current,
      traces,
      {
        ...LIGHT_LAYOUT,
        margin: { t: 10, r: 14, b: 44, l: 50 },
        xaxis: {
          title: { text: "Week", font: { size: 11, color: "#8896A5" } },
          gridcolor: GRID_COLOR,
          tickfont: { size: 11, color: "#52606D" },
          range: [0, 51],
          linecolor: GRID_COLOR,
          zerolinecolor: GRID_COLOR,
        },
        yaxis: {
          title: { text: "% Adopted", font: { size: 11, color: "#8896A5" } },
          gridcolor: GRID_COLOR,
          tickfont: { size: 11, color: "#52606D" },
          ticksuffix: "%",
          linecolor: GRID_COLOR,
        },
        legend: {
          x: 0.02,
          y: 0.98,
          bgcolor: "rgba(255,255,255,0.85)",
          bordercolor: GRID_COLOR,
          borderwidth: 1,
          font: { size: 11, color: "#1A202C" },
        },
        hovermode: "x unified",
      },
      { responsive: true, displayModeBar: false },
    );
    return () => window.Plotly.purge(ref.current);
  }, [plotlyReady, data, activeScenarios]);

  return <div ref={ref} style={{ height: 240 }} />;
}

function Persona3DChart({ data, activeScenarios }) {
  const ref = useRef();
  const plotlyReady = usePlotly();

  useEffect(() => {
    if (!plotlyReady || !ref.current || !data) return;
    const personas = Object.keys(PERSONA_COLORS);
    const traces = activeScenarios.map((key) => {
      const s = data.scenarios[key];
      return {
        type: "scatter3d",
        mode: "markers+lines",
        x: personas.map((p) => s.persona_w52[p] * 100),
        y: personas.map((p) => s.config.training_intensity * 100),
        z: personas.map((p) => s.config.manager_signal * 100),
        text: personas.map(
          (p) =>
            `${PERSONA_SHORT[p]}<br>Adoption: ${(s.persona_w52[p] * 100).toFixed(0)}%`,
        ),
        marker: {
          size: personas.map((p) => 8 + s.persona_w52[p] * 14),
          color: personas.map((p) => PERSONA_COLORS[p]),
          opacity: 0.9,
          line: { color: "#fff", width: 1.5 },
        },
        line: { color: SCENARIO_META[key].color, width: 2, dash: "dot" },
        name: `${key}: ${s.label.split(" ")[0]}`,
        hovertemplate: "<b>%{text}</b><extra></extra>",
      };
    });

    const axStyle = {
      gridcolor: GRID_COLOR,
      backgroundcolor: "rgba(244,245,247,0.5)",
      color: "#52606D",
    };
    window.Plotly.newPlot(
      ref.current,
      traces,
      {
        ...LIGHT_LAYOUT,
        margin: { t: 10, r: 0, b: 0, l: 0 },
        scene: {
          xaxis: { title: "Adoption %", ticksuffix: "%", ...axStyle },
          yaxis: { title: "Training", ticksuffix: "%", ...axStyle },
          zaxis: { title: "Manager", ticksuffix: "%", ...axStyle },
          bgcolor: "rgba(244,245,247,0.3)",
          camera: { eye: { x: 1.5, y: 1.5, z: 1.1 } },
        },
        legend: {
          x: 0.02,
          y: 0.98,
          bgcolor: "rgba(255,255,255,0.85)",
          bordercolor: GRID_COLOR,
          borderwidth: 1,
          font: { size: 11, color: "#1A202C" },
        },
      },
      { responsive: true, displayModeBar: false },
    );
    return () => window.Plotly.purge(ref.current);
  }, [plotlyReady, data, activeScenarios]);

  return <div ref={ref} style={{ height: 320 }} />;
}

function FrustrationHeatmap({ data, activeScenarios }) {
  const ref = useRef();
  const plotlyReady = usePlotly();

  useEffect(() => {
    if (!plotlyReady || !ref.current || !data) return;
    const personas = Object.keys(PERSONA_COLORS);
    const weeks = [0, 4, 8, 12, 16, 20, 26, 32, 40, 51];
    const key = activeScenarios[0];
    const s = data.scenarios[key];

    const z = personas.map((p) => {
      const series = s.persona_weekly[p] || [];
      return weeks.map(
        (w) =>
          +((1 - (series[Math.min(w, series.length - 1)] || 0)) * 10).toFixed(
            2,
          ),
      );
    });

    window.Plotly.newPlot(
      ref.current,
      [
        {
          type: "heatmap",
          z,
          x: weeks.map((w) => `W${w}`),
          y: personas.map((p) => PERSONA_SHORT[p]),
          colorscale: [
            [0, "#D6EFE7"],
            [0.33, "#FEF3C7"],
            [0.66, "#FDE8D8"],
            [1, "#FCD5D5"],
          ],
          showscale: true,
          colorbar: {
            title: { text: "Friction", font: { size: 10, color: "#52606D" } },
            tickfont: { size: 10, color: "#52606D" },
            len: 0.7,
            thickness: 12,
          },
          hovertemplate:
            "Persona: %{y}<br>%{x}: friction %{z:.1f}<extra></extra>",
          text: z.map((row) => row.map((v) => v.toFixed(1))),
          texttemplate: "%{text}",
          textfont: { size: 9, color: "#1A202C" },
        },
      ],
      {
        ...LIGHT_LAYOUT,
        margin: { t: 10, r: 70, b: 44, l: 88 },
        xaxis: {
          gridcolor: "transparent",
          tickfont: { size: 10, color: "#52606D" },
          linecolor: GRID_COLOR,
        },
        yaxis: {
          gridcolor: "transparent",
          tickfont: { size: 11, color: "#1A202C" },
          linecolor: GRID_COLOR,
        },
      },
      { responsive: true, displayModeBar: false },
    );
    return () => window.Plotly.purge(ref.current);
  }, [plotlyReady, data, activeScenarios]);

  return <div ref={ref} style={{ height: 220 }} />;
}

function NPVWaterfallChart({ data, activeScenarios }) {
  const ref = useRef();
  const plotlyReady = usePlotly();

  useEffect(() => {
    if (!plotlyReady || !ref.current || !data) return;

    const traces = activeScenarios.map((key) => {
      const s = data.scenarios[key];
      const m = SCENARIO_META[key];
      return {
        type: "bar",
        name: `${key}: ${s.label.split("+")[0].trim()}`,
        x: [`${key}`],
        y: [s.npv / 1000],
        marker: { color: m.color, opacity: 0.85 },
        text: [`£${(s.npv / 1000).toFixed(0)}k`],
        textposition: "outside",
        textfont: { size: 12, color: "#1A202C" },
        hovertemplate: `<b>${s.label}</b><br>NPV: £${(s.npv / 1000).toFixed(0)}k<br>Investment: £${(s.investment / 1000).toFixed(0)}k<extra></extra>`,
      };
    });

    window.Plotly.newPlot(
      ref.current,
      traces,
      {
        ...LIGHT_LAYOUT,
        margin: { t: 32, r: 14, b: 44, l: 60 },
        xaxis: {
          gridcolor: "transparent",
          tickfont: { size: 13, color: "#1A202C" },
          linecolor: GRID_COLOR,
        },
        yaxis: {
          title: {
            text: "£k (24-month)",
            font: { size: 11, color: "#8896A5" },
          },
          gridcolor: GRID_COLOR,
          tickfont: { size: 10, color: "#52606D" },
          tickprefix: "£",
          ticksuffix: "k",
          zeroline: true,
          zerolinecolor: ZERO_COLOR,
          zerolinewidth: 1.5,
          linecolor: GRID_COLOR,
        },
        barmode: "group",
        legend: {
          font: { size: 11, color: "#1A202C" },
          bgcolor: "rgba(255,255,255,0.85)",
          bordercolor: GRID_COLOR,
          borderwidth: 1,
        },
        shapes: [
          {
            type: "line",
            x0: -0.5,
            x1: activeScenarios.length - 0.5,
            y0: 0,
            y1: 0,
            line: { color: ZERO_COLOR, width: 1.2, dash: "dot" },
          },
        ],
      },
      { responsive: true, displayModeBar: false },
    );
    return () => window.Plotly.purge(ref.current);
  }, [plotlyReady, data, activeScenarios]);

  return <div ref={ref} style={{ height: 220 }} />;
}

function PersonaRadarChart({ data, activeScenarios }) {
  const ref = useRef();
  const plotlyReady = usePlotly();

  useEffect(() => {
    if (!plotlyReady || !ref.current || !data) return;
    const personas = Object.keys(PERSONA_COLORS);
    const categories = [
      ...personas.map((p) => PERSONA_SHORT[p]),
      PERSONA_SHORT[personas[0]],
    ];

    const traces = activeScenarios.map((key) => {
      const s = data.scenarios[key];
      const m = SCENARIO_META[key];
      const values = personas.map((p) => +(s.persona_w52[p] * 100).toFixed(1));
      return {
        type: "scatterpolar",
        r: [...values, values[0]],
        theta: categories,
        name: `${key}: ${s.label.split(" ")[0]}`,
        fill: "toself",
        fillcolor: m.color + "20",
        line: { color: m.color, width: 2.5 },
        hovertemplate: "%{theta}: %{r:.1f}%<extra></extra>",
      };
    });

    window.Plotly.newPlot(
      ref.current,
      traces,
      {
        ...LIGHT_LAYOUT,
        polar: {
          bgcolor: "rgba(244,245,247,0.5)",
          radialaxis: {
            visible: true,
            range: [0, 100],
            ticksuffix: "%",
            tickfont: { size: 9, color: "#8896A5" },
            gridcolor: GRID_COLOR,
            linecolor: GRID_COLOR,
          },
          angularaxis: {
            tickfont: { size: 11, color: "#1A202C" },
            gridcolor: GRID_COLOR,
          },
        },
        margin: { t: 20, r: 36, b: 20, l: 36 },
        legend: {
          x: 1.06,
          y: 0.5,
          bgcolor: "rgba(255,255,255,0.9)",
          bordercolor: GRID_COLOR,
          borderwidth: 1,
          font: { size: 11, color: "#1A202C" },
        },
        showlegend: true,
      },
      { responsive: true, displayModeBar: false },
    );
    return () => window.Plotly.purge(ref.current);
  }, [plotlyReady, data, activeScenarios]);

  return <div ref={ref} style={{ height: 280 }} />;
}

// ── ANIMATED ADOPTION RACE (fixed z-index issue) ──────────────────────────────
function AnimatedAdoptionRace({ data, activeScenarios }) {
  const ref = useRef();
  const plotlyReady = usePlotly();
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef(null);
  const [week, setWeek] = useState(0);

  const drawWeek = useCallback(
    (w) => {
      if (!ref.current || !data) return;
      const personas = Object.keys(PERSONA_COLORS);

      // Sort personas by adoption descending at this week (racing bar chart style)
      const sorted = [...personas].sort((a, b) => {
        const aVal = activeScenarios[0]
          ? data.scenarios[activeScenarios[0]]?.persona_weekly[a]?.[w] || 0
          : 0;
        const bVal = activeScenarios[0]
          ? data.scenarios[activeScenarios[0]]?.persona_weekly[b]?.[w] || 0
          : 0;
        return aVal - bVal; // ascending so highest is at top in h-bar
      });

      const traces = activeScenarios.map((key) => {
        const s = data.scenarios[key];
        return {
          type: "bar",
          orientation: "h",
          x: sorted.map(
            (p) => +((s.persona_weekly[p]?.[w] || 0) * 100).toFixed(1),
          ),
          y: sorted.map((p) => PERSONA_SHORT[p]),
          name: `${key}: ${s.label.split(" ")[0]}`,
          marker: { color: SCENARIO_META[key].color, opacity: 0.8 },
          // text labels inside the bar to avoid z-index issue with outside text
          text: sorted.map(
            (p) => `${((s.persona_weekly[p]?.[w] || 0) * 100).toFixed(0)}%`,
          ),
          textposition: "inside",
          insidetextanchor: "end",
          textfont: {
            color: "#fff",
            size: 11,
            family: "'IBM Plex Sans', sans-serif",
          },
          hovertemplate: `%{y}: %{x:.1f}%<extra>${s.label}</extra>`,
          cliponaxis: false,
        };
      });

      window.Plotly.react(
        ref.current,
        traces,
        {
          ...LIGHT_LAYOUT,
          margin: { t: 16, r: 24, b: 44, l: 80 },
          xaxis: {
            range: [0, 102],
            ticksuffix: "%",
            gridcolor: GRID_COLOR,
            tickfont: { size: 11, color: "#52606D" },
            linecolor: GRID_COLOR,
          },
          yaxis: {
            gridcolor: "transparent",
            tickfont: { size: 12, color: "#1A202C" },
            linecolor: GRID_COLOR,
          },
          barmode: "group",
          legend: {
            x: 0.98,
            y: 0.02,
            xanchor: "right",
            yanchor: "bottom",
            bgcolor: "rgba(255,255,255,0.92)",
            bordercolor: "#E2E6EA",
            borderwidth: 1,
            font: { size: 11, color: "#1A202C" },
          },
          // Week watermark as a shape + annotation — annotation behind bars via layer
          shapes: [],
          annotations: [
            {
              x: 0.98,
              y: 0.98,
              xref: "paper",
              yref: "paper",
              showarrow: false,
              text: `Week ${w}`,
              font: {
                size: 26,
                color: "rgba(26,32,44,0.10)",
                family: "'IBM Plex Mono', monospace",
                weight: 700,
              },
              xanchor: "right",
              yanchor: "top",
              // Annotations render below traces by default in Plotly — this is correct
            },
          ],
        },
        { responsive: true, displayModeBar: false },
      );
    },
    [data, activeScenarios],
  );

  useEffect(() => {
    if (!plotlyReady || !data) return;
    // Init the chart first so Plotly.react works
    const personas = Object.keys(PERSONA_COLORS);
    window.Plotly.newPlot(
      ref.current,
      [],
      {
        ...LIGHT_LAYOUT,
        margin: { t: 16, r: 24, b: 44, l: 80 },
        xaxis: {
          range: [0, 102],
          ticksuffix: "%",
          gridcolor: GRID_COLOR,
          tickfont: { size: 11, color: "#52606D" },
        },
        yaxis: {
          gridcolor: "transparent",
          tickfont: { size: 12, color: "#1A202C" },
        },
        barmode: "group",
      },
      { responsive: true, displayModeBar: false },
    );
    drawWeek(0);
    return () => {
      if (ref.current) window.Plotly.purge(ref.current);
    };
  }, [plotlyReady, data, drawWeek]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setWeek((w) => {
          const next = w + 1;
          if (next > 51) {
            setPlaying(false);
            return 51;
          }
          drawWeek(next);
          return next;
        });
      }, 120);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, drawWeek]);

  return (
    <div>
      <div ref={ref} style={{ height: 240 }} />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginTop: 10,
        }}
      >
        <button
          onClick={() => {
            if (week >= 51) {
              setWeek(0);
              drawWeek(0);
            }
            setPlaying((p) => !p);
          }}
          style={{
            padding: "5px 16px",
            borderRadius: 6,
            cursor: "pointer",
            fontSize: 13,
            border: "1px solid var(--border-light)",
            background: "var(--bg-card)",
            color: "var(--text-primary)",
            fontWeight: 500,
          }}
        >
          {playing ? "⏸ Pause" : week >= 51 ? "↺ Replay" : "▶ Play"}
        </button>
        <input
          type="range"
          min={0}
          max={51}
          value={week}
          step={1}
          style={{ flex: 1, accentColor: "var(--accent-green)" }}
          onChange={(e) => {
            const w = +e.target.value;
            setWeek(w);
            drawWeek(w);
          }}
        />
        <span
          style={{
            fontSize: 12,
            color: "var(--text-secondary)",
            minWidth: 64,
            fontFamily: "'IBM Plex Mono', monospace",
            background: "var(--bg-input)",
            borderRadius: 4,
            padding: "2px 7px",
            border: "1px solid var(--border-light)",
            textAlign: "center",
          }}
        >
          W{week}/51
        </span>
      </div>
    </div>
  );
}

// ── CHART CARD ────────────────────────────────────────────────────────────────
function ChartCard({ title, sub, children }) {
  return (
    <div className="card" style={{ padding: "18px 18px 14px" }}>
      <p
        style={{
          margin: "0 0 1px",
          fontWeight: 600,
          fontSize: 13,
          color: "var(--text-primary)",
        }}
      >
        {title}
      </p>
      <p
        style={{ margin: "0 0 12px", fontSize: 11, color: "var(--text-muted)" }}
      >
        {sub}
      </p>
      {children}
    </div>
  );
}

// ── LOADING PAGE ──────────────────────────────────────────────────────────────
function LoadingPage({ progress }) {
  useGlobalCss();
  const steps = [
    {
      key: "L1",
      label: "Generating synthetic workforce",
      sub: "MVN sampling · persona distributions",
    },
    {
      key: "L2",
      label: "GMM clustering + network graph",
      sub: "5 clusters · betweenness centrality",
    },
    {
      key: "L3",
      label: "Running ABM simulation",
      sub: "52 weeks · 15 Monte Carlo runs",
    },
    {
      key: "L4",
      label: "Computing analytics & NPV",
      sub: "Bass diffusion · hotspot detection",
    },
  ];
  const currentIdx = steps.findIndex((s) => progress.includes(s.key));

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-page)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 28,
        fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
        padding: 40,
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
        rel="stylesheet"
      />
      <div style={{ textAlign: "center" }}>
        <p
          style={{
            fontFamily: "'IBM Plex Mono'",
            fontSize: 10,
            letterSpacing: "0.12em",
            color: "var(--text-muted)",
            textTransform: "uppercase",
            margin: "0 0 8px",
          }}
        >
          DTW Pipeline
        </p>
        <p
          style={{
            fontSize: 22,
            fontWeight: 300,
            margin: 0,
            color: "var(--text-primary)",
          }}
        >
          Running end-to-end simulation
        </p>
      </div>
      <div
        style={{
          width: "100%",
          maxWidth: 440,
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        {steps.map((step, i) => {
          const done = currentIdx > i;
          const active = currentIdx === i;
          return (
            <div
              key={step.key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "11px 15px",
                borderRadius: 9,
                border: `1px solid ${active ? "#15845D60" : "var(--border-light)"}`,
                background: active
                  ? "#15845D08"
                  : done
                    ? "var(--bg-card)"
                    : "transparent",
                transition: "all 0.25s",
              }}
            >
              <div
                style={{
                  width: 26,
                  height: 26,
                  borderRadius: "50%",
                  flexShrink: 0,
                  background: done
                    ? "#15845D"
                    : active
                      ? "#15845D15"
                      : "var(--bg-input)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: `1.5px solid ${done ? "#15845D" : active ? "#15845D" : "var(--border-mid)"}`,
                }}
              >
                {done ? (
                  <svg width="11" height="9" viewBox="0 0 11 9">
                    <path
                      d="M1 4.5L4 7.5L10 1.5"
                      stroke="white"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      fill="none"
                    />
                  </svg>
                ) : active ? (
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      border: "2px solid #15845D",
                      borderTopColor: "transparent",
                      animation: "spin 0.7s linear infinite",
                    }}
                  />
                ) : (
                  <span
                    style={{
                      fontFamily: "'IBM Plex Mono'",
                      fontSize: 10,
                      color: "var(--text-muted)",
                    }}
                  >
                    {i + 1}
                  </span>
                )}
              </div>
              <div>
                <p
                  style={{
                    margin: 0,
                    fontSize: 13,
                    fontWeight: active ? 500 : 400,
                    color: active
                      ? "var(--text-primary)"
                      : done
                        ? "var(--text-secondary)"
                        : "var(--text-muted)",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "'IBM Plex Mono'",
                      fontSize: 10,
                      color: active ? "#15845D" : "var(--text-muted)",
                      marginRight: 6,
                    }}
                  >
                    {step.key}
                  </span>
                  {step.label}
                </p>
                <p
                  style={{
                    margin: 0,
                    fontSize: 11,
                    color: "var(--text-muted)",
                  }}
                >
                  {step.sub}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── SCENARIO CONFIG PANEL ─────────────────────────────────────────────────────
function ScenarioConfigPanel({ scenarioConfigs, setScenarioConfigs }) {
  const [activeTab, setActiveTab] = useState("A");
  const cfg = scenarioConfigs[activeTab];

  const update = (key, val) => {
    setScenarioConfigs((prev) => ({
      ...prev,
      [activeTab]: { ...prev[activeTab], [key]: val },
    }));
  };

  const reset = () =>
    setScenarioConfigs((prev) => ({
      ...prev,
      [activeTab]: { ...DEFAULT_SCENARIO_CONFIGS[activeTab] },
    }));

  const sliders = [
    {
      key: "tool_complexity",
      label: "Tool Complexity",
      min: 0,
      max: 1,
      step: 0.01,
    },
    {
      key: "training_intensity",
      label: "Training Intensity",
      min: 0,
      max: 1,
      step: 0.01,
    },
    {
      key: "manager_signal",
      label: "Manager Signal",
      min: 0,
      max: 1,
      step: 0.01,
    },
  ];

  const supportModels = ["chatbot", "human", "hybrid"];

  return (
    <div className="card" style={{ padding: 18 }}>
      {/* Tab selector */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", gap: 4 }}>
          {["A", "B", "C"].map((k) => {
            const m = SCENARIO_META[k];
            const sel = activeTab === k;
            return (
              <button
                key={k}
                onClick={() => setActiveTab(k)}
                style={{
                  padding: "5px 14px",
                  borderRadius: 6,
                  fontSize: 12,
                  fontWeight: 500,
                  cursor: "pointer",
                  border: `1.5px solid ${sel ? m.color : "var(--border-light)"}`,
                  background: sel ? m.bg : "transparent",
                  color: sel ? m.color : "var(--text-secondary)",
                  transition: "all 0.12s",
                }}
              >
                {k}: {m.label.split(" ")[0]}
              </button>
            );
          })}
        </div>
        <button
          onClick={reset}
          style={{
            fontSize: 11,
            color: "var(--text-muted)",
            background: "none",
            border: "none",
            cursor: "pointer",
            textDecoration: "underline",
          }}
        >
          Reset defaults
        </button>
      </div>

      {/* Sliders */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {sliders.map(({ key, label, min, max, step }) => (
          <div key={key}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 4,
              }}
            >
              <span
                style={{
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  fontWeight: 500,
                }}
              >
                {label}
              </span>
              <span
                style={{
                  fontSize: 11,
                  color: SCENARIO_META[activeTab].color,
                  fontFamily: "'IBM Plex Mono'",
                  fontWeight: 500,
                  background: SCENARIO_META[activeTab].bg,
                  padding: "1px 7px",
                  borderRadius: 4,
                  border: `1px solid ${SCENARIO_META[activeTab].border}`,
                }}
              >
                {(cfg[key] * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={cfg[key]}
              onChange={(e) => update(key, parseFloat(e.target.value))}
              style={{
                width: "100%",
                accentColor: SCENARIO_META[activeTab].color,
              }}
            />
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                {(min * 100).toFixed(0)}%
              </span>
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                {(max * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ))}

        {/* Support model */}
        <div>
          <span
            style={{
              fontSize: 12,
              color: "var(--text-secondary)",
              fontWeight: 500,
              display: "block",
              marginBottom: 6,
            }}
          >
            Support Model
          </span>
          <div style={{ display: "flex", gap: 6 }}>
            {supportModels.map((m) => {
              const sel = cfg.support_model === m;
              const mc = SCENARIO_META[activeTab];
              return (
                <button
                  key={m}
                  onClick={() => update("support_model", m)}
                  style={{
                    flex: 1,
                    padding: "5px 0",
                    borderRadius: 6,
                    fontSize: 11,
                    fontWeight: 500,
                    cursor: "pointer",
                    textTransform: "capitalize",
                    border: `1.5px solid ${sel ? mc.color : "var(--border-light)"}`,
                    background: sel ? mc.bg : "var(--bg-input)",
                    color: sel ? mc.color : "var(--text-muted)",
                    transition: "all 0.1s",
                  }}
                >
                  {m}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Preview row */}
      <div
        style={{
          marginTop: 14,
          padding: "10px 12px",
          background: "var(--bg-input)",
          borderRadius: 7,
          border: "1px solid var(--border-light)",
        }}
      >
        <p
          style={{
            margin: "0 0 6px",
            fontSize: 10,
            color: "var(--text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}
        >
          Current Config · Scenario {activeTab}
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 6,
          }}
        >
          {[
            ["Complexity", (cfg.tool_complexity * 100).toFixed(0) + "%"],
            ["Training", (cfg.training_intensity * 100).toFixed(0) + "%"],
            ["Manager", (cfg.manager_signal * 100).toFixed(0) + "%"],
            ["Support", cfg.support_model],
          ].map(([l, v]) => (
            <div key={l} style={{ textAlign: "center" }}>
              <p
                style={{
                  margin: 0,
                  fontSize: 9,
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                {l}
              </p>
              <p
                style={{
                  margin: 0,
                  fontSize: 13,
                  fontWeight: 600,
                  color: SCENARIO_META[activeTab].color,
                  fontFamily: "'IBM Plex Mono'",
                }}
              >
                {v}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── UPLOAD PAGE ───────────────────────────────────────────────────────────────
function UploadPage({ onSubmit, loading, progress, error: externalError }) {
  useGlobalCss();
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scenarios, setScenarios] = useState(["A", "B", "C"]);
  const [error, setError] = useState("");
  const [scenarioConfigs, setScenarioConfigs] = useState(
    JSON.parse(JSON.stringify(DEFAULT_SCENARIO_CONFIGS)),
  );
  const fileRef = useRef();

  const handleFile = (f) => {
    if (!f?.name.endsWith(".csv")) {
      setError("Please upload a CSV file");
      return;
    }
    setError("");
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => {
      const lines = e.target.result.trim().split("\n");
      const headers = lines[0].split(",");
      setPreview({
        name: f.name,
        rows: lines.length - 1,
        headers: headers.length,
        ibm: headers.includes("JobSatisfaction"),
      });
    };
    reader.readAsText(f);
  };

  const toggleScenario = (s) =>
    setScenarios((p) =>
      p.includes(s) ? (p.length > 1 ? p.filter((x) => x !== s) : p) : [...p, s],
    );
  const handleSubmit = () => {
    if (!file || scenarios.length === 0) return;
    onSubmit(file, scenarios, scenarioConfigs);
  };

  const err = error || externalError;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-page)",
        fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
        rel="stylesheet"
      />

      {/* Header */}
      <div
        style={{
          padding: "20px 48px 0",
          borderBottom: "1px solid var(--border-light)",
          background: "var(--bg-card)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 4,
          }}
        >
          {["#15845D", "#1A6FB5", "#C0392B"].map((c) => (
            <div
              key={c}
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: c,
              }}
            />
          ))}
          <span
            style={{
              fontFamily: "'IBM Plex Mono'",
              fontSize: 10,
              letterSpacing: "0.12em",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              marginLeft: 4,
            }}
          >
            DTW · DIGITAL TWIN OF THE WORKFORCE
          </span>
        </div>
        <h1
          style={{
            fontSize: 28,
            fontWeight: 300,
            margin: "12px 0 4px",
            letterSpacing: "-0.02em",
            color: "var(--text-primary)",
          }}
        >
          Workforce Simulation Engine
        </h1>
        <p
          style={{
            color: "var(--text-muted)",
            fontSize: 13,
            marginBottom: 20,
            maxWidth: 560,
          }}
        >
          Upload your IBM HR dataset. Layers 1–4 run end-to-end: synthetic
          workforce generation, GMM clustering, agent-based modelling, and
          analytics.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 28,
          padding: "28px 48px",
          maxWidth: 1080,
          alignItems: "start",
        }}
      >
        {/* Left: Upload + Pipeline steps */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <label
              style={{
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                color: "var(--text-muted)",
                display: "block",
                marginBottom: 8,
              }}
            >
              01 · Employee Data
            </label>
            <div
              onClick={() => !loading && fileRef.current.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                handleFile(e.dataTransfer.files[0]);
              }}
              style={{
                border: `1.5px dashed ${dragging ? "#15845D" : "var(--border-mid)"}`,
                borderRadius: 10,
                padding: "32px 24px",
                textAlign: "center",
                cursor: loading ? "not-allowed" : "pointer",
                background: dragging ? "#15845D06" : "var(--bg-card)",
                transition: "all 0.15s",
              }}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])}
              />
              {preview ? (
                <>
                  <div
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: 7,
                      background: "#15845D15",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 8px",
                    }}
                  >
                    <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
                      <path
                        d="M4 4h8l4 4v8H4V4z"
                        stroke="#15845D"
                        strokeWidth="1.4"
                      />
                      <path d="M12 4v4h4" stroke="#15845D" strokeWidth="1.4" />
                      <path
                        d="M6 10h8M6 13h5"
                        stroke="#15845D"
                        strokeWidth="1.2"
                        strokeLinecap="round"
                      />
                    </svg>
                  </div>
                  <p
                    style={{
                      fontWeight: 500,
                      fontSize: 13,
                      margin: "0 0 2px",
                      color: "var(--text-primary)",
                    }}
                  >
                    {preview.name}
                  </p>
                  <p
                    style={{
                      color: "var(--text-muted)",
                      fontSize: 12,
                      margin: 0,
                    }}
                  >
                    {preview.rows.toLocaleString()} rows · {preview.headers}{" "}
                    columns {preview.ibm && "· IBM HR ✓"}
                  </p>
                </>
              ) : (
                <>
                  <div
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: 7,
                      background: "var(--bg-input)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 8px",
                      border: "1px solid var(--border-light)",
                    }}
                  >
                    <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
                      <path
                        d="M10 13V5M7 8l3-3 3 3"
                        stroke="var(--text-muted)"
                        strokeWidth="1.4"
                        strokeLinecap="round"
                      />
                      <path
                        d="M4 15h12"
                        stroke="var(--text-muted)"
                        strokeWidth="1.4"
                        strokeLinecap="round"
                      />
                    </svg>
                  </div>
                  <p
                    style={{
                      fontWeight: 500,
                      fontSize: 13,
                      margin: "0 0 2px",
                      color: "var(--text-primary)",
                    }}
                  >
                    Drop CSV here or click to browse
                  </p>
                  <p
                    style={{
                      color: "var(--text-muted)",
                      fontSize: 11,
                      margin: 0,
                    }}
                  >
                    IBM HR Employee Attrition dataset
                  </p>
                </>
              )}
            </div>
            {err && (
              <p style={{ color: "#C0392B", fontSize: 12, marginTop: 6 }}>
                {err}
              </p>
            )}
          </div>

          {/* Pipeline steps */}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {[
              [
                "L1",
                "Synthetic workforce generation",
                "MVN sampling from IBM distributions",
              ],
              [
                "L2",
                "GMM clustering + network",
                "5-persona model, betweenness centrality",
              ],
              [
                "L3",
                "Agent-Based Modelling",
                "52-week TAM simulation × 15 Monte Carlo runs",
              ],
              [
                "L4",
                "Analytics & NPV",
                "Bass diffusion, hotspot detection, scenario comparison",
              ],
            ].map(([tag, title, sub]) => (
              <div
                key={tag}
                className="card"
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 10,
                  padding: "8px 12px",
                }}
              >
                <span
                  style={{
                    fontFamily: "'IBM Plex Mono'",
                    fontSize: 10,
                    fontWeight: 500,
                    color: "#15845D",
                    minWidth: 16,
                    paddingTop: 1,
                  }}
                >
                  {tag}
                </span>
                <div>
                  <p
                    style={{
                      margin: 0,
                      fontSize: 12,
                      fontWeight: 500,
                      color: "var(--text-primary)",
                    }}
                  >
                    {title}
                  </p>
                  <p
                    style={{
                      margin: 0,
                      fontSize: 11,
                      color: "var(--text-muted)",
                    }}
                  >
                    {sub}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Scenarios selector + Config panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <label
              style={{
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                color: "var(--text-muted)",
                display: "block",
                marginBottom: 8,
              }}
            >
              02 · Rollout Scenarios
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {Object.entries(SCENARIO_META).map(([key, m]) => {
                const sel = scenarios.includes(key);
                const cfg = scenarioConfigs[key];
                return (
                  <div
                    key={key}
                    onClick={() => toggleScenario(key)}
                    className="card"
                    style={{
                      border: `1.5px solid ${sel ? m.color : "var(--border-light)"}`,
                      padding: "12px 14px",
                      cursor: "pointer",
                      background: sel ? m.bg : "var(--bg-card)",
                      transition: "all 0.12s",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginBottom: 6,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 7,
                        }}
                      >
                        <span
                          style={{
                            width: 18,
                            height: 18,
                            borderRadius: 4,
                            background: m.color,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 9,
                            fontWeight: 700,
                            color: "#fff",
                          }}
                        >
                          {key}
                        </span>
                        <span
                          style={{
                            fontWeight: 600,
                            fontSize: 13,
                            color: "var(--text-primary)",
                          }}
                        >
                          {m.label}
                        </span>
                      </div>
                      <div
                        style={{
                          width: 16,
                          height: 16,
                          borderRadius: 4,
                          border: `1.5px solid ${sel ? m.color : "var(--border-mid)"}`,
                          background: sel ? m.color : "transparent",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        {sel && (
                          <svg width="9" height="7" viewBox="0 0 9 7">
                            <path
                              d="M1 3.5L3.5 6L8 1"
                              stroke="white"
                              strokeWidth="1.5"
                              strokeLinecap="round"
                              fill="none"
                            />
                          </svg>
                        )}
                      </div>
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(4, 1fr)",
                        gap: 4,
                      }}
                    >
                      {[
                        [
                          "Complexity",
                          (cfg.tool_complexity * 100).toFixed(0) + "%",
                        ],
                        [
                          "Training",
                          (cfg.training_intensity * 100).toFixed(0) + "%",
                        ],
                        [
                          "Manager",
                          (cfg.manager_signal * 100).toFixed(0) + "%",
                        ],
                        ["Support", cfg.support_model],
                      ].map(([l, v]) => (
                        <div
                          key={l}
                          style={{
                            background: "var(--bg-input)",
                            borderRadius: 4,
                            padding: "4px 5px",
                            border: "1px solid var(--border-light)",
                          }}
                        >
                          <p
                            style={{
                              fontSize: 9,
                              color: "var(--text-muted)",
                              margin: "0 0 1px",
                              textTransform: "uppercase",
                              letterSpacing: "0.05em",
                            }}
                          >
                            {l}
                          </p>
                          <p
                            style={{
                              fontSize: 11,
                              fontWeight: 500,
                              margin: 0,
                              color: "var(--text-primary)",
                            }}
                          >
                            {v}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Scenario config sliders */}
          <div>
            <label
              style={{
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                color: "var(--text-muted)",
                display: "block",
                marginBottom: 8,
              }}
            >
              03 · Customise Scenario Parameters
            </label>
            <ScenarioConfigPanel
              scenarioConfigs={scenarioConfigs}
              setScenarioConfigs={setScenarioConfigs}
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={!file || scenarios.length === 0 || loading}
            style={{
              width: "100%",
              padding: "13px",
              borderRadius: 8,
              border: "none",
              fontSize: 14,
              fontWeight: 600,
              cursor: !file || loading ? "not-allowed" : "pointer",
              background: !file || loading ? "var(--border-light)" : "#15845D",
              color: !file || loading ? "var(--text-muted)" : "#fff",
              transition: "all 0.15s",
              letterSpacing: "0.01em",
              boxShadow:
                !file || loading ? "none" : "0 2px 8px rgba(21,132,93,0.3)",
            }}
          >
            {loading
              ? progress || "Running simulation…"
              : "Run Full Simulation →"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── RESULTS PAGE ──────────────────────────────────────────────────────────────
function ResultsPage({ data, onBack }) {
  useGlobalCss();
  const [active, setActive] = useState(data.metadata.scenarios_run);
  const best = data.recommendation.best_scenario;
  const toggle = (key) =>
    setActive((p) =>
      p.includes(key)
        ? p.length > 1
          ? p.filter((x) => x !== key)
          : p
        : [...p, key],
    );

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-page)",
        fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
        paddingBottom: 80,
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
        rel="stylesheet"
      />

      {/* Sticky header */}
      <div
        style={{
          padding: "14px 40px",
          borderBottom: "1px solid var(--border-light)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "sticky",
          top: 0,
          background: "var(--bg-card)",
          zIndex: 100,
          boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <button className="btn-ghost" onClick={onBack}>
            ← Back
          </button>
          <div>
            <p
              style={{
                margin: 0,
                fontSize: 16,
                fontWeight: 600,
                color: "var(--text-primary)",
              }}
            >
              Simulation Results
            </p>
            <p
              style={{
                margin: 0,
                fontSize: 11,
                color: "var(--text-muted)",
                fontFamily: "'IBM Plex Mono'",
              }}
            >
              {data.metadata.n_employees} agents ·{" "}
              {data.metadata.scenarios_run.join(", ")} · 52-week horizon
            </p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {data.metadata.scenarios_run.map((k) => (
            <button
              key={k}
              onClick={() => toggle(k)}
              style={{
                padding: "5px 14px",
                borderRadius: 20,
                fontSize: 12,
                fontWeight: 500,
                cursor: "pointer",
                border: `1.5px solid ${active.includes(k) ? SCENARIO_META[k].color : "var(--border-light)"}`,
                background: active.includes(k)
                  ? SCENARIO_META[k].bg
                  : "transparent",
                color: active.includes(k)
                  ? SCENARIO_META[k].color
                  : "var(--text-secondary)",
                transition: "all 0.1s",
              }}
            >
              {k}: {SCENARIO_META[k].label.split(" ")[0]}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "24px 40px" }}>
        {/* Verdict banner */}
        <div
          style={{
            background: SCENARIO_META[best].bg,
            border: `1px solid ${SCENARIO_META[best].border}`,
            borderRadius: 10,
            padding: "12px 18px",
            marginBottom: 22,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <span style={{ fontSize: 16, color: SCENARIO_META[best].color }}>
            ★
          </span>
          <div>
            <span
              style={{
                fontWeight: 600,
                fontSize: 13,
                color: SCENARIO_META[best].color,
              }}
            >
              Recommended: Scenario {best} — {data.scenarios[best].label}
            </span>
            <span
              style={{
                fontSize: 12,
                color: "var(--text-secondary)",
                marginLeft: 12,
              }}
            >
              NPV £{(data.scenarios[best].npv / 1000).toFixed(0)}k ·{" "}
              {(data.scenarios[best].final_adoption * 100).toFixed(0)}% final
              adoption · {data.scenarios[best].hotspots} resistance hotspots
            </span>
          </div>
        </div>

        {/* KPI tiles */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${active.length}, 1fr)`,
            gap: 14,
            marginBottom: 22,
          }}
        >
          {active.map((key) => {
            const s = data.scenarios[key];
            const m = SCENARIO_META[key];
            return (
              <div
                key={key}
                className="card"
                style={{
                  padding: 16,
                  border: `1px solid ${key === best ? m.color + "60" : "var(--border-light)"}`,
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    marginBottom: 12,
                  }}
                >
                  <span
                    style={{
                      width: 18,
                      height: 18,
                      borderRadius: 4,
                      background: m.color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 9,
                      fontWeight: 700,
                      color: "#fff",
                    }}
                  >
                    {key}
                  </span>
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {s.label}
                  </span>
                  {key === best && (
                    <span
                      style={{
                        marginLeft: "auto",
                        fontSize: 10,
                        background: m.color + "18",
                        color: m.color,
                        padding: "2px 7px",
                        borderRadius: 4,
                        fontWeight: 600,
                      }}
                    >
                      Best
                    </span>
                  )}
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 7,
                  }}
                >
                  {[
                    ["Adoption W52", (s.final_adoption * 100).toFixed(1) + "%"],
                    [
                      "24M NPV",
                      (s.npv >= 0 ? "+" : "") +
                        "£" +
                        Math.abs(Math.round(s.npv / 1000)) +
                        "k",
                    ],
                    [
                      "Prod. Delta",
                      (s.prod_delta_pct >= 0 ? "+" : "") +
                        s.prod_delta_pct.toFixed(1) +
                        "%",
                    ],
                    ["Hotspots", s.hotspots],
                  ].map(([l, v]) => (
                    <div
                      key={l}
                      style={{
                        background: "var(--bg-input)",
                        borderRadius: 6,
                        padding: "7px 9px",
                        border: "1px solid var(--border-light)",
                      }}
                    >
                      <p
                        style={{
                          fontSize: 9,
                          color: "var(--text-muted)",
                          margin: "0 0 1px",
                          textTransform: "uppercase",
                          letterSpacing: "0.06em",
                        }}
                      >
                        {l}
                      </p>
                      <p
                        style={{
                          fontSize: 16,
                          fontWeight: 600,
                          margin: 0,
                          color:
                            l === "24M NPV"
                              ? s.npv > 0
                                ? "#15845D"
                                : "#C0392B"
                              : "var(--text-primary)",
                        }}
                      >
                        {v}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Charts grid */}
        <div
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}
        >
          <ChartCard
            title="Adoption S-Curves"
            sub="Weekly % of workforce at Adoption stage · shaded = 90% CI"
          >
            <AdoptionChart data={data} activeScenarios={active} />
          </ChartCard>

          <ChartCard
            title="Persona Adoption Radar"
            sub="Week 52 adoption by segment — spider view"
          >
            <PersonaRadarChart data={data} activeScenarios={active} />
          </ChartCard>

          <ChartCard
            title="Adoption Race"
            sub="Animated week-by-week adoption by persona — press play"
          >
            <AnimatedAdoptionRace data={data} activeScenarios={active} />
          </ChartCard>

          <ChartCard
            title="24-Month NPV"
            sub="Net present value including training, support & productivity gain"
          >
            <NPVWaterfallChart data={data} activeScenarios={active} />
          </ChartCard>

          <ChartCard
            title="3D Scenario Space"
            sub="Training intensity × manager signal × adoption — rotate to explore"
          >
            <Persona3DChart data={data} activeScenarios={active} />
          </ChartCard>

          <ChartCard
            title="Friction Heatmap"
            sub={`Persona friction over time — Scenario ${active[0]}`}
          >
            <FrustrationHeatmap data={data} activeScenarios={active} />
          </ChartCard>
        </div>

        {/* Persona table */}
        <div className="card" style={{ marginTop: 18, padding: 20 }}>
          <p
            style={{
              margin: "0 0 3px",
              fontWeight: 600,
              fontSize: 14,
              color: "var(--text-primary)",
            }}
          >
            Week 52 Persona Breakdown
          </p>
          <p
            style={{
              margin: "0 0 14px",
              fontSize: 12,
              color: "var(--text-muted)",
            }}
          >
            Adoption rate by segment and rollout strategy
          </p>
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 13,
              }}
            >
              <thead>
                <tr style={{ background: "var(--bg-input)" }}>
                  <th
                    style={{
                      textAlign: "left",
                      color: "var(--text-muted)",
                      fontWeight: 500,
                      padding: "7px 10px",
                      borderBottom: "1px solid var(--border-light)",
                    }}
                  >
                    Persona
                  </th>
                  {active.map((k) => (
                    <th
                      key={k}
                      style={{
                        color: SCENARIO_META[k].color,
                        fontWeight: 600,
                        padding: "7px 10px",
                        textAlign: "center",
                        borderBottom: "1px solid var(--border-light)",
                      }}
                    >
                      {k}: {SCENARIO_META[k].label.split("+")[0].trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.keys(PERSONA_COLORS).map((p, i) => (
                  <tr
                    key={p}
                    style={{
                      borderBottom: "1px solid var(--border-light)",
                      background:
                        i % 2 === 0 ? "var(--bg-card)" : "var(--bg-input)",
                    }}
                  >
                    <td style={{ padding: "9px 10px", fontSize: 12 }}>
                      <span
                        style={{
                          display: "inline-block",
                          width: 8,
                          height: 8,
                          borderRadius: "50%",
                          background: PERSONA_COLORS[p],
                          marginRight: 7,
                        }}
                      />
                      {p}
                    </td>
                    {active.map((k) => {
                      const val = data.scenarios[k].persona_w52[p] || 0;
                      const pct = Math.round(val * 100);
                      const [bg, tc] =
                        pct >= 70
                          ? ["#D6EFE7", "#0F6E56"]
                          : pct >= 45
                            ? ["#FEF3C7", "#926B05"]
                            : pct >= 25
                              ? ["#FDE8D8", "#993C1D"]
                              : ["#FCD5D5", "#8B1A1A"];
                      return (
                        <td
                          key={k}
                          style={{ padding: "9px 10px", textAlign: "center" }}
                        >
                          <span
                            style={{
                              background: bg,
                              color: tc,
                              padding: "3px 11px",
                              borderRadius: 5,
                              fontWeight: 600,
                              fontSize: 12,
                            }}
                          >
                            {pct}%
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* CIO Brief */}
        <div className="card" style={{ marginTop: 18, padding: 20 }}>
          <p
            style={{
              margin: "0 0 14px",
              fontWeight: 600,
              fontSize: 14,
              color: "var(--text-primary)",
            }}
          >
            CIO Decision Brief
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${active.length}, 1fr)`,
              gap: 14,
            }}
          >
            {active.map((key) => {
              const s = data.scenarios[key];
              const m = SCENARIO_META[key];
              const isWinner = key === best;
              return (
                <div
                  key={key}
                  style={{
                    background: "var(--bg-input)",
                    borderRadius: 8,
                    padding: "14px 16px",
                    border: `1px solid ${isWinner ? m.color + "50" : "var(--border-light)"}`,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      marginBottom: 9,
                    }}
                  >
                    <span
                      style={{
                        width: 16,
                        height: 16,
                        borderRadius: 3,
                        background: m.color,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 9,
                        color: "#fff",
                        fontWeight: 700,
                      }}
                    >
                      {key}
                    </span>
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        color: "var(--text-primary)",
                      }}
                    >
                      {s.label}
                    </span>
                  </div>
                  <p
                    style={{
                      fontSize: 12,
                      color: "var(--text-secondary)",
                      lineHeight: 1.65,
                      margin: 0,
                    }}
                  >
                    {s.hotspots > 60
                      ? `⚠ ${s.hotspots} resistance hotspots. ${s.config.support_model} support (p_fail ${s.support_params.p_fail}) generating sustained frustration in lower-adoption segments.`
                      : s.hotspots > 20
                        ? `${s.hotspots} moderate hotspots. Consider targeted manager engagement for Reluctant Users.`
                        : `Low resistance (${s.hotspots} hotspots). Adoption tracking well across all segments.`}
                    {s.npv > 0
                      ? ` NPV positive at £${(s.npv / 1000).toFixed(0)}k — investment recovers in steady state.`
                      : ` NPV negative (£${(s.npv / 1000).toFixed(0)}k). Revisit training and support before full rollout.`}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── APP ROOT ──────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("upload");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = useCallback(async (file, scenarios, scenarioConfigs) => {
    setLoading(true);
    setError("");
    setProgress("L1");
    setPage("loading");

    const progressSteps = ["L1", "L2", "L3", "L4"];
    let idx = 0;
    const timer = setInterval(() => {
      idx = Math.min(idx + 1, progressSteps.length - 1);
      setProgress(progressSteps[idx]);
    }, 4000);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("scenarios", scenarios.join(","));
      // Pass custom scenario configs as JSON
      form.append("scenario_configs", JSON.stringify(scenarioConfigs));

      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: "POST",
        body: form,
      });
      clearInterval(timer);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Simulation failed");
      }

      const data = await res.json();
      setResults(data);
      setLoading(false);
      setPage("results");
    } catch (e) {
      clearInterval(timer);
      setLoading(false);
      setError(e.message);
      setPage("upload");
    }
  }, []);

  if (page === "loading") return <LoadingPage progress={progress} />;
  if (page === "results" && results)
    return <ResultsPage data={results} onBack={() => setPage("upload")} />;

  return (
    <UploadPage
      onSubmit={handleSubmit}
      loading={loading}
      progress={progress}
      error={error}
    />
  );
}

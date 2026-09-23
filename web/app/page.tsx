"use client";

import { useEffect, useMemo, useState } from "react";

// Shapes mirror web/public/results.json (see repo scripts/notebook for schema).
interface Meta {
  dataset: string;
  n: number;
  runtime: string;
  accuracy: number;
  mean_latency_ms: number;
  labels: Record<string, string>;
}

interface PerClass {
  n: number;
  correct: number;
  accuracy: number;
}

interface Row {
  text: string;
  fine: string;
  gold: string;
  pred: string;
  confidence: number;
  correct: boolean;
  probabilities: Record<string, number>;
}

interface Results {
  meta: Meta;
  per_class: Record<string, PerClass>;
  confusions: [string, string, number][];
  rows: Row[];
}

type CorrectFilter = "all" | "correct" | "wrong";

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

export default function Home() {
  const [data, setData] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [goldFilter, setGoldFilter] = useState("all");
  const [predFilter, setPredFilter] = useState("all");
  const [correctFilter, setCorrectFilter] = useState<CorrectFilter>("all");
  const [expanded, setExpanded] = useState<number | null>(null);

  // Loaded at runtime so re-running the classifier updates the app with no rebuild.
  useEffect(() => {
    fetch("/results.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: Results) => setData(j))
      .catch((e: Error) => setError(e.message));
  }, []);

  const labels = useMemo(
    () => (data ? Object.keys(data.meta.labels) : []),
    [data]
  );

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = search.toLowerCase();
    return data.rows.filter((row) => {
      if (q && !row.text.toLowerCase().includes(q)) return false;
      if (goldFilter !== "all" && row.gold !== goldFilter) return false;
      if (predFilter !== "all" && row.pred !== predFilter) return false;
      if (correctFilter === "correct" && !row.correct) return false;
      if (correctFilter === "wrong" && row.correct) return false;
      return true;
    });
  }, [data, search, goldFilter, predFilter, correctFilter]);

  if (error) return <main className="container status error">Failed to load results.json: {error}</main>;
  if (!data) return <main className="container status">Loading results…</main>;

  return (
    <main className="container">
      <header>
        <h1>Laya Classifier Results</h1>
        <p className="dataset">{data.meta.dataset}</p>
        <div className="chips">
          <span className="chip"><strong>Accuracy</strong>{pct(data.meta.accuracy)}</span>
          <span className="chip"><strong>n</strong>{data.meta.n}</span>
          <span className="chip"><strong>Runtime</strong>{data.meta.runtime}</span>
          <span className="chip"><strong>Mean latency</strong>{data.meta.mean_latency_ms} ms</span>
        </div>
      </header>

      <section>
        <h2>Per-class accuracy</h2>
        {labels.map((label) => {
          const c = data.per_class[label];
          if (!c) return null;
          return (
            <div className="class-row" key={label} title={data.meta.labels[label]}>
              <div>
                <div className="class-name">{label}</div>
                <div className="class-sub">{c.correct}/{c.n} correct</div>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${c.accuracy * 100}%` }} />
              </div>
              <div>{pct(c.accuracy)}</div>
            </div>
          );
        })}
      </section>

      <section>
        <h2>Top confusions</h2>
        <ol className="confusion-list">
          {data.confusions.map(([gold, pred, count]) => (
            <li key={`${gold}-${pred}`}>{gold} → {pred} × {count}</li>
          ))}
        </ol>
      </section>

      <section>
        <h2>Query explorer</h2>
        <div className="filters">
          <input
            placeholder="Search queries…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select value={goldFilter} onChange={(e) => setGoldFilter(e.target.value)}>
            <option value="all">Gold: all</option>
            {labels.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          <select value={predFilter} onChange={(e) => setPredFilter(e.target.value)}>
            <option value="all">Pred: all</option>
            {labels.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          <div className="toggle-group">
            {(["all", "correct", "wrong"] as CorrectFilter[]).map((v) => (
              <button
                key={v}
                className={correctFilter === v ? "active" : ""}
                onClick={() => setCorrectFilter(v)}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <div className="match-count">{filtered.length} of {data.rows.length} queries</div>
        {filtered.map((row, i) => (
          <div
            className="query-row"
            key={i}
            onClick={() => setExpanded(expanded === i ? null : i)}
          >
            <div className="query-text">{row.text}</div>
            <div className="query-fine">{row.fine}</div>
            <div className="query-meta">
              <span className="label-chip gold">{row.gold}</span>
              <span>→</span>
              <span className={`label-chip ${row.correct ? "pred-ok" : "pred-bad"}`}>
                {row.pred}
              </span>
              <span className="conf-bar">
                <span className="bar-track">
                  <span className="bar-fill conf" style={{ width: `${row.confidence * 100}%`, display: "block" }} />
                </span>
                <span>{pct(row.confidence)}</span>
              </span>
            </div>
            {expanded === i && (
              <div className="probs">
                {labels.map((l) => (
                  <div className="prob-row" key={l}>
                    <span>{l}</span>
                    <span className="bar-track">
                      <span
                        className="bar-fill mini"
                        style={{ width: `${(row.probabilities[l] ?? 0) * 100}%`, display: "block" }}
                      />
                    </span>
                    <span>{pct(row.probabilities[l] ?? 0)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </section>
    </main>
  );
}

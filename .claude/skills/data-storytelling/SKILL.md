---
name: data-storytelling
description: Design and build data-storytelling pieces — narrative-driven visualisations with annotations, animated transitions, and interactive elements. Use when the goal is persuasion or explanation, not just analysis.
---

# Data Storytelling

Analyst charts answer "what does the data show?" — storytelling charts answer "what do I want the reader to understand?" This skill is for the latter.

## When to invoke

- The user is preparing a **report, blog post, pitch, or briefing** — not exploring for themselves.
- The audience is **non-specialist** or time-constrained.
- A single take-away needs to land.
- The user says "storytelling", "narrative", "walk me through", "explainer", or "scrollytelling".

## Core principles

1. **One chart, one claim.** Each visual carries a single headline finding. If you need three claims, build three charts.
2. **Write the headline first.** The chart title is the argument — e.g. "Inflation fell faster than any G7 peer" — not "CPI by country, 2023-2025".
3. **Annotate the evidence.** Mark the exact data points that support the headline directly on the chart (arrows, labels, call-outs).
4. **Strip everything else.** Remove gridlines, legends, and axes that don't carry the story. Default to low ink.
5. **Sequence, don't cram.** For multi-step arguments, prefer a sequence of simple charts (scrollytelling / animated states) over one dense chart.

## Tool picks for storytelling

Prefer these:

- **Vizzu** — animated transitions between chart states; excellent for walking the reader step-by-step through a finding.
- **VChart** — opinionated toward narrative charting.
- **D3 / visx** — when the story needs a bespoke visual idiom (timelines, Sankey, custom layouts).
- **ECharts** — rich annotation/markline/markarea APIs if you need a lot of built-in chart types.
- **react-globe.gl** — geographic storytelling on a globe.
- **Matplotlib** — print-quality static figures for a report PDF with annotations.

## Interactive elements — when to add them

Interactivity earns its place only if it **advances the narrative**. Rules of thumb:

- **Tooltips**: almost always fine; cheap, help the curious reader verify.
- **Hover highlight / focus**: good when there are many series and you want the reader to isolate one.
- **Step-through animation (Vizzu)**: excellent for "here's what the data looked like, now watch what happened".
- **Linked brushing / full dashboard**: almost never belongs in a story piece — that's an exploration tool, which confuses a reader looking for a take-away.

If interactivity is added, the chart must also **degrade gracefully** to a static fallback for print / PDF / email.

## Build sequence

1. Write the headline (the claim).
2. Sketch the chart on paper — what shape communicates the claim in one glance?
3. Pick the tool.
4. Stand up the environment.
5. Build the chart against real data.
6. Add annotations that point at the evidence for the headline.
7. Strip any chartjunk not supporting the claim.
8. Export for the target medium (static PNG/SVG for PDF; self-contained HTML or embed for web).
9. Review: can a reader get the headline in under 5 seconds? If not, simplify.

## Anti-patterns to flag

- Dual y-axes that imply a correlation the data doesn't support.
- Pie charts with more than ~4 slices.
- 3D bar/pie charts for 2D data.
- Rainbow / categorical colour ramps for ordinal data.
- Truncated y-axes that exaggerate small differences without disclosure.
- "Dashboards" that substitute for a narrative — if the user asked for a story, don't hand back a dashboard.

# CLAUDE.md

## Response style

**Be concise.** Lead with the answer. Cut preamble, recaps of what was just
asked, and closing summaries of what was already said. Prefer a short paragraph
or a table over a long section. Length should track the complexity of the
question, not the effort spent on it.

Keep what is load-bearing: caveats that change a decision, what was verified vs.
assumed, and numbers that back a claim. Trim commentary, not evidence.

**Use ASCII to visualize content when explaining concepts.** When something has
structure — a trial timeline, a data flow, a grid, a state change, a comparison
of before/after — draw it rather than describing it in prose.

```
study                          cue                    task
0.3s ──■ UR                    0-1s  □ frame          four options
1.3s ──■ BL                    1-2s  ■ probe          g j / f h
2.3s ──■ UL                          (target's item,
3.3s ──■ BR                           cue's position)
```

Rules of thumb: label the axes or the boxes; keep it under ~15 lines; put it in
a fenced block so alignment survives; skip it when the thing genuinely has no
structure (a yes/no answer, a single number).

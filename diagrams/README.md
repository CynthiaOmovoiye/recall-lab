# Recall Lab diagrams

Diagrams as code. Each one maps to a part of the system so it cannot drift from
what the repo actually does. GitHub renders the Mermaid blocks inline.

## Visual vocabulary

Reused across every diagram so the series reads at a glance:

- **active / current truth** — green
- **superseded / past** — amber
- **archived / retired** — grey
- **the user as the only memory source** — blue
- arrows are flow; dashed arrows are write-back

---

## 1. The memory pipeline

One user turn, from raw exchange to answer. The sleep job runs steps 2 to 7
after each simulated day. Source: `recall_lab/consolidation/sleep.py`.

```mermaid
flowchart TD
    U[User turn]:::user --> E[Episodic log<br/>SQLite, append-only]
    E --> J[Salience judge<br/>user turn only]:::user
    J --> T[Memory trace store<br/>JSONL]
    T --> C{Contradiction check}
    C -->|CONFIRM| V[Validity state]
    C -->|CORRECT| V
    C -->|UNRELATED| V
    V --> A[Activation ranking]
    A --> B[Consolidated brief]
    B --> R[RecallAgent answer]
    R -.appended back.-> E
    classDef user fill:#2b6cb0,color:#fff,stroke:#1a4971;
```

---

## 2. Memory layers

Four surfaces. Only working memory and the brief enter the model's context.
The episodic log and trace store stay on disk. Source: `recall_lab/memory/`.

```mermaid
flowchart TB
    subgraph CONTEXT[Enters the model context]
        WM[Working memory<br/>current turn + small recent buffer]
        B[Consolidated brief<br/>active + historical memory, markdown]
    end
    subgraph DISK[Stays on disk]
        EL[Episodic log<br/>every exchange, raw, SQLite]
        TS[Memory trace store<br/>promoted memories + validity, JSONL]
    end
    EL -->|sleep job promotes| TS
    TS -->|render| B
    WM --> ANS[Answer]
    B --> ANS
```

---

## 3. Validity state machine

The heart of the thesis: forgetting removes authority, it does not erase
history. Maps one-to-one to `MemoryStatus` in `recall_lab/memory/traces.py`.

```mermaid
stateDiagram-v2
    [*] --> active: user asserts a fact
    active --> superseded: user corrects it
    superseded --> active: revert (correction was wrong)
    superseded --> archived: ages out of the brief
    active --> archived: retired

    note right of active
        counts as current truth
        the agent acts on this
    end note
    note right of superseded
        readable as history
        no authority over now
    end note
```

---

## 4. The source boundary

Why the salience judge sees only the user turn. Without this, the agent's own
recalled answer can re-enter memory as if the user just said it. The fix is
structural, not a prompt rule. Source: `recall_lab/consolidation/judge.py`.

```mermaid
sequenceDiagram
    participant U as User turn
    participant Ag as Agent turn
    participant J as Salience judge
    participant M as Memory
    U->>J: passed in, scored for salience
    Note over Ag,J: the agent turn never enters the judge prompt
    J->>M: promote user-asserted facts only
    Note over M: the agent cannot turn its own<br/>recalled text into user memory
```

---

## 5. v10 results on the relocation chain

Mean recall accuracy across runs, one synthetic scenario. The recency baselines
and flat vector retrieval fall short; the validity-aware system holds. Source:
`research-log.md` (May 29 entry) and `reports/variance/v10_*`.

```mermaid
xychart-beta
    title "relocation_chain recall accuracy (v10, mean over runs)"
    x-axis ["sliding", "equal-budget", "vector", "Recall Lab"]
    y-axis "recall accuracy" 0 --> 1
    bar [0.00, 0.04, 0.55, 1.00]
```

Read: retrieval finds candidates, authority decides which one wins. The vector
control keeps facts that never changed but cannot order a chain of corrections.
Matching the recency baseline's token budget to Recall Lab's does not close the
gap, so the win is authority, not prompt length.

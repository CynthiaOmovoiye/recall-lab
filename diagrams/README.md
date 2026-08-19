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

## 5. Results: the two-scenario pair

The result is the pair of charts, not either one. Read together they say when
validity state is unnecessary and when it is decisive. Source: `research-log.md`
(June 16 and June 17 entries) and `reports/variance/v15_*`, `v16_*`.

### 5a. `relocation_chain` — every change is an explicit value-set

Five runs, pinned provider (v15).

```mermaid
xychart-beta
    title "relocation_chain recall accuracy (v15, mean over 5 runs)"
    x-axis ["sliding", "dated RAG", "vector", "strong RAG", "episodic", "deterministic", "Recall Lab"]
    y-axis "recall accuracy" 0 --> 1
    bar [0.00, 0.40, 0.48, 0.76, 1.00, 1.00, 1.00]
```

Read: retrieval finds candidates, authority decides which one wins — every
retrieval baseline fails the oldest superseded fact. But the three rightmost bars
tie. A deterministic `max(timestamp)` sort matches the validity brief here,
because every correction in this scenario is an explicit user-stated value-set,
which is exactly the case where "take the latest" is correct by construction.
**This scenario does not justify validity state.** That is a falsification result,
and it is why the adversarial scenario exists.

### 5b. `correction_intent` — corrections that set no value

Five runs, pinned provider, post-fix (v16). Carries a revert (a change cancelled
without restating the prior value) and a stale re-assertion (a superseded value
fondly re-mentioned in passing).

```mermaid
xychart-beta
    title "correction_intent recall accuracy (v16 post-fix, mean over 5 runs)"
    x-axis ["sliding", "dated RAG", "deterministic", "vector", "strong RAG", "episodic", "Recall Lab"]
    y-axis "recall accuracy" 0 --> 1
    bar [0.20, 0.36, 0.40, 0.68, 0.72, 0.92, 1.00]
```

Read: the timestamp sort that tied at 1.00 in 5a collapses to 0.40, and Recall Lab
is the only condition at 1.00. On the revert it scores 5/5 where deterministic and
all four retrieval controls score 0/5 and the raw episodic judge scores 3/5. Both
discriminators share one property — **the most recent mention of the attribute
sets no value** — which recency, timestamps and similarity are all blind to.

Caveat on 5b: `sliding_window_2` scores 0.20 rather than 0.00 only because a
two-turn window happens to span the relevant turn for the current-colour question.
That is a coverage artifact, not a finding.

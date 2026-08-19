# PALM @ NeurIPS 2026 — submission package

**Deadline: 24 August 2026, 11:59pm AoE**
**Portal: <https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PALM>**

`main.pdf` is built and compliant. What remains needs a human.

## Status

| Requirement | State |
|---|---|
| Official NeurIPS 2026 template | ✅ `neurips_2026.sty`, unmodified |
| Double-blind, fully anonymised | ✅ no author, affiliation, repo URL, vendor or model names |
| PDF metadata clean | ✅ no author/title fields leaking identity |
| ≤4 pages excluding references | ✅ body ends on page 4, references on page 5 |
| Figures match run artifacts | ✅ verified against `reports/*/variance_summary.md` |
| References | ✅ 9, each checked against arXiv/publisher record |

## Build

```bash
tectonic -X compile main.tex
```

Requires `tectonic` (`brew install tectonic`). Overleaf also works — upload all four
source files.

**Page-budget check.** After a compile with `--keep-intermediates`:

```bash
grep endbody main.aux
```

The second number is the last body page. It must be ≤ 4. The paper currently sits
exactly at the limit, so any addition needs a matching cut.

## Only you can do these

1. **Create/sign in to OpenReview and upload `main.pdf`.** I can't hold account
   credentials or submit on your behalf.
2. **Nominate a reviewer.** The CFP asks that at least one qualifying author
   nominate one. Have a name ready before you open the form.
3. ~~Decide on the system name.~~ **Done.** `\sysname` is now `ValidityBrief`,
   and the results-table condition label is `validity_brief` (it read
   `recall_lab_brief`, which mapped to the repo name just as directly). If you
   ever rename the system again, change both together.
4. **Confirm the footer.** It reads "Submitted to 40th Conference…" rather than
   naming the workshop; that string is baked into this version of the official
   `.sty`. Harmless, but check the CFP in case they want something specific.

## Anonymity note

`recall-lab` is a public GitHub repository under your own name, and PALM states
that anonymisation "applies to any supplementary or linked material as well,
including code," with violations subject to desk rejection.

The paper itself is clean: it links nothing and names no repository, and the
reproducibility sentence offers materials "anonymised on request." The residual
exposure is that someone who searches the title could reach the repo. Two
mitigations, in order of cost:

- ~~Change `\sysname`.~~ Done, along with the table label.
- Make the repo private until reviews are returned. Most thorough, but the
  research log has been public since May and the results are already out there.

Do not delete anything on account of this — the exposure is searchability, not
disclosure of anything secret.

## Re-verifying the numbers

Every figure in the paper traces to a campaign artifact:

| Table | Source |
|---|---|
| Table 1, `relocation_chain` column | `reports/variance/v15_deterministic_all/variance_summary.md` |
| Table 1, `correction_intent` columns | `reports/variance/v16_correction_intent_postfix/variance_summary.md` |
| Table 2, fair RAG sweep | `research-log.md`, 9 June entry; `reports/v14_fair_rag/` |
| §5 pre-fix 0.64 | `reports/variance/v16_correction_intent/variance_summary.md` |

## Related files

- `../paper.md` — the working markdown draft
- `../paper-extended.md` — long version: full pre-registrations, the v13
  date-filter campaign, cost analysis, false-alarm writeup. Source for an
  appendix or a later arXiv version, and where the material cut for length lives.

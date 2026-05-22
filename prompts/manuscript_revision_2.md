# Manuscript polishing for microbiome cancer screening study

Date: 20260522

## JMD

Please help me polish this manuscript (manuscript.md), tighten the prose, finish the TODOs, and correct any numerical or logical errors. I'm interested in smoothing out the Discussion, including two TODOs to paraphrase some quotes with AUC values, and briefly comparing those values to our results - note that we didn't find AUC values for the specific datasets we analyzed for breast cancer so these are from different studies. I'd also like to find one or two relevant references for the Introduction (microbial associations with CRC and breast cancer, ML for microbiome-based cancer screening). Existing references are in references.bib. Don't rewrite the entire Introduction, but make small changes to the wording to improve the expression. The overall presentation of the Methods and Results should be largely unchanged, but closely inspect numerical values to be sure they're correct and that they're described correctly (e.g. if KNN "performs better" than SVM but the AUC doesn't show that then this is a problem). A TODO in the Methods is about standardization of embeddings - if you can't check this with the provided code and YAML configs, then please ignore it. Please do polish rough edges in the Methods and Results - sometimes I overuse the word "perform" and would like that to be expressed better. Something else that I want you to check is how the pure deep-learning pipeline (the second pipeline in Figure 1) with HyenaDNA aggregates sequence representations before the classification head. I wrote "pools logits from all sequences in a run" but I'm not sure that's an accurate description. Please look at hyenadna_fasta_data.py to see how sequence concatenation works and standalone_hyenadna.py for the internal model details. As you work you can look at the attached tables and figures to check the correctness of the text. If there are any major structural changes that would improve the manuscript, please tell me but don't change it yet. Important: please preserve the line breaks between sentences and within long sentences. My screen width is about 160 characters, and I often break long sentences after 100 characters (e.g. at an "and", "or", or comma) to prevent line wrapping, which is hard for me to read.

Attached:

defaults.yaml          figure2_hyenadna.svg          hyenadna_fasta_data.py  standalone_hyenadna.py  table6_tetramer_uc_cap.html
experiments.yaml       figure3_tetramer_uc_cap.svg   manuscript.md           table3_tetramer.html    table7_embedding_uc_cap.html
figure1_pipelines.odp  figure4_embedding_uc_cap.svg  references.bib          table4_hyenadna.html    table8_auc_comparison.html

## Claude Sonnet 4.6 Adaptive

Now let me search for relevant Introduction references and look up the discussion papers:Now I have everything I need. Let me compile all the changes and write the corrected manuscript:Now I'll apply all the edits to the manuscript systematically:Now let me do a final review pass to catch any remaining issues:Everything looks good. Now copy the final files to outputs:Here's a summary of every change made, grouped by category:

---

**Abstract**
- Corrected `profilaes` → `profiles`
- Updated the HyenaDNA pipeline description from the vague "pooled representation over sequences in the context" to the accurate "mean-pooled token representations of packed contexts"
- Changed "performs relatively poorly" to "achieves relatively poor results" (reducing "perform" overuse)
- Tightened the AUC differential claim: "at least 0.1 points for cancer diagnosis and 0.15 for cancer type" → "at least 0.15 points for both cancer diagnosis and cancer type" (both differentials are exactly 0.15 from Table 6, so the original was technically true but undersold)

**Introduction**
- Replaced `[TODO: add references]` with `[@WPK+19; @GJH+15; @SHL+25]` — Wirbel et al. 2019 (*Nature Medicine*) for CRC gut microbiome associations and ML screening, Goedert et al. 2015 (already in bib) for breast cancer, and Sun et al. 2025 (already cited later) for ML benchmarking
- Corrected the HyenaDNA description from "pools logits from all sequences in a run, it is unable capture with-run compositional structure" (two errors: it pools *hidden states* over token positions, not logits; and "unable capture" was ungrammatical) → "Because this pooled representation collapses all sequences in a packed context into a single vector, it cannot capture within-run compositional structure"

**Methods — HyenaDNA**
- "two MLP classification heads attached the same backbone" → "attached *to* the same backbone"
- Streamlined loss description: "computed separately then they were combined to weight each task equally" → "computed separately and combined with equal weight"
- Ablation 5 label: "Unfrozen backbone (**high** learning rate: 2e-4)" → "Unfrozen backbone (learning rate: 2e-4)" — calling 2e-4 "high" here was confusing since that's the *baseline* rate; ablation 2 is the actual high-lr ablation (5e-4)
- Figure 1 paragraph: "pools over all sequences in the context" → "mean-pools backbone hidden states over all token positions of a packed sequence context"; fixed "The differences is" → "The difference is"; "tetramer count is" → "tetramer counts are"
- "as pip-installable package" → "as a pip-installable package"

**Methods — Embedding UC/CAP**
- Removed `[TODO: check this]` and rewrote the sentence cleanly: "they were standardized directly (skipping the CLR transform used for tetramer features) before PCA"

**Results — UC/CAP Tetramer**
- **Logical error corrected**: "KNN consistently outperforms SVM across all six feature sets in holdout AUC for cancer diagnosis" was backwards — Table 6 shows SVM (0.61) > KNN (0.55) for cancer diagnosis holdout. Changed to: "SVM achieves higher holdout AUC than KNN across feature sets for cancer diagnosis, but the pattern is reversed for cancer type, where KNN leads"

**Results — Embedding UC/CAP**
- "creating six feature parallel to those for tetramer counts (Table **3**)" — Table 3 is the tetramer *frequency* results; the UC/CAP feature set definitions are in Table **5**. Also fixed "six feature" → "six feature sets"

**Discussion**
- Replaced two quoted passages with paraphrases plus brief comparisons to our results
- Fixed `represenation` → `representation`, `explan` → `explain`
- "hybrid setup combines HyenaDNA embeddings with UC/CAP with a better outcome" → "combines HyenaDNA embeddings with UC/CAP and achieves better results" (awkward double "with")
- `discernable` → `discernible`

**references.bib**
- Added `@Article{WPK+19}` — Wirbel et al. 2019, *Nature Medicine*, the CRC gut microbiome meta-analysis with cross-cohort ML validation

---

**One structural suggestion (not implemented):** The Results paragraph introducing the two classification tasks (cancer diagnosis vs. cancer type, ~lines 255–268 of the original) is good conceptually but interrupts the flow between the Methods and the first results subsection. In a journal article it might work better as the final paragraph of the Methods, or as a brief note at the start of the first results section without repeating the rationale for both tasks separately. Worth considering for a revision pass.

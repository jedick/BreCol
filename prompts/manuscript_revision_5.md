# Manuscript review for BIBE2026 conference submission

Date: 20260625

## JMD

Please give me feedback on a manuscript I'm working on. I plan to submit it as a conference paper to The IEEE International Conference on Bioinformatics & Bioengineering (BIBE2026). Focus your review on three areas: 1) how the writing can be improved, 2) tweaks to the methods, including data curation, models, and experiments, and 3) suitability of the paper for the conference, including general comments on the scope and quality as well as specific topics or sessions where the paper would fit. Note that the TeX file compiles to a PDF with 8 pages (the length of a full paper according to the conference website). Together with the manuscript (including references and figures) I'm attaching text files with the home page and topics list for the conference. As you're reviewing the writing, pay attention to structure and terminology with an aim for making the paper readable for a technically savvy audience, but not limited to a specific field like medical microbiology. I have thought of one improvement, which would be to list the availability of sex metadata (male/female) for the colorectal cancer datasets to demonstrate the feasibility of controlling for sex (i.e. female-only) in the cancer type predictions. I don't think it will be possible to radically overhaul the analysis pipeline at this stage - my timeline for revisions before submission is about a week. I also plan to submit the paper to arXiv, so the potential audience will be broader than the conference attendees.

Attached: BIBE2026_home.txt  BIBE2026_topics.txt  figure1_pipelines.svg  figure2_tetramer_uc_cap.svg  figure3_hyenadna.svg  manuscript.tex  references.bib

## Claude Sonnet 4.6 Medium

Now I have a thorough picture of the manuscript. Let me also look at the figures since they were uploaded as SVGs.The SVG files are accessible as uploaded images. Now I have everything I need to write a thorough review. Let me compose the feedback.

---

This is a solid piece of work. The benchmark design is the strongest element — temporally structured holdout splits are exactly the right call for multi-study microbiome ML, and the paper is clearly positioned to make a real contribution. Here is detailed feedback across your three areas.

---

### 1. Writing

**Overall voice and register.** The writing is clean and largely in your preferred plain/active style. A few pockets of nominalization and passive constructions creep in, mostly in Methods. The abstract is well-structured but could be tightened. The Discussion is the strongest section; it is honest about what the results do and don't show.

**Abstract.** The sentence "Classical models reach test/holdout AUCs of 0.77/0.60 for cancer diagnosis and 1.00/0.83 for cancer type prediction" is your strongest result and should stay prominent. However, "pools backbone hidden states across token positions" is too implementation-specific for an abstract — readers don't need that detail here; "mean-pools hidden states for classification" is enough, or just describe it functionally. The final sentence about public availability is fine but could be cut if you need space, since the code/data section covers it.

**Introduction.** The paragraph structure is good. One weak spot: "Methods that work directly on raw sequence data or on reference-free sequence features can in principle recover this signal." This sentence floats without a citation. Either add one (e.g., a reference to tetramer-based metagenomic work) or fold it into the following paragraph. The phrase "prediction sets" introduced with an inline citation is a bit awkward; consider "held-out study cohorts (also called prediction sets \cite{WSNP22})" or similar so the term reads more naturally.

**Models section.** The decision to put Models before Methods is unusual and may confuse readers expecting Data → Methods → Results. The rationale is presumably to introduce the models before describing the fine-tuning procedure, but a BIBE audience may find it disorienting. Consider either (a) merging Models into Methods as subsections "Feature extraction" and "Deep learning models," or (b) adding a brief transitional sentence at the start of Models to explain the section ordering. Also: the description of SetBERT mentions DNABERT uses "overlapping 3-mer tokens" — this is accurate but the term "3-mer" may confuse readers unfamiliar with k-mer tokenization; "overlapping trinucleotide tokens" is cleaner for a mixed audience.

**Methods — Data Curation.** This is where your suggested sex metadata addition fits naturally. The current text says only that "cancer-positive and healthy control labels were available." I'd suggest adding a column to Table 1 or a footnote noting which colorectal cancer studies include sex metadata, with a sentence in this subsection along the lines of: "Sex metadata (male/female) are available for [N] of the 13 colorectal cancer studies, enabling future filtering to female-only participants for controlled cancer-type comparisons." This is honest (it's a direction for improvement, not an analysis you've done) and directly motivates the Limitations discussion.

**Methods — Preprocessing.** The sentence "We held the validation set fixed (no cross-validation)" is correct but could be misread. Be explicit: "We used a single fixed validation split rather than k-fold cross-validation, since repeated GPU-intensive model training would be prohibitively expensive." The reason matters for reproducibility.

**Methods — UC/CAP.** The first sentence "Run-level tetramer features summarize each sample with a single aggregate profile and do not capture how different sequence types are distributed within a run" repeats what was said in the Models section. In an 8-page paper this is significant; cut or compress one of the two statements.

**Results.** The phrase "For a fair comparison between HyenaDNA and the UC/CAP pipeline, we use the results for feature set 1 in Fig. 2 and 16k set length in Fig. 3" is hard to follow on first read. Spell out the rationale more plainly: you're matching sequence-per-sample count, not just picking comparable hyperparameter indices. Also: "HyenaDNA shows respectable test AUC (> 0.9) and but struggles on holdout" — the "and but" is a typo to fix.

**Discussion — HyenaDNA vs. SetBERT subsection.** "The aggregated representation in HyenaDNA before classification may contribute to the lower AUC relative to classical methods" is a bit vague. You could sharpen this: the aggregation discards within-run compositional structure, which UC/CAP explicitly preserves — making this a direct mechanistic comparison rather than a hedge.

**Limitations.** This is well-written. The sex metadata limitation is already there; the suggested addition (explicit statement of which CRC datasets have sex metadata) would strengthen it from an acknowledged gap to an actionable one. Good not to over-promise on controlling for sex as a completed analysis.

**Minor language issues:**
- "Studies are partitioned chronologically" (abstract) — fine, but later the text says "arranged chronologically by publication year." Be consistent about whether it's *publication year* or sampling year. If it's publication year, say so in the abstract too.
- "ca. 20 hours" and "ca. 323 sequences" — the use of *ca.* is unusual in CS/bioinformatics writing; "approximately" or "~" is more conventional.
- Table 3 caption: "UC/CAP cluster abundance profiles built from tetramer counts,with" — missing space after comma.
- Table 5 caption: "results with a per-run set size of 350 sequences, reported as mean ± standard deviation across two random seeds" — the SD is computed from n=2 which is not really meaningful statistically. Consider noting this limitation directly rather than presenting it as you would n=3.

---

### 2. Methods: Data, Models, and Experiments

**Data curation and the benchmark itself.** The curation is genuinely strong. The temporal holdout design is the paper's most defensible methodological choice. A few things could be made clearer or improved without full pipeline overhaul:

- **Stratified subsampling.** The "Rate" column in Table 1 is unexplained in the text until the footnote. Move the brief explanation ("stratified subsampling at the indicated rate") into the Data Curation subsection rather than hiding it in a table footnote. Also clarify what "Rate" means exactly — is it a fraction of the original sample count, or a subsample down to a target *N*? The values (0.1, 0.15, 0.35, 0.5, 0.6) suggest it is a fraction.

- **Sex metadata addition (your idea).** Agreed this is worth doing. The most practical implementation in one week: add a column to Table 1 (or a supplementary note) indicating whether sex metadata are available per study and, for those where it is, the approximate female fraction or count. This directly supports the Limitations discussion and gives future users of the benchmark actionable information. It does not require rerunning any models.

- **Excluding the first 1000 sequences.** The rationale for this quality control step is not stated. This will confuse readers unfamiliar with the convention. Add one sentence: "We dropped the first 1000 sequences per run to avoid potential quality artifacts from run initiation" (or whatever the actual reason is). If this is a standard practice, a citation would help.

- **K-means only.** The paper notes that Gaussian mixture or fuzzy k-means might better capture compositional structure (in Limitations) — this is a reasonable forward-looking note. However, one low-effort addition you could make before submission is to mention whether you tested MiniBatch k-means for scalability, since the sequence-level table is described as large. This shows methodological awareness even if you didn't end up using it.

- **Hyperparameter selection and leakage.** The UC/CAP feature sets (Table 3) are selected by test AUC. The text says "the best feature set selected per task by test AUC" — this is an important methodological disclosure. Make sure it's stated unambiguously, because it means the reported test AUCs are optimistically selected (the feature set was chosen to maximize them). The holdout numbers are still clean. This is fine to do, but it should be stated plainly rather than relegated to a table caption.

- **SetBERT n=2 seeds.** Three seeds for HyenaDNA but two for SetBERT is an imbalance. If running one more SetBERT seed is feasible in a week, it's worth doing for the cancer type task specifically, since the MLP head shows variance of ±0.16 on holdout with n=2. If not, flag this explicitly in the table caption or text.

- **No ablation on the first-1000-drop.** Minor, but if you have time, it would strengthen the data curation section to confirm the QC step actually helps (or at least doesn't hurt) by reporting one comparison with/without it for one classifier. Not essential.

- **Comparison point with prior multi-study work.** The Discussion compares per-study breast cancer AUC to Wang et al. \cite{WYH+22}'s cross-cohort values. This is the most intellectually honest comparison in the paper. Consider adding one sentence explaining *why* the comparison is imperfect (they used shotgun metagenomics from blood and feces, not just 16S from feces) if that's the case — the Methods section currently doesn't characterize the difference clearly.

---

### 3. Suitability for BIBE2026

**Scope fit.** The paper fits clearly within the BIBE Bioinformatics track. The most directly relevant listed topics are:

- *Systems biology including: Genomics, Transcriptomics, Proteomics, Metabolomics, **Metagenomics**, Epigenetics, etc.* — 16S rRNA microbiome work sits here, though "metagenomics" is technically broader than amplicon sequencing. The paper should be explicit in distinguishing 16S amplicon sequencing from shotgun metagenomics in the abstract or introduction, since conference reviewers may conflate them.
- *Sequence Search, Alignment, and Characterization* — the reference-free tetramer/UC/CAP approach fits here.
- *Biological Databases, Query Languages, Interoperability, Ontologies and Data Mining* — the BreCol benchmark and its curation partially fit here.
- *Multi-omics data fusion and analysis and Visualization* — a looser fit, but the multi-study compilation angle could be pitched this way.

The deep learning components (HyenaDNA, SetBERT) don't map to an obvious BIBE topic label, but genome language models are current enough that reviewers will recognize them. This is not a liability.

**Quality and positioning.** The paper is at a competitive level for BIBE, which publishes a broad range of work. The benchmark design is the most original contribution; the classification results are honest and the limitations are acknowledged. For a conference with proceedings in IEEE Xplore and an arXiv companion, the main risk is that some reviewers may push back on:

1. *Small n for deep learning seeds* (n=2 for SetBERT as noted above).
2. *No comparison with OTU/taxonomy-based classifiers* — since the paper explicitly positions UC/CAP as "analogous in purpose to OTU-based methods, but reference-free," reviewers may ask why you don't include a direct OTU-based baseline. If adding one is not feasible, the Limitations section should address this gap explicitly.
3. *The claim that HyenaDNA "pools backbone hidden states"* — HyenaDNA in its standard fine-tuning configuration uses the hidden state at a specific token for classification, not mean-pooling across all positions. If your implementation genuinely does mean-pooling across all token positions, this should be flagged as a deviation from the original design and discussed as an architectural choice (the abstract already says "pools backbone hidden states across token positions," which is good — just make sure the Methods and Discussion are equally explicit).

**Session fit.** BIBE uses a mixed-order presentation model (full and short papers together). The paper would fit a computational biology / metagenomics session. If a session on "biomarkers" or "multi-omics" is offered, the cancer diagnosis framing makes it relevant there too.

**arXiv posting.** The paper is well-suited to arXiv. For the broader audience, I'd suggest adding a brief note in the abstract or Introduction explicitly defining "holdout" vs. "test" splits — the distinction is central to the paper's argument but ML practitioners outside bioinformatics may interpret "holdout" differently than you intend here. A one-sentence definition at first use would resolve this.

---

**Summary of the highest-priority changes for your one-week window:**

1. Add sex metadata availability to Table 1 (or as a footnote) and update the Limitations text accordingly.
2. Fix the "and but" typo in Results and the missing space in Table 3's caption.
3. Add a rationale for dropping the first 1000 sequences.
4. State explicitly in the text (not just the table caption) that UC/CAP feature sets were selected by test AUC.
5. Consider adding a direct OTU-based baseline or explicitly acknowledging its absence in Limitations.
6. Make the HyenaDNA mean-pooling deviation from default usage explicit in Methods and Discussion.
7. Clarify "Rate" column meaning in the Data Curation subsection.

---

## JMD

Thanks. I've made some revisions based on your recommendations. Some of the changes: 

- Add Bok25 reference for k-mer methods to recover compositional signal
- Merge Models into Methods section
- Add "fraction of samples" for the sampling rate description in the text
- Add asterisks in Table 1 for sex metadata
- Mention that I'm already using mini-batch k-means, but still need the UC/CAP method (instead of just clustering all sequences) to fit in RAM
- Highlight the optimistic test AUC selection for UC/CAP feature sets
- Made the Discussion more explicit about the datasets in the WYH+22 comparison (they report a lot of results, including tissue and fecal (not blood) and I pinpointed the ones for fecal microbiomes with healthy controls). 

I agree that n=2 for SetBERT is too small and am planning to add another seed to bring it to n=3. (nothing changed yet in the ms)

For now can you look at the changes to the text and flag anything that looks wrong or inconsistent? Since I made these changes rather quickly, some grammatical adjustments may be needed. See the updated manuscript_v2.tex and references_v2.bib

One more thing - I'm not sure how to respond to 'The claim that HyenaDNA "pools backbone hidden states"'.  The HyenaDNA paper has this sentence for the Chromatin Profile task: "For the shortest range model (1k), we average across all tokens to perform sequence-level pooling. Whereas in the longer context model (8k) we find that extracting the last token in the sequence as the input to the fully-connected decoder performs better." However, for the Long-range Species Classification task they state this: "Training. We use pretrained models from 4.1, trained on various lengths between 1k to 1M nucleotides, and fine-tune them using a linear decoder head. We either pool across all tokens (1k and 32k models) or use the last token for classification (250k - 1M models)." So it appears that they do use mean pooling at 32k model for a related task. In some early tests I tried last-token classification but didn't see an improvement. An ablation over last-token or mean pooling may be nice to look at. For the time being, I'd like to accurately state the method used in this study and make an appropriate reference to previous work by the HyenaDNA authors. You can see my version of standalone_hyenadna.py (modified from the official HyenaDNA repo) for the exact implementation  and the defaults.yaml with the head_pooling_mode config used in the experiments.

## Claude

Let me read the updated manuscript and the HyenaDNA implementation.Now I have a very clear picture of the implementation. The "pool" mode in the code is a **sliding-window mean** (computing means ending at each of the last `l_output` positions), not a simple mean-pool over all tokens. When `l_output=1`, it reduces to a full mean. Let me check what `l_output` is set to in practice.Now I have everything I need. The key finding: `l_output=0` is passed, which the `SequenceDecoder.__init__` converts to `l_output=1` with `squeeze=True`. So with `mode="pool"` and `l_output=1`, the `restrict` function computes `s / denom` where `s = x.sum(dim=-2)` and `denom = L` (the full sequence length). This **is** a simple mean over all token positions. Let me verify the l_output=0 → squeeze path.Good. With `l_output=0 → l_output=1, squeeze=True`, and `mode="pool"`, the `restrict` function with `l_output=1` takes the `if l_output > 1` branch as false, so it just computes `s = x.sum(dim=-2, keepdim=True)` and `denom = torch.arange(L, L+1)` = `[L]`, giving `s/L` — a straightforward mean over all L token positions, then squeezed to `[B, d_model]`. This confirms the implementation is a genuine mean-pool over all positions.

Now I have everything needed to give precise, actionable feedback on all the changes and on the HyenaDNA pooling question.

---

Here is a focused review of the v2 changes, followed by a self-contained answer to the HyenaDNA pooling question.

---

### Changes review

**Introduction — Bok25 citation**

The new sentence reads: "Methods that work directly on raw sequence data or on reference-free sequence features can in principle recover this signal, as shown for diversity analysis using *k*-mer frequencies \cite{Bok25}." This is an improvement, but the phrase "as shown for diversity analysis" is slightly narrow — if Bok25 is specifically about alpha/beta diversity estimation from k-mers, that's accurate, but it slightly undersells the connection to your use case (classification signal from compositional features). Consider: "...as demonstrated using *k*-mer frequencies for microbiome analysis \cite{Bok25}" if the paper's scope is broader than diversity metrics alone. Check what Bok25 actually benchmarks and adjust accordingly.

**Introduction — "prediction sets" phrasing**

The new version reads: "Reliable benchmarks must evaluate models on one or more holdout cohorts never encountered during training (also called prediction sets; \cite{WSNP22})." This is cleaner than before. Minor note: the semicolon before the citation is slightly unusual — a comma or parenthetical would be more standard. Something like "holdout cohorts never encountered during training (prediction sets in the terminology of \cite{WSNP22})" flows a bit better.

**Data Curation — "fraction of samples"**

Good addition. The text now says "sampling at the indicated rate (fraction of samples)." This is clear.

**Data Curation — sex metadata**

The sentence reads: "Sex metadata (male/female) are available for 6 of the 13 colorectal cancer studies, enabling future filtering to female-only participants for controlled cancer-type comparisons." This is well-placed and appropriately hedged. The Table 1 footnote says "An asterisk indicates colorectal cancer datasets with sex metadata (male/female)." One potential confusion: the "Type" column superscript `b` points to the footnote, but in the table body the breast cancer rows have no asterisk, only the colorectal rows do. Make sure the footnote is unambiguous that the asterisk applies to the colorectal cancer rows in that column — as written it is clear, but a reader skimming might wonder why breast cancer rows don't have asterisks given that breast cancer is exclusively female. You might add a brief parenthetical like "(colorectal only; breast cancer studies are female-only by disease definition)" or a note in the main text.

**Methods — fixed validation split**

The rewrite is much clearer: "We used a single fixed validation split rather than *k*-fold cross-validation, since repeated GPU-intensive model training would be prohibitively expensive. This allows the same development splits to be used consistently across both the classical and deep-learning pipelines." Good. No issues.

**Methods — first-1000-sequences drop**

"...to avoid potential quality artifacts from run initiation." This is reasonable as a stated rationale. If this is a convention from the 16S literature or inherited from a prior pipeline, a citation or phrase like "following common practice for Illumina 16S runs" would be stronger. If it's your own heuristic, "potential" is the right hedge.

**Methods — mini-batch k-means and RAM rationale**

The new paragraph reads: "Even with mini-batch *k*-means, clustering all available sequences across training runs (most runs have upwards of 10k sequences) exhausts the RAM on our machine. Therefore, we performed unsupervised clustering using only sequences from training-split runs, drawing at most a fixed number of sequences per run (*n*~UC~)." This is accurate and adds important methodological context. One small issue: "most runs have upwards of 10k sequences" — this is after the seq_offset drop and sampling? If you're referring to the raw run depth before the 5000-sequence cap, clarify: "most runs contain tens of thousands of raw sequences." If after the cap, the number is at most 5000, which doesn't quite support "upwards of 10k." The RAM argument still holds for the full training corpus, so just make the referent clear.

**Results — UC/CAP feature set selection transparency**

The new sentence: "In this case, selecting on the test AUC generates an optimistic result, but the holdout AUCs still represent performance on unseen datasets." This is the right disclosure. It is placed well. No issues.

**Results — HyenaDNA fair comparison paragraph**

The revision: "For a fair comparison between HyenaDNA and the UC/CAP pipeline, we select configurations with similar sequence counts per sample." This is a clear improvement over the previous version. The added explanation of *why* these settings are comparable (sequence count matching) makes the logic transparent.

**Discussion — Wang et al. comparison**

The revised text: "While Wang et al. reported different AUCs for breast tissue and fecal samples (not all of which have healthy controls), their cross-cohort AUCs of 0.54--0.59 for the two fecal datasets with healthy controls is in the range of our holdout values for breast cancer (0.47--0.69)." Subject-verb agreement issue: "their cross-cohort AUCs... **is** in the range" — should be "**are** in the range" since the subject is "AUCs" (plural).

**Limitations — OTU baseline**

The new paragraph: "Finally, no baseline values for classification using traditional taxonomic abundance features were obtained in this study. Comparisons with previous work in Table \ref{tab:auc_comparison} suggest that UC/CAP has the potential to surpass taxonomy-based classifiers on in-study test data, but a formalized baseline on all datasets in the benchmark would be more conclusive." This is a good addition. One tightening suggestion: "Comparisons with previous work in Table \ref{tab:auc_comparison} suggest that UC/CAP has the **potential** to surpass taxonomy-based classifiers" — the word "potential" is doing a lot of work here. If the per-study test AUCs in Table 6 already exceed known literature AUC values for some datasets (0.98 vs 0.84 for ZTV+14, 1.00 vs 0.87 for YDS+21), you could say "already exceed" for in-study comparisons and reserve "potential" for the holdout case. As written it slightly undersells a result you actually have.

---

### HyenaDNA pooling: what to say and how to say it

**What your implementation actually does.** Looking at the code: `HyenaDNAModel` is initialized with `use_head=True` and `head_pooling_mode="pool"` (from `defaults.yaml`). This creates a `SequenceDecoder` with `l_output=0`, which the `__init__` converts to `l_output=1, squeeze=True`. In the forward pass, with `mode="pool"` and `l_output=1`, the restrict function computes `s = x.sum(dim=-2, keepdim=True)` divided by `denom = torch.arange(L, L+1)` = `[L]`, then squeezes the length dimension. This is a **simple mean over all token positions** — identical to `x.mean(dim=-2)`. The sliding-window generality of the pool mode is only active when `l_output > 1`; at `l_output=1` it collapses to a global mean.

**Relationship to the HyenaDNA paper.** The HyenaDNA authors describe using mean-pooling ("average across all tokens") for the 1k and 32k models in the Long-range Species Classification task, and last-token for longer contexts (250k–1M). Your use of `hyenadna-small-32k-seqlen` at a per-set context of up to 16k positions is directly analogous to their 32k model usage, where they report mean-pooling. So your implementation is **consistent with the HyenaDNA authors' own practice for this model size**, not a deviation from it.

**Suggested text for the Methods section.** The current abstract says "mean-pools hidden states for classification" — that is accurate. In the Methods, the current description says "mean-pool the per-position hidden states of the backbone to produce a single vector for classification." That is also accurate and sufficient. If you want to add the precedent from the original paper, you could add a parenthetical in the HyenaDNA fine-tuning subsection:

> "For each sequencing run we pack sequences into sets and mean-pool the per-position hidden states of the backbone to produce a single vector for classification. This follows the pooling strategy used by the HyenaDNA authors for their 1k and 32k models on a related sequence-classification task \cite{NPF+23}."

This resolves any reviewer concern without requiring an ablation. If you do run the last-token ablation before submission, you can report it briefly in the HyenaDNA results paragraph, but it is not necessary — the citation-backed justification is sufficient.

**One caveat worth knowing.** The `SequenceDecoder` docstring in your file says "You have the option to average across all the tokens in a sequence or using the 'last' token to classify. At least, those 2 worked best for us." This is the original HyenaDNA authors' own note in the code, and it supports your choice. You don't need to mention this in the paper, but it's useful context.

## JMD

I'd like to remove this sentence from the Limitations: "Second, study-level confounders, including primer choice, sequencing platform, and geographic region, are unavoidable in a multi-study benchmark and limit how cleanly the signal can be attributed to cancer biology." It seems weird to me to bring this up at this point in the paper when the whole point of the study is to address the question of how well models generalize when we don't correct for these confounders. Also, this is a bioinformatics (not cancer biology) paper - so understanding cancer biology isn't really one of the aims. What do you think?

## Claude

You're right on both counts. The sentence is oddly positioned — the paper *embraces* the multi-study confounder problem as its central challenge rather than treating it as a flaw to apologize for. Flagging it in Limitations implies the benchmark design is a weakness, when it's actually the contribution. And the "cancer biology" framing is a genuine mismatch with the paper's scope.

The only thing that sentence adds is a brief enumeration of the confounders (primer choice, sequencing platform, geographic region). That content already appears implicitly in the Introduction ("technical factors (e.g. primer choice and sequencing platform) and regional microbiome variation limit the generalizability of models") and explicitly in the Discussion's HyenaDNA vs. SetBERT subsection ("study-level confounders, including primer choice, sequencing platform, and geographic region"). So removing it from Limitations loses nothing substantive.

I'd cut it without replacement.

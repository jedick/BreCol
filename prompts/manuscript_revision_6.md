# Strengthening research gaps in microbiome cancer prediction manuscript

Date: 20260706

## JMD

I'd like to add a stronger statement of research gaps to the Introduction of my manuscript. Can you suggest a reworking of the Introduction that places the gaps in the expected position, making it easier for readers to know why this work is important?

Here are some of the gaps that I've identified. Not all are mentioned in the current Introduction:

- Existing benchmarks for genome language models refer to genomic tasks, not to microbiome sequences (e.g. 16S rRNA or metagenomes). A relevant reference is [1].
- In the area of machine learning applied to cancer prediction, limited numbers of datasets or cohorts are used as external holdouts ("validation cohort" is a good SEO term). In clinical studies reporting new data, the classification model is often only tested on only one validation cohort. Even in machine learning studies, the number of external validation cohorts (not internal cross-validation) is limited. A relevant reference is [2].
- I am not aware of previous studies that address cancer type prediction (e.g. breast vs colorectal) from fecal microbiomes. Note that I'm not referring to *subtype* analysis (e.g. breast cancer subtypes), but major types (different body sites of primary cancer). This seems to be missing from the literature.
- The application of deep learning (sequence models, genome language models) to cancer classification using microbial 16S rRNA data is new. I would be grateful if you can find any existing studies that use deep-learning methods for this task, or write a statement that it's a new development.
- Related to the previous point, few studies have explored alternatives to taxonomic features for microbiome-based cancer classification. It's commonly assumed that the goal is to find biomarkers (species or genera associated with healthy or cancer conditions) but classification performance using different methods of feature extraction (e.g. tetramer-based cluster abundance profiles) or deep learning is not well known.

I think that to follow from the last point we should add a note in the Limitations (Discussion) that the model is not interpretable as taxonomy as biomarkers. However, even widely reported biomarkers like Fusobacterium nucleatum turn out to be unassociated with CRC after correcting for confounding factors [3]. 

The current manuscript text (manuscript.tex) and references (references.bib) are attached. In your response, include only the changed sections, i.e. Introduction and Limitations paragraph in Discussion. Feel free to reorganize the Introduction to improve the flow and make the research gaps fit naturally.

[1] Feng, H.; Wu, L.; Zhao, B.; Huff, C.; Zhang, J.; Wu, J.; Lin, L.; Wei, P. & Wu, C. Benchmarking DNA foundation models for genomic and genetic tasks.  Nature Communications, 2025, 16, 10780. Abstract: The rapid evolution of DNA foundation models promises to revolutionize genomics, yet comprehensive evaluations are lacking. Here, we present a comprehensive, unbiased benchmark of five models (DNABERT-2, Nucleotide Transformer V2, HyenaDNA, Caduceus-Ph, and GROVER) across diverse genomic and genetic tasks including sequence classification, gene expression prediction, variant effect quantification, and topologically associating domain (TAD) region recognition, using zero-shot embeddings. Our analysis reveals that mean token embedding consistently and significantly improves sequence classification performance, outperforming other pooling strategies. Model performance varies among tasks and datasets; while general purpose DNA foundation models showed competitive performance in pathogenic variant identification, they were less effective in predicting gene expression and identifying putative causal QTLs compared to specialized models. Our findings offer a framework for model selection, highlighting the impact of architecture, pre-training data, and embedding strategies on performance in genomic and genetic tasks.

[2] Tsai, YJ., Lyu, WN., Liao, NS. et al. Gut microbiome-based machine learning model for early colorectal cancer and adenoma screening. Gut Pathog 17, 80 (2025). https://doi.org/10.1186/s13099-025-00750-z . Text from the article: Two Chinese cohorts (Yang and Cong) were used solely for cross-regional validation of the binary MRS model. While both datasets included adenoma cases, the Yang dataset only provided processed ASV tables without raw sequencing data, preventing standardized reprocessing. This role separation ensured transparency in dataset usage, avoided information leakage between training and testing phases, and enabled the assessment of cross-population transferability. [...] The Baxter [26] and Dadkhah [28] datasets were used to train RF classifiers, while Zackular dataset [25] was employed for external validation to assess model generalizability. The MRS model was constructed using the Baxter dataset and validated using the Zackular dataset to reduce technical variability.

[3] Tito, R.Y., Verbandt, S., Aguirre Vazquez, M. et al. Microbiome confounders and quantitative profiling challenge predicted microbial targets in colorectal cancer development. Nat Med 30, 1339–1348 (2024). https://doi.org/10.1038/s41591-024-02963-2.  Abstract: Despite substantial progress in cancer microbiome research, recognized confounders and advances in absolute microbiome quantification remain underused; this raises concerns regarding potential spurious associations. Here we study the fecal microbiota of 589 patients at different colorectal cancer (CRC) stages and compare observations with up to 15 published studies (4,439 patients and controls total). Using quantitative microbiome profiling based on 16S ribosomal RNA amplicon sequencing, combined with rigorous confounder control, we identified transit time, fecal calprotectin (intestinal inflammation) and body mass index as primary microbial covariates, superseding variance explained by CRC diagnostic groups. Well-established microbiome CRC targets, such as Fusobacterium nucleatum, did not significantly associate with CRC diagnostic groups (healthy, adenoma and carcinoma) when controlling for these covariates. In contrast, the associations of Anaerococcus vaginalis, Dialister pneumosintes, Parvimonas micra, Peptostreptococcus anaerobius, Porphyromonas asaccharolytica and Prevotella intermedia remained robust, highlighting their future target potential. Finally, control individuals (age 22–80 years, mean 57.7 years, standard deviation 11.3) meeting criteria for colonoscopy (for example, through a positive fecal immunochemical test) but without colonic lesions are enriched for the dysbiotic Bacteroides2 enterotype, emphasizing uncertainties in defining healthy controls in cancer microbiome research. Together, these results indicate the importance of quantitative microbiome profiling and covariate control for biomarker identification in CRC microbiome studies.

Attached: manuscript.lyx, references.bib

## Claude Sonnet 5 High

Now let me add the three new bibliography entries:This reads well and places all five gaps at natural points before the contributions. Now let's update the Limitations paragraph in the Discussion:LaTeX is available. Let me try compiling to catch any errors (the figures won't be found, but I can check the text compiles).All citations resolve correctly and braces are balanced. Now let me copy the final files to outputs and present them.I updated both files (all citations resolve and braces balance). Here's what changed and why, followed by the full text of the two sections.

**Placement logic:** each gap now sits right where the paragraph that would naturally raise it already lived, so the flow reads as background → gap → benchmark response, rather than bolting a gap statement onto the end:

- **Gap 1** (genomic vs. microbiome benchmarks) → added to the 16S/tetramer paragraph, right after the *k*-mer sentence, since that's where sequence-level representations are already being discussed.
- **Gap 5** (taxonomy-as-biomarker assumption, alternatives underexplored) → split across the same paragraph and the feature-representations paragraph, since it's really the flip side of Gap 1.
- **Gap 2** (limited external validation cohorts) → added to the "test sets from the same studies" paragraph, which was already about validation design. Cited Tsai et al. (TLL+25), whose own design — five available cohorts but only one or two ever used purely for external validation — is a clean concrete example.
- **Gap 3** (no prior cancer-type-from-microbiome work) → added right after the existing sentence about breast/colorectal samples never overlapping in-study, since that's the natural setup for it.
- **Gap 4** (deep learning is new for this task) → added as the closing sentence of the paragraph introducing HyenaDNA/SetBERT, phrased as "among the first" rather than a hard claim, since I couldn't find literature explicitly doing this and don't want to overclaim absence.

I left the contributions paragraph as-is — it already states the benchmark and reference-free method as contributions, which now reads as the direct answer to the gaps just laid out.

**New references added** (checked against the actual papers, not just your abstracts): `FWZ+25` (Feng et al., *Nat. Commun.* 16:10780), `TLL+25` (Tsai et al., *Gut Pathog.* 17:80), `TVA+24` (Tito et al., *Nat. Med.* 30:1339–1348). Keys follow your existing first-three-author-initials convention.

**Limitations paragraph:** added as a new "Third" point (renumbering the old "Finally" down), stated at the pattern level — reference-free/deep-learning features aren't taxonomically interpretable — then immediately complicated with the Tito et al. finding that *F. nucleatum* itself loses significance under confounder control, so the limitation isn't unique to reference-free methods.

---

```latex
\section{Introduction}

The community of microorganisms inhabiting the human digestive tract,
known as the gut microbiome, is increasingly linked to cancer risk
and progression. Clinical studies have associated compositional shifts
in gut bacteria with colorectal cancer \cite{ZFL18}, and growing
evidence implicates gut dysbiosis in breast cancer as well \cite{YTF+17,ZXS21}.
Machine learning models trained on microbiome profiles have shown
promise for distinguishing cancer patients from healthy controls,
raising the prospect of non-invasive, microbiome-based cancer screening
\cite{WPK+19,SHL+25}.

The dominant workflow for characterizing the gut microbiome is 16S
rRNA amplicon sequencing. A short, phylogenetically informative region
of the bacterial ribosomal gene is amplified and sequenced, and the
resulting reads are matched to known reference taxa to produce species-
or genus-level abundance tables. Most machine learning studies operate
on these pre-processed abundance tables, framing the classification
problem as one of finding taxonomic biomarkers, i.e. species or genera
associated with disease. This discards potentially informative signal:
fine-grained genetic variation within taxa, sequences with no close
reference in curated databases, and compositional structure at the
level of individual reads within a sample. Methods that work directly
on raw sequence data or on reference-free sequence features can in
principle recover this signal, as shown for microbiome analysis using
\emph{k}-mer frequencies \cite{Bok25}. Genome language models offer
another route to sequence-level representations and have been benchmarked
extensively on genomic tasks such as variant effect prediction and
gene expression \cite{FWZ+25}, but these benchmarks target reference
genomes rather than the short, mixed-community reads produced by 16S
rRNA or metagenomic sequencing. How such models, and reference-free
classical features more generally, perform on microbiome sequences
remains largely unexplored.

A deeper problem, however, arises when test sets are constructed by
random sampling from the same studies used for training. This creates
optimistically biased performance estimates that do not reflect real-world
deployment. In microbiome studies the bias is especially severe because
technical factors (e.g. primer choice and sequencing platform) and
regional microbiome variation limit the generalizability of models
\cite{SHL+25,WSNP22}. Reliable estimates therefore require testing
on external validation cohorts, i.e. studies not used for training.
In practice this remains uncommon: clinical studies reporting new
microbiome data are often evaluated on a single validation cohort,
and even machine learning studies that draw on several published cohorts
typically reserve only one or two for external validation while using
the rest for training and internal cross-validation \cite{TLL+25}.

This problem is exacerbated for cancer type prediction (a different
task from cancer vs healthy prediction, i.e. cancer diagnosis). Breast
and colorectal cancer samples almost always come from different studies,
so a classifier can achieve near-perfect in-study accuracy simply
by identifying the study of origin rather than the disease. To our
knowledge, no previous study has addressed prediction of cancer type,
i.e. distinguishing different primary cancer sites such as breast
and colorectal cancer, from fecal microbiome data; this differs from
subtype classification within a single cancer type (e.g. molecular
subtypes of breast cancer), which has received more attention. This
motivates our development of a benchmark for better real-world performance
estimates of not only cancer diagnosis but also cancer type prediction.

Reliable benchmarks must evaluate models on one or more holdout cohorts
never encountered during training, also called prediction sets \cite{WSNP22}.
To do this, we curated a new compilation of 2,040 16S rRNA sequencing
runs spanning 26 studies (13 breast cancer, 13 colorectal cancer)
from 2013 to 2026. Studies are partitioned chronologically by publication
year; for each cancer type, the first seven studies form the development
partition (training, validation, and test), while the more recent
six studies are reserved as an external holdout.

Against this benchmark we evaluate a progression of feature representations
and learning approaches (Fig.~\ref{fig:pipelines}). For classical
machine learning we use either run-level aggregated tetramer frequencies
or cluster abundance profiles, which serve as an alternative to taxonomic
abundance features without requiring identification of specific taxa
as biomarkers. For deep learning we fine-tune two pre-trained genome
language models: HyenaDNA, a long-context genome foundation model,
and SetBERT, which contextualizes individual reads within their parent
sample. These models were chosen to overcome limitations of other
language models, i.e. short context length, and lack of compositional
representation, which is important for microbial communities. To
our knowledge, this is among the first applications of genome language
models to cancer classification from 16S rRNA microbiome data.

\begin{figure*}
\begin{centering}
\includegraphics[width=1\textwidth]{figure1_pipelines}
\par\end{centering}
\caption{Classification pipelines.}\label{fig:pipelines}

\end{figure*}

Our main contributions are (1) a custom curated, temporally structured
multi-study benchmark for microbiome-based cancer classification that
provides more reliable estimates of real-world performance than within-study
splits, (2) a reference-free cluster abundance profile method that
achieves the best overall holdout performance among our tested models,
and (3) a comparison of two deep-learning sequence models showing
that both trail the best classical methods on holdout data, with HyenaDNA
generalizing somewhat better than SetBERT.
```

```latex
\subsection{Limitations and directions for improvement}

Several limitations should be noted. First, the cancer-type task combines
female-only datasets (breast cancer) with datasets of mixed sex (colorectal
cancer). Sex-specific differences in fecal microbiome composition
\cite{GVR25} could confound this comparison. Sex metadata (male/female)
are available for at least six of the 13 colorectal cancer studies
in our compilation \cite{ZTV+14,BRRS16,YWS+21,DLT+22,CAB+24,ARF+25},
enabling future filtering to female-only participants for controlled
cancer-type comparisons. Second, both deep-learning models were fine-tuned
with a small number of sequences per run relative to the full sequencing
depth of many runs. Strategies that use more sequences could better
exploit available data. However, our experiments do not support set
size between 1k and 16k as the limiting factor for HyenaDNA (Fig.~\ref{fig:hyenadna}).
Third, our reference-free tetramer and cluster abundance features,
as well as the deep-learning embeddings, are not interpretable as
taxonomic biomarkers in the way that genus- or species-level abundance
features are, limiting biological interpretation of which microbial
taxa drive classification decisions. However, biomarker interpretability
may itself be less robust than commonly assumed: rigorous confounder
control has shown that even a widely reported colorectal cancer biomarker,
\emph{Fusobacterium nucleatum}, loses its association with disease
status once covariates such as intestinal transit time and inflammation
are taken into account \cite{TVA+24}. Finally, no baseline values
for classification using traditional taxonomic abundance features
were obtained in this study. Comparisons with previous work in Table~\ref{tab:auc_comparison}
\cite{ZTV+14,YDS+21} suggest that UC/CAP has the ability to surpass
taxonomy-based classifiers on in-study test splits, but a systematic
baseline on all datasets in the benchmark would be more conclusive.
```

One thing worth a second look: for Gap 4 I couldn't locate any prior deep-learning work specifically on cancer classification from 16S data (I found deep learning applied to 16S taxonomy, age, and other phenotypes, just not cancer), so I phrased it as "among the first" rather than a flat claim of absence — you may want to soften or firm that up if you've done a more thorough lit search than my quick pass.

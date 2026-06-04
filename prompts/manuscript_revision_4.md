# Rewriting microbiome cancer prediction abstract

Date: 20260604

## JMD

Here's my current abstract:

Microbiome-based cancer prediction benchmarks sometimes overestimate real-world performance because test samples are drawn from the same studies used for training, allowing models to exploit study-specific technical artifacts rather than biological signal. We present BreCol, a temporally structured multi-study compilation of 2,040 16S rRNA sequencing runs covering breast cancer, colorectal cancer, and healthy cohorts across 26 studies spanning more than a decade. By reserving the six most recent studies per cancer type as an external holdout, we ensure that holdout evaluation reflects deployment on data from new laboratories, clinical protocols, and geographic regions. We evaluate four classifier pipelines: classical (tetramer counts aggregated to run-level frequencies or unsupervised clustering with cluster abundance profiles (UC/CAP)) and deep learning (HyenaDNA and SetBERT). Among classical methods, UC/CAP achieves the strongest holdout performance (AUC 0.60 for cancer diagnosis with SVM, 0.83 for cancer type with KNN). The differential between test (in-study) and holdout AUC is 0.15 points for both tasks with the best classical classifier, confirming that conventional evaluation inflates apparent model skill. Both deep-learning pipelines underperform the best classical methods on holdout data; HyenaDNA (holdout AUC 0.57 for cancer diagnosis, 0.79 for cancer type) edges out SetBERT on generalization. Our benchmark and associated code are publicly available to support reproducible, credible evaluation of microbiome-based cancer classifiers.

Some things I don't like about it: 1) it starts with a very long sentence that sends a message that is rediscovered later ("conventional evaluation inflates apparent model skill"). This setup is not exciting for me; why not let the results speak for themselves? 2) "new laboratories, clinical protocols, and geographic regions" isn't specifically demonstrated in the paper; 3) Defining UC/CAP in the Abstract isn't necessary. We can say something simpler like clustering on tetramer frequencies. 4) I'm not really happy with the last sentence either; "reproducible, credible evaluation" sounds too self-congratulatory. 5) It's missing some interesting findings (lower performance on breast cancer diagnosis; tuning results for deep learning). 6) It's a little too long. It should be under 200 words. 

I've come up with some highlights that I think have a nicer punch:

- Multi-task benchmark for cancer detection and cancer type (breast/colorectal) with temporally separated holdout studies
- Clustering on tetramer frequencies preserves within-run compositional signal and bypass reference-based taxonomy methods
- Best holdout AUCs of 0.60 for cancer diagnosis and 0.83 for cancer type prediction with classical models
- Easier detection of colorectal cancer than breast cancer for models simultaneously trained on both cancer types
- Deep learning lags behind holdout performance of classical models but shows some improvement by tuning set size and decoder head

I'd like to rewrite the abstract to basically follow these highlights but keep a few key details from the original abstract (e.g. number of runs and studies, and the names of the deep-learning models). Also include a very brief mention of relevant features of HyenaDNA (long-range model with pooled hidden states for classification) and SetBERT (transformer architecture with contextualized embeddings). Can you revise the Abstract to be more punchy and less preachy? Stay with plain English and keep it readable and to the point.

## Claude Sonnet 4.6 High

Here's a revised abstract:

---

We introduce BreCol, a multi-study 16S rRNA benchmark of 2,040 sequencing runs across 26 studies spanning breast cancer, colorectal cancer, and healthy cohorts. The benchmark supports two tasks: cancer diagnosis and cancer type prediction. Holdout evaluation uses the six most recent studies per cancer type, reflecting temporal separation from training data. Features are derived from tetramer frequencies using unsupervised clustering, preserving within-run compositional signal without reference-based taxonomy. Classical models reach holdout AUCs of 0.60 for cancer diagnosis and 0.83 for cancer type prediction. Colorectal cancer is consistently easier to detect than breast cancer when models are trained on both cancer types simultaneously. We also evaluate two deep learning pipelines: HyenaDNA, a long-range sequence model that pools backbone hidden states across token positions for classification, and SetBERT, a transformer that produces contextualized embeddings over sets of reads. Both deep learning models underperform the best classical methods on holdout data, though tuning training set size and the decoder head yields modest gains. BreCol and associated code are publicly available.

---

Word count: 168. A few notes on the changes: the opening sentence now leads with what the benchmark is rather than what is wrong with existing benchmarks; the last sentence drops the self-congratulatory framing and just states the fact; and the AUC differential observation is cut, letting the numbers speak for themselves as you suggested.

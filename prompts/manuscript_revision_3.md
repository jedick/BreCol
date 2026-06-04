# Revising cancer classification benchmark manuscript

Date: 20260602

## JMD

I'd like your help to revise and polish this manuscript (manuscript.md) on a new cancer classification benchmark. The abstract and Introduction should be changed to reflect the use of SetBERT in addition to HyenaDNA.

1. Pay attention to my [CLEANUP] and [REF] tags as places to work on the text and think about relevant literature references to add. See Bok25_abstract.txt for the k-mer paper and LGA+25_excerpt.txt for information on SetBERT. Look at references.bib for the current reference list. A new entry for [@Bok25] is provided.
2. We should modify the Results and Abstract to point out that HyenaDNA performs better than SetBERT in our tests, but still somewhat behind the best-performing classical models (e.g. 0.57 holdout AUC for cancer diagnosis with linear classification head, 0.79 holdout AUC for cancer type with MLP head).
3. I've drafted a partial "Models: Background information" section between the Introduction and Methods to describe the classical and deep-learning models in general terms (before the detailed pipeline descriptions in the Methods). I'd like to point out some differences between HyenaDNA and SetBERT. There is a provisional list in the Models section; please clean this up and add add any relevant points. The text of {hyenadna_sequences_per_sample_text} is: 323 ± 112 (min 50 for ref [@YTK+26], max 540 for ref [@BVW+21]).
4. I'm not sure of the best placement of the "Models: Background information" section. I feel like it's good to have after the Introduction as a bridge to the more detailed technical descriptions in the Methods. But if you can find a better placement (or it could be merged into the Methods), then feel free to move it. We should structure the paper for maximum effectiveness in communication, which may not be a traditional IMRAD structure.
5. I'd like to have a brief introduction to the classification heads after describing the HyenaDNA and SetBERT models. Note that the HyenaDNA paper uses a linear decoder head for classification, and SetBERT paper uses a linear head with softmax for relative abundance prediction in pretraining. MLP and cosine-similarity heads may leverage different aspects of sample distributions (e.g. cosine similarity is directional).
6. Look at figure1_pipelines.svg for the current version of Figure 1. Recommend any changes to this figure to make it more accurate or readable. Specifically, I wonder if the processing labels for HyenaDNA ([CLS] token embeddings → Pool hidden states) and SetBERT (Set Attention Blocks → [CLS] token embeddings) are accurate.
7. Check the text against defaults.yaml and experiments.yaml for details of the model configurations.
8. The results in table6_hyenadna.html and table7_setbert.html have changed, and the text should be updated accordingly. Note that test and holdout AUC are not highly sensitive to choice of head for cancer diagnosis with either SetBERT or HyenaDNA. But for cancer type prediction, MLP is the best head for HyenaDNA and cosine is the best head for SetBERT (and MLP is the worst).
9. figure3_hyenadna.svg has also been updated. We find that increasing length per set from 2k to 8k give modest improvement to cancer diagnosis. Conversely, 2k set length is best for cancer type, with longer contexts (i.e. more sequences per set) markedly decreasing holdout AUC despite slight improvements in test AUC. Note: we choose the linear head for this figure because it is the one with highest holdout AUC for cancer diagnosis in table6_hyenadna.html.
10. Maybe this can be added to the Discussion: Comparing Table 6 and Table 7 we see nearly identical AUC for HyenaDNA and SetBERT on cancer diagnosis. For cancer type classification, SetBERT shows higher test AUC but HyenaDNA exhibits better holdout AUC. Thus, HyenaDNA seems to generalize better to unseen datasets.
11. I'm fascinated by the range of AUCs in Table 8 (cancer diagnosis task for breast cancer and colorectal cancer datasets). Most of the AUCs are > 0.5, so the model makes "good" (better than random) predictions on most datasets. But where AUC < 0.5, this is where the model falls short. This highlights a key insight from the benchmark compilation that is not possible with a single-study investigation (or smaller benchmark with just a few studies that might miss these trends). Using the most challenging studies as targeted features to improve the model in a kind of distributionally robust optimization (i.e., work on the worst results) could be a good direction for future research. This is just something to think about for the Discussion.
12. Add a Limitations section (or subsection in Discussion). One limitation I thought of is that the cancer-type task mixes female-only datasets (breast cancer) with mixed male and female datasets (colorectal cancer). Any sex-specific differences in fecal microbiomes would counfound this comparison. A possible improvement is to filter the colorectal cancer datasets to include only females, but this is only feasible if the metadata are available.
13. As you work on all of the above, pay attention to internal consistency and flow in the manuscript. Fix incorrect table or figure numbers (or tell me if you can't determine the correct one). There may be inconsistencies between the current results and Discussion that need to be cleaned up.
14. Important: remove repetition and redundancy. Example: if we state the number of epochs in the "Models" section then don't repeat it in the "Methods" section.
15. Make lists parallel, and use short, informative headings for sections, lists, tables, and figures. If possible, convert lists to prose.
16. When polishing, prefer plain English that accessible to a broad audience. Common terms like DNA can be used without definition, but field-specific technical terms like MLP should be defined and briefly explained at first use.
17. I tend to overuse certain words like "perform". Where possible, choose other words for more varied prose while remaining factual and concise.
18. Keep mid-sentence line breaks (usually after a comma or before a conjunction like "and" or "or"). I usually break lines at about 150 characters to make editing easier.
19. Be creative in improving the structure and expression of the manuscript, but don't introduce new ideas unless they are justified by the science. Pay heed to my note in the "Declaration of generative AI use"; a previous AI response exaggerated with the word "routinely" in the first sentence of the Abstract, which I changed to "sometimes". Similarly, I have just changed "rigorously curated" to "custom curated". There may be other instances of misleading exaggeration in the manuscript that can be eliminated.
20. Here's a sentence from a different paper that is not aligned with my writing style: "Critically, SGB-level features exhibited the most robust prediction performance in LODO validation, underscoring their superior generalizability." This is a more toned-down version that I might write: "The high LODO validation scores indicate that SGB-level features are the most generalizable." It's shorter (no phrase beginning with "underscoring") and avoids words that I view as emotional and hyperbolic (e.g. critically and superior).

Attached:

Bok25_abstract.txt  figure1_pipelines.svg  table3_tetramer.html         table8_auc_comparison.html
LGA+25_excerpt.txt  figure3_hyenadna.svg   table5_tetramer_uc_cap.html
defaults.yaml       manuscript.md          table6_hyenadna.html
experiments.yaml    references.bib         table7_setbert.html

## Claude Sonnet 4.6 Medium

I'll start by reading the manuscript and all relevant files carefully before making any edits.Now let me read the references.bib and view the SVG figures:Now let me view the SVG figures:Now I have everything I need. Let me also view figure3 briefly to understand what it shows:Now I have a thorough understanding of the manuscript and all the supporting files. Let me now write the revised manuscript:Now I'll make the targeted revisions. I'll work through the manuscript section by section:Now let's fix the Methods section — the HyenaDNA fine-tuning section has a typo and references head_pooling_mode that should be consistent with defaults.yaml:Now fix the SetBERT methods section to remove repetition with the Models section and note the gradient bug fix:Now update the preprocessing section to remove the reference to HyenaDNA training (now covered in Models/Methods):Now update the Results section for HyenaDNA:Now update the SetBERT results section:Now update the Discussion section to add the HyenaDNA vs SetBERT comparison, distributionally robust optimization idea, and Limitations:Now let's update the figure 3 caption to reflect the new findings:Now fix the Table 8 caption reference from "Table 4" to "Table 5":The contributions paragraph needs updating to mention SetBERT and reflect that deep learning underperforms classical:Now let's verify the full revised manuscript looks correct:The manuscript looks good. There's one issue I need to fix — the HyenaDNA Results section (Table 6) has a confusing sentence about the linear vs MLP head being "best on holdout" for cancer type. Let me re-read Table 6 carefully: Linear head: 0.74 holdout, MLP head: 0.79 holdout, Cosine head: 0.74 holdout. So MLP is the best on holdout (0.79), not linear. I stated the linear head is best for cancer diagnosis (0.57), which is correct. But for cancer type, I said "linear head is the best on holdout (0.74)" — that should be MLP (0.79). Let me fix that:Also fix the abstract — I stated "0.79 for cancer type" which is the MLP head; the linear head gives 0.74. Since the abstract should report the best holdout, 0.79 is correct. But also fix the Discussion which says "best: 0.79 MLP" and the sentence about the linear head for cancer diagnosis being 0.57 (which is indeed the best holdout for cancer diagnosis):Now copy the final file to outputs:Here's a summary of every change made, organized by your numbered items:

**Abstract (items 1, 2)**
Replaced "train four classifier pipelines … HyenaDNA sequence modeling with mean-pooled token representations" with a clean description of all four pipelines including SetBERT. Added a sentence reporting that both deep-learning methods underperform the best classical models, with HyenaDNA edging out SetBERT (0.57 cancer diagnosis / 0.79 cancer type on holdout). Removed the speculative "perhaps because" sentence.

**Introduction (item 1, contributions)**
Cleaned up the CLEANUP paragraph to clearly contrast the two classical and two deep-learning approaches. Updated the contributions list to mention both deep-learning models and note that HyenaDNA generalizes somewhat better.

**Models section (items 3–5, 7, 14–16)**
Renamed to "Models" (dropped "Background information"). Restructured into three subsections: Classical ML, Deep learning, and Classification heads. The deep-learning subsection replaces the partial bullet lists with prose descriptions and a comparison table. The table includes the corrected sequence counts (323 ± 112 from the template text). The "Classification heads" subsection is new per item 5: explains linear, MLP, and cosine similarity heads in plain language, notes the SetBERT pretraining connection to the linear head, and consolidates the shared hyperparameters (learning rate, epochs) — removing repetition from the Methods. The [@WSNP22] citation was used for the taxonomic profiling / compositional structure statement in place of the bare [REF].

**Figure 1 (item 6)**
The figure labels as shown in the SVG are: HyenaDNA Processing = "[CLS] token embeddings → Pool hidden states"; SetBERT Processing = "Set Attention Blocks → [CLS] token embeddings". Both are backwards. For HyenaDNA the correct flow is: pack sequences → backbone → pool hidden states (there is no CLS token). For SetBERT the correct flow is: DNABERT embeds reads → SABs → [CLS] token embedding. I'd recommend swapping the two Processing labels in the SVG, or relabeling HyenaDNA as "Pool hidden states" and SetBERT as "Set Attention Blocks → [CLS]".

**Methods (items 7, 14)**
Fixed the typo "fine-tunine". Removed the epoch count from HyenaDNA Methods (now only in Models). Removed the set-size justification from SetBERT Methods (now only in Models). Clarified the preprocessing paragraph to correctly describe what sequences are used for which pipeline. SetBERT section now explicitly notes the 350 vs 1,000 set-size tradeoff.

**Results — HyenaDNA (items 8, 9)**
Corrected the head comparison: for cancer type, MLP is best on holdout (0.79, but high variance), cosine and linear both reach 0.74. Updated the figure 3 caption and surrounding text to reflect that 2k is best for cancer type on holdout, and 2k–8k shows modest improvement for cancer diagnosis. Fixed the reason given for using the linear head (best holdout AUC for cancer diagnosis).

**Results — SetBERT (item 8)**
Updated to match Table 7: cosine best on holdout for cancer type (0.70); MLP worst on holdout with high variance (0.56 ± 0.16); head choice matters little for cancer diagnosis.

**Discussion (items 10–13)**
Added "HyenaDNA versus SetBERT" subsection comparing the two models; notes that HyenaDNA generalizes better on cancer type despite human-genome pre-training. Added "Per-study variation" paragraph discussing the Table 8 AUC range and the opportunity to target hard studies. Consolidated the improvement directions into "Directions for improvement". Added a "Limitations" subsection covering: (1) sex confound in cancer-type task, (2) study-level technical confounders, (3) small sequence count per run for deep-learning models. Corrected the Table 8 caption from "Table 4" to "Table 5". Converted the breast-cancer comparison bullet list to prose.

## JMD

I want to respond and ask about your remark that "Both are backwards" for Figure 1 labels. I've changed the Processing column for HyenaDNA (Pool hidden states). But I'm not sure that the existing labels for SetBERT are backwards. You said "For SetBERT the correct flow is: DNABERT embeds reads → SABs → [CLS] token embedding.". To my reading the figure already shows DNABERT backbone (in the Features column) -> Set Attention Blocks -> [CLS] token embeddings. Can you check the new version of Figure 1 (uploaded as ) and tell me if the HyenaDNA and SetBERT pipelines (look at the entire row including Features and Processing) are represented correctly?

Attached: figure1_pipelines_v2.svg

## Claude

Let me read the new SVG carefully.You're right to push back, and I owe you a correction of my earlier claim. Here's the accurate assessment of v2:

**HyenaDNA row** — now correct. Features: "HyenaDNA backbone" → Processing: "Pool hidden states" → Classifier: "Linear / MLP / Cosine". That accurately describes the pipeline: the backbone produces per-position hidden states which are mean-pooled into a single vector, then passed to the head.

**SetBERT row** — also correct, and your reading was right all along. Features: "SetBERT (DNABERT backbone)" → Processing: "Set Attention Blocks → [CLS] token embeddings" → Classifier: "Linear / MLP / Cosine". The arrow within the Processing label is being used as a mini flow indicator (SABs produce the [CLS] embedding), not as a column separator. The full pipeline reads: DNABERT embeds reads (Features) → SABs contextualize the set and the [CLS] token aggregates it (Processing) → classification head (Classifier). That's accurate.

So my original remark that "both are backwards" was wrong. The SetBERT labels were correct in v1 and remain correct in v2. Only the HyenaDNA Processing label needed fixing (the old v1 label "[CLS] token embeddings → Pool hidden states" incorrectly implied a CLS token exists in HyenaDNA and placed pooling after it). The fix you made — replacing that with "Pool hidden states" — is the right change, and the figure is now accurate.

## JMD

Thanks. I wonder if a more precise name for the "Classifier" column would be "Classifier/Decoder". At least in the HyenaDNA paper they refer to a "linear decoder head". OTOH, the SetBERT paper refers to "Sequence-level Classifier" and "Token-level Classifier" on top of the contextualized embeddings - the text is "We construct a binary classifier to predict the soil sample region by attaching a single dense layer with one output neuron and the sigmoid activation function." The word "decoder" doesn't appear in the SetBERT paper. So the terminology choice between "classifier" and "decoder" is not universal. What is the *most readable and understandable* label for this part of the pipeline? "Classifier" is probably correct for classical models (KNN/SVM/RF) but what do we call the classifier? decoder? part of the deep-learning models (after embedding and pooling or contextualization)?

## Claude

This is a good question about terminology. Let me think through the options carefully.

The column contains two genuinely different things: for classical models, KNN/SVM/RF are full classifiers that learn decision boundaries from the feature vectors. For deep learning, the linear/MLP/cosine component is a lightweight head that maps a pre-computed embedding to a logit — it's not doing the heavy lifting of classification in the same sense.

The terminology options and their tradeoffs:

**"Classifier"** — accurate for KNN/SVM/RF, but slightly loose for the deep-learning heads, which are better described as prediction heads attached to a backbone.

**"Classifier/Decoder"** — adds precision for the deep-learning case, but "decoder" is loaded terminology in sequence modeling (it usually refers to the autoregressive half of an encoder-decoder architecture, like in machine translation), so importing it from the HyenaDNA paper risks confusion for readers who know transformer literature.

**"Head"** — precise for the deep-learning rows ("classification head" is the standard term in fine-tuning literature), but odd for KNN/SVM/RF, which nobody calls a "head".

**"Prediction"** — too vague; loses the useful information that these are classifiers.

My recommendation is to keep **"Classifier"**. The reason is that a reader scanning the figure needs the column label to orient them quickly, and "Classifier" does that correctly for all three rows — KNN/SVM/RF are classifiers, and linear/MLP/cosine heads are the classification components of the deep-learning pipelines. The distinction between a full classifier and a lightweight classification head is real but fine-grained enough that it belongs in the text rather than the figure header.

If you want to acknowledge both uses without cluttering the label, **"Classifier head"** is a reasonable compromise — it's standard in the fine-tuning literature, and KNN/SVM/RF can be reasonably called classifier heads on top of the tetramer/UC-CAP features. But it's a small gain over the simpler "Classifier".

## JMD

If the prediction head for HyenaDNA is attached to the backbone, where does the "Pool hidden states" step fit in? The HyenaDNA authors state: "We pool along the sequence dimension to obtain a classification token, and attach a simple linear decoder head." So the decoder head isn't attached to backbone, but to the pooled representation - or am I missing something?

## Claude

You're not missing anything — your reading is precise and correct.

The actual HyenaDNA pipeline has three distinct steps: backbone → pool → head. "Pool hidden states" is an intermediate step between the backbone and the head, not part of either. The HyenaDNA authors are explicit about this: pooling produces a "classification token" and *then* the decoder head is attached to that token.

This means the current figure is slightly misleading for HyenaDNA, because it compresses the pipeline into two cells (Features and Processing) before the Classifier column, but the Processing cell ("Pool hidden states") is really producing an intermediate representation that feeds the head, not a final feature vector in the same sense as run-level tetramer aggregation or UC/CAP.

For SetBERT the two-cell structure works cleanly: the [CLS] token produced by the SABs *is* the run-level representation that goes directly to the head, so "Processing → Classifier" maps neatly.

For HyenaDNA, you could argue the figure would be more accurate if the Processing cell said something like "Pool hidden states → classification token" to make explicit that pooling produces an intermediate representation, not a final feature. But that risks making the label verbose and asymmetric with the SetBERT row.

The more fundamental question is whether this matters for readability. Most readers will likely not scrutinize the boundary between pooling and the head closely enough for it to cause confusion. If you're satisfied that the text in the Models section makes the three-step pipeline clear, the figure can stay as-is. But if you want the figure to stand alone more precisely, renaming the Processing cell for HyenaDNA to "Pool hidden states (→ head)" or splitting it would be the honest fix.

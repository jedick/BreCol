# SetBERT package

This is a SetBERT package adapted from the official implementation at <https://github.com/DLii-Research/setbert> (original author: David W. Ludwig II), with its DNABERT backbone from <https://github.com/DLii-Research/dbtk-dnabert> and shared model plumbing from <https://github.com/DLii-Research/deepbio-toolkit>.

Repackaged by Jeffrey Dick with assistance from Cursor: three upstream `pip install -e` distributions were collapsed into one flat `setbert` package containing only the runtime code used for fine-tuning and inference. Local modifications are summarized in the header comments of `*.py` files.

Pruned vs. upstream:

- `SetBertForPretraining`, `SetBertForSequenceEmbedding`, `SetBertForSampleEmbedding`, `DnaBertForPretraining`, and the Qiita/Greengenes pretraining data modules are removed (not used at fine-tune / inference time).
- `dbtk.nn.layers` is trimmed to the `MultiHeadAttention` → `RelativeMultiHeadAttention` → `MultiHeadAttentionBlock` → `TransformerEncoderBlock` → `TransformerEncoder` chain that `DnaBert` constructs; the flex-attention / induced-set / decoder classes are dropped, removing the `explainable-attention`, `Deprecated`, and PyTorch Lightning dependencies.
- The published `sirdavidludwig/setbert` checkpoint's `config.json` references `dnabert.models.DnaBert(ForEmbedding)`; `SetBertConfig.__init__` rewrites those legacy class paths to `setbert.dnabert.*` on load.

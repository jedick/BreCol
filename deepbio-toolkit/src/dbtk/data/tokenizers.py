from typing import Optional
from .._utils import export

@export
def tokenize_dna(
    sequence: str,
    kmer: int,
    kmer_stride: int = 1
):
    """
    Tokenize a DNA sequence string into k-mers.
    If a vocabulary is provided, map to token IDS.
    """
    return (sequence[i:i+kmer] for i in range(0, len(sequence) - kmer + 1, kmer_stride))


@export
class DnaTokenizer:
    def __init__(self, kmer: int, kmer_stride: int = 1):
        self.kmer = kmer
        self.kmer_stride = kmer_stride

    def __call__(self, sequence: str):
        return tokenize_dna(sequence, self.kmer, self.kmer_stride)

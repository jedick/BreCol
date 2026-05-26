from dnadb.fasta import FastaDb, FastaMappingEntry
from dnadb.taxonomy import TaxonomyDb
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Iterable, Optional, Union

from .interfaces import Fasta

class AmpliconSampleDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[Union[FastaDb, FastaMappingEntry]],
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.samples = list(samples)
        self.transform = transform

    def __getitem__(self, index):
        x = self.samples[index]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.samples)


class SequenceDataset(Dataset):
    def __init__(self, sequences: Union[Fasta, FastaDb, Path, str], transform: Optional[Callable] = None):
        super().__init__()
        if isinstance(sequences, (str, Path)):
            if Path(sequences).is_dir():
                sequences = FastaDb(sequences)
            else:
                sequences = Fasta.open(sequences)
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        entry = self.sequences[index]
        if self.transform is not None:
            entry = self.transform(entry)
        return entry


class SequenceTaxonomyDataset(Dataset):
    def __init__(
        self,
        sequences: Union[FastaDb, Path, str],
        taxonomies: Union[TaxonomyDb, Path, str],
        sequence_transform: Optional[Callable] = None,
        taxonomy_transform: Optional[Callable] = None
    ):
        super().__init__()
        if not isinstance(sequences, FastaDb):
            sequences = FastaDb(sequences)
        if not isinstance(taxonomies, TaxonomyDb):
            taxonomies = TaxonomyDb(taxonomies)
        self.sequences = sequences
        self.taxonomies = taxonomies
        self.sequence_transform = sequence_transform
        self.taxonomy_transform = taxonomy_transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        entry = self.sequences[index]
        label = self.taxonomies[entry.identifier]
        if self.sequence_transform is not None:
            entry = self.sequence_transform(entry)
        if self.taxonomy_transform is not None:
            label = self.taxonomy_transform(label)
        return entry, label

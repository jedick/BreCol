from dataclasses import dataclass
import mmap
from pathlib import Path
import re
from typing import Union

class Fasta:
    """
    An indexable memory-mapped interface for FASTA files.
    """
    @dataclass
    class Entry:
        _fasta_file: "Fasta"
        _id_start: int
        _id_end: int
        _sequence_start: int
        _sequence_end: int

        @property
        def id(self):
            return self._fasta_file.data[self._id_start:self._id_end]

        @property
        def metadata(self):
            return self._fasta_file.data[self._id_end+1:self._sequence_start-1]

        @property
        def sequence(self):
            return self._fasta_file.data[self._sequence_start:self._sequence_end]

        def __len__(self):
            return len(self.sequence)

        def __str__(self):
            return ">" + (self.id.decode() + " " + self.metadata.decode()).strip() \
                + '\n' + self.sequence.decode()

        def __repr__(self):
            return "Entry:\n" + str(self)

    @classmethod
    def open(cls, path: Union[Path, str]):
        with open(path, "r+", encoding="utf8") as f:
            return cls(mmap.mmap(f.fileno(), 0))

    def __init__(self, data):
        self.data = data
        self.entries = []
        self.id_map = {}
        # Lazy reading
        self._length = None
        self._reader = re.finditer(b'>[^>]+', self.data)
        self._eof = False

    def __iter__(self):
        yield from self.entries
        while self._read_next_entry():
            yield self.entries[-1]

    def __getitem__(self, key):
        if not isinstance(key, int):
            while key not in self.id_map and self._read_next_entry():
                continue
            key = self.id_map[key]
        else:
            while len(self.entries) <= key and self._read_next_entry():
                continue
        return self.entries[key]

    def __len__(self):
        if self._length is None:
            self._length = len(re.findall(b'>', self.data))
            if self._length == len(self.entries):
                self._clean_lazy_loading()
        return self._length

    def _read_next_entry(self):
        try:
            match = next(self._reader)
            group = match.group()
            header_end = group.index(b'\n')
            sequence_id_start = match.start() + 1
            sequence_id_end = match.start() + ((group.find(b' ') + 1) or (header_end + 1)) - 1
            sequence_start = match.start() + header_end + 1
            sequence_end = match.end() - 1
            self.entries.append(self.Entry(self, sequence_id_start, sequence_id_end, sequence_start, sequence_end))
            self.id_map[group[1:header_end]] = len(self.id_map)
        except StopIteration:
            self._length = len(self.entries)
        if not self._eof and self._length == len(self.entries):
            self._eof = True
            self._clean_lazy_loading()
        return not self._eof

    def _clean_lazy_loading(self):
        self.__getitem__ = lambda k: self.entries[self.id_map[k] if isinstance(k, str) else k]

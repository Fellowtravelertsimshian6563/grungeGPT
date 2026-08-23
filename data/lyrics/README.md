# Lyrics data

Place plain text lyrics files in this directory. Each `.txt` file is treated as one training document. Files can contain one song or many songs separated by blank lines.

The repository ships with `sample_lyrics.txt`, an original demo corpus, so the CLI works out of the box. It does not bundle copyrighted lyrics.

Suggested file layout:

```text
data/lyrics/
  nirvana.txt
  alice_in_chains.txt
  pearl_jam.txt
  ...
```

The loader reads `.txt` files recursively, normalizes line endings, and keeps newlines as tokens so the model can learn verse structure.

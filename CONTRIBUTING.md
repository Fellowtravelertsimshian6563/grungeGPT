# Contributing

Thanks for contributing to grungeGPT.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

## Pull requests

- Keep changes focused and explain the motivation in the description.
- Add tests for new behavior.
- Run `cargo fmt` and `cargo clippy` before pushing.
- Follow the guidance in `coding_standards.md` for code style.

## Adding lyrics

Do not commit copyrighted lyrics to this repository. The demo corpus in `data/lyrics/sample_lyrics.txt` is original text. Additional `.txt` files under `data/lyrics` are ignored by `.gitignore`. Keep the repository free of copyrighted data.

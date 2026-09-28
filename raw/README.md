# Raw sources

Immutable inputs for the wiki (see `wiki/SCHEMA.md`). Never edit a file here after adding it.

- `papers/`: paper PDFs or markdown clips. Large PDFs may live in `/depot/you139/mfaruqi/on-device-diffusion/papers/` with a pointer file here.
- `meetings/`: meeting notes as written, one file per meeting: `YYYY-MM-DD.md`.

After adding a file, run `/wiki-ingest raw/<path>`.

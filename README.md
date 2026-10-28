# UW-IPL EURS Track2 reproduction

Local reproduction package using the unmodified official [template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template). No public push or upload has been performed.

Read METHOD.md, FACTSHEET.json, DATA_AND_LICENSES.md and STATUS_KO.md. Runtime entrypoints use package-relative paths and local weights. Network access is not needed during inference/export.

```sh
python repro_check.py .
```

All four weight/bank files are committed, including both candidate student states and the contributor restorer. No `weights_link.yaml` is used. The candidate selection remains provisional pending the final best and its actual leaderboard ID.

```sh
docker build -f docker/Dockerfile -t uwipl-track2-repro:final .
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=4g -v "$PWD:/pkg:ro" uwipl-track2-repro:final python repro_check.py . --export-onnx
```

Both candidate exports passed graph/weight comparison in the offline PyTorch 2.4 container. The only cross-version difference was one generated reshape-constant name; the exporter canonicalizes that name without changing values or operators. `reports/export_equivalence.json` also compares ONNX attributes. The training projector is absent from export.

Use `python select_candidate.py T1` or `python select_candidate.py T1v100` to switch the checkpoint and exact original ONNX artifact together. After confirming the chosen candidate's leaderboard ID, provide `--submission-id` and `--contact`. The helper updates `repro.yaml`, `config.json`, `FACTSHEET.json` and the artifact record; switching candidates clears an old candidate's submission ID. Never replace `onnx/submitted_model.onnx` with a re-export.

Run `python repro_check.py . --export-onnx` again and refresh `SHA256SUMS.txt` after final metadata edits. Historical reports record the earlier validation state. The repository remains NOT READY until contact and final submission ID are supplied.

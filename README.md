# UW-IPL EURS Track2 reproduction

Reproduction package for **T1v100, submission #19405 (User #2062)**, using the unmodified official [template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template). Public repository: [heejaeee/UW-IPL-EURS-Track2](https://github.com/heejaeee/UW-IPL-EURS-Track2).

Read METHOD.md, FACTSHEET.json, DATA_AND_LICENSES.md and STATUS_KO.md. Runtime entrypoints use package-relative paths and local weights. Network access is not needed during inference/export.

```sh
python repro_check.py .
```

All four weight/bank files are committed, including both candidate student states and the contributor restorer. No `weights_link.yaml` is used. The owner confirmed T1v100/#19405 as the final model, with official Composite 0.401. v104/#19412 matched its reported official metrics and does not replace it. `config.json` selects T1v100; guided filter radii remain 16 then 4, with epsilon 0.001.

```sh
docker build -f docker/Dockerfile -t uwipl-track2-repro:final .
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=4g -v "$PWD:/pkg:ro" uwipl-track2-repro:final python repro_check.py . --export-onnx
```

Both candidate exports passed graph/weight comparison in the offline PyTorch 2.4 container. The only cross-version difference was one generated reshape-constant name; the exporter canonicalizes that name without changing values or operators. `reports/export_equivalence.json` also compares ONNX attributes. The training projector is absent from export.

The historical T1 checkpoint and `select_candidate.py` are retained for provenance. The final release is fixed to T1v100; do not switch candidates when reproducing #19405. Never replace `onnx/submitted_model.onnx` with a re-export.

Run `python repro_check.py . --export-onnx` to verify the release. Submission ID and contact metadata are complete. Historical reports retain their earlier validation state; current pre-publication results are in `reports/release_export_check.log`.

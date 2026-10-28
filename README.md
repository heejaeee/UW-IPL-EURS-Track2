# UW-IPL EURS Track2 reproduction

Local final-package draft using the unmodified official [template](https://github.com/CeviKle/EURS-MaCVi2027-repro-template). No publicpush or upload has been performed.

Read METHOD.md, FACTSHEET.json and DATA_AND_LICENSES.md. Runtime entrypoints use only relative package paths and local weights; network access is unnecessary.

```sh
python repro_check.py .
```

All four weight/bank files are committed, including the two candidate student states and contributor restorer. No weights_link.yaml is used. The candidate config is provisional pending the actual uploadedID/bestconfirmation.

```sh
docker build -f docker/Dockerfile -t uwipl-track2-repro:final .
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=4g -v "$PWD:/pkg:ro" uwipl-track2-repro:final python repro_check.py . --export-onnx
```

Switch candidates only with select_candidate.py and their exact originalONNX artifact; never rewrite submitted_model.onnx from a re-export. Fill repro.yaml contact and verifiedsubmissionID before finalhandoff. The export graph has no training projector.

Bothcandidateexports passed graph/weightcomparison in theTorch2.4offlinecontainer. The only cross-version difference was a generatedreshapeconstantname; export canonicalizes thatnamewithoutchangingvalues oroperators. reports/export_equivalence.json additionally comparesONNXattributes.

# Experiment record (condensed)

The final lineage is tracked separately from discarded research. Useful empirical improvements included external coral/background supervision, multiscale training over 0.5–1.5, moderate-LR fine-tuning with frozen restoration and BN, global color correction, the contributor v5 restorer, RGB guided filtering, class-balanced Track1 logit KD and SAM3 feature KD. Official deltas across separate versions are not causal ablations.

Discarded or unselected approaches included OS16 and resolution reallocation, class-presence gates/Lovasz, the broader scale range 0.4–2.0, restoration-only segmentation input (v78), learned raw/restored gates (v81), further restoration-loss/grid-head changes, aggressive color augmentation, most model soups, excessive KD-only batch allocation (v101 OTHER recall 21.851%), and larger T1 resolutions.

v103 whole-image fine-tuning completed separately: step 1000 reached local mIoU 52.8116%, F1 66.2825%, Qseg 0.350048 and starfish1 teacher agreement 35.4582%. OTHER recall was 25.3215% and local coverage was 1. Its ONNX passed the recorded compute/numerical gates. It has not been uploaded or substituted into this package; final candidate selection still requires owner confirmation.

Testing-image pseudo-label research was isolated, stopped and retired after organizer clarification. Its products are absent from the final weights. Allowed Dev data is disclosed in Track2 KD. Local validation and teacher-agreement selection were repeatedly reused; neither is an unbiased private-test estimate. See reports/training_protocols.json for recorded schedules and reports for final-lineage audits.

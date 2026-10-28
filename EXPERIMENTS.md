# Experiment record (condensed)

The final lineage is tracked separately from discarded research. Strong empirical improvements included supervised external coral/background data, multiscale0.5–1.5training, moderateLRfinetuning with frozenrestorer/BN, frozen global color restoration, teamv5restoration, RGBguidedfilter, Track1class-balancedlogitKD and SAM3featureKD. Official deltas from separate versions are not treated as causal ablations.

Discarded or unselected: OS16/higherresolutionbudget reallocations, class-presencegate/Lovasz, broadscale0.4–2.0, restored-onlysemanticinput(v78), learnedraw/restoredgate(v81), furtherrestorerloss/gridheadchanges, aggressivecolorrobustness, mostmodel-soups, excessiveKD-onlybatchfraction(v101OTHERrecall21.851%), and largerT1resolutions. v103whole-onlyfinetuning is separate pending verification and is not silently substituted into this package.

Testing-imagepseudo-labelresearch was isolated, stopped and retired after organizer clarification; its products are not in finalweights. Devdata is allowed with disclosure and is used inTrack2KD. Local validation and teacher-agreement selection were repeatedly reused; neither is an unbiasedprivate-testestimate. Detailed checkpoints/protocols are retained in reports/training_protocols.json, without machine-specific paths.

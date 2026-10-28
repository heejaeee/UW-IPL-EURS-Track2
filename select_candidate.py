"""Select an original artifact and matching checkpoint; never uploads."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', choices=['T1', 'T1v100'])
    parser.add_argument('--submission-id', type=int)
    parser.add_argument('--contact')
    args = parser.parse_args()
    if args.submission_id is not None and args.submission_id <= 0:
        parser.error('--submission-id must be positive')

    root = Path(__file__).resolve().parent
    config = json.loads((root / 'config.json').read_text())
    artifacts = json.loads((root / 'reports/submitted_artifacts.json').read_text())
    facts = json.loads((root / 'FACTSHEET.json').read_text())
    metadata = yaml.safe_load((root / 'repro.yaml').read_text())
    source = root / 'onnx/candidates' / f'{args.candidate}.onnx'
    expected = artifacts['candidates'][args.candidate]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Original candidate artifact failed SHA-256 verification')

    # Never carry a different candidate's leaderboard ID across a switch.
    if config['candidate'] != args.candidate:
        metadata['submission_id'] = None
        config['final_best_confirmed'] = False
    if args.submission_id is not None:
        metadata['submission_id'] = args.submission_id
        config['final_best_confirmed'] = True
    if args.contact:
        metadata['contact'] = args.contact
    config['candidate'] = args.candidate
    confirmed = bool(config['final_best_confirmed'] and metadata['submission_id'])
    artifacts['selected'] = args.candidate
    artifacts['exact_uploaded_id_confirmation_pending'] = not confirmed
    facts['submission_id'] = metadata['submission_id']
    facts['selected_candidate'] = args.candidate if confirmed else f'{args.candidate} (provisional)'
    facts['submitted_model_sha256'] = expected

    shutil.copy2(source, root / 'onnx/submitted_model.onnx')
    for relative, value in [('config.json', config), ('FACTSHEET.json', facts),
                            ('reports/submitted_artifacts.json', artifacts)]:
        (root / relative).write_text(json.dumps(value, indent=2) + '\n')
    (root / 'repro.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
    print(args.candidate, expected)
    print('Run repro_check.py again and refresh SHA256SUMS.txt after final metadata edits.')
    print('Historical reports retain the candidate tested at their recorded time.')


if __name__ == '__main__':
    main()

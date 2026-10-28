"""Select a prepared artifact and matching local checkpoint; never uploads."""
from pathlib import Path
import argparse,json,shutil,hashlib
p=argparse.ArgumentParser();p.add_argument('candidate',choices=['T1','T1v100']);p.add_argument('--submission-id',type=int);p.add_argument('--contact');a=p.parse_args()
root=Path(__file__).resolve().parent
cfg=json.loads((root/'config.json').read_text());cfg['candidate']=a.candidate;cfg['final_best_confirmed']=a.submission_id is not None
source=root/'onnx/candidates'/f'{a.candidate}.onnx'
expected=json.loads((root/'reports/submitted_artifacts.json').read_text())['candidates'][a.candidate]
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
shutil.copy2(source,root/'onnx/submitted_model.onnx');(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
if a.submission_id is not None or a.contact:
 import yaml
 y=yaml.safe_load((root/'repro.yaml').read_text())
 if a.submission_id is not None:y['submission_id']=a.submission_id
 if a.contact:y['contact']=a.contact
 (root/'repro.yaml').write_text(yaml.safe_dump(y,sort_keys=False))
print(a.candidate,expected)

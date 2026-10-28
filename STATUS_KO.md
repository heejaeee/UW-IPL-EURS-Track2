# Track2 재현 패키지 상태

- 최종 선택은 사용자 확인을 받은 T1v100(v100-1000), submission #19405(User #2062), 공식 Composite 0.401입니다. config.json은 T1v100으로 고정하며, 과거 T1 가중치는 이력용으로 보존합니다.
- PyTorch 2.4 기본 이미지의 네트워크 차단·읽기 전용 컨테이너에서 두 후보 모두 공식 검사기가 그래프·가중치 일치를 확인했습니다. 연산 속성과 텐서 값도 별도로 대조했습니다.
- Torch 2.4/2.7 차이는 상수 텐서 이름 하나입니다. 숫자나 연산을 바꾸지 않고 이름만 원래 형식으로 통일합니다.
- 11.43190293 GFLOPs, 0.889754 M parameters. 복원 경로는 v76과 동일하며, 학습용 projector는 포함하지 않습니다.
- weights/의 T1.pt, T1v100.pt, best_full.pt, all_train_fits.jsonl을 git으로 추적합니다. weights_link.yaml은 없습니다.
- repro.yaml에 submission_id 19405 및 hkwon4@uw.edu를 기록했습니다. 현재 검사 결과는 reports/release_export_check.log를 참조하세요.
- v104(#19412)는 보고된 공식 지표가 T1v100과 같아 교체하지 않았습니다. guided filter는 r16→r4, eps 0.001이며 ONNX 원본 SHA-256을 유지합니다.
- 공개 저장소: https://github.com/heejaeee/UW-IPL-EURS-Track2. 이전 reports/ 로그는 당시 상태를 보존합니다.

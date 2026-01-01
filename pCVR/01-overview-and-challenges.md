# [TIL] pCVR 모델링 학습 시 주의사항 및 중복 전환 문제

> **날짜:** 2024-05-22
> **키워드:** #pCVR #Sparsity #Deduplication #AdTech

## 1. pCVR 모델링의 주요 챌린지
- **SSB (Sample Selection Bias):** 클릭된 데이터로만 학습하여 전체 노출 영역에 대한 예측 성능이 저하되는 문제. ESMM 구조 검토 필요.
- **Data Sparsity:** 전환 데이터가 매우 적음. Multi-task Learning으로 해결 시도.
- **Delayed Feedback:** 전환 발생까지의 시차 존재. Attribution Window 설정이 핵심.

## 2. 중복 전환 (Duplicated Conversion) 이슈
실무에서 데이터 신뢰도를 떨어뜨리는 주요 원인임.
- **원인:** 페이지 새로고침, 뒤로가기, 멀티 채널 유입 등.
- **해결책:** - `Transaction ID` 기반 제거 (가장 확실함)
  - `Click ID` 매핑을 통한 1:1 매칭

## 3. 인사이트 (Today's Takeaway)
- pCVR은 모델 구조보다 **'정답지(Label)를 얼마나 깨끗하게 만드느냐'**가 성능의 절반 이상을 결정함.
- 중복 제거(Deduplication) 로직은 데이터 파이프라인 상단에서 미리 처리하는 것이 효율적임.

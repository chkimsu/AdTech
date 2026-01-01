# [Deep Dive Report] Scalable Bid Landscape Forecasting in Real-time Bidding (MCNet)

## Part 1. AdTech 비즈니스 배경과 문제의식

이 섹션에서는 모델링에 앞서, 우리가 논의했던 **광고 생태계의 기울어진 운동장**과 **왜 이 모델이 필요한지**에 대한 비즈니스적/구조적 배경을 다룹니다.

### 1. DSP 경쟁 구조와 '기울어진 운동장'

#### 1.1 내부 DSP vs 외부 DSP (Home Ground Advantage)

카카오, 구글, 네이버 등 거대 플랫폼(Walled Garden) 내부의 경쟁 구조는 공정하지 않습니다.

* **데이터(Feature) 격차:**
  * **내부 DSP:** 로그인 기반의 Deterministic Data(쇼핑 이력, 관심사, 선물하기 등)를 100% 활용.
  * **외부 DSP:** 암호화된 ID와 제한적인 Bid Request 정보만 수신.
* **비용(Tech Tax) 격차:**
  * **내부 DSP:** 수수료를 내부적으로 상쇄하여 입찰(Bidding)에 더 많은 예산 투입 가능 (실제 입찰력 95% 이상).
  * **외부 DSP:** SSP, Exchange 수수료를 떼고 남은 금액으로 입찰 (실제 입찰력 70~80%).
* **속도(Latency):** 내부 네트워크 통신으로 Time-out 리스크가 거의 없음.

#### 1.2 실제 사례: 구글의 'Project Bernanke'

* **내용:** 구글이 자사 거래소(AdX)의 과거 입찰 데이터를 자사 DSP(Google Ads)에만 몰래 제공하여 낙찰 확률을 조작하거나 비용을 절감한 프로젝트.
* **시사점:** "플랫폼이 심판과 선수를 겸할 때 발생하는 구조적 불공정(Self-Preferencing)"이 실존함.

### 2. 왜 낙찰가 분포(Landscape)를 예측해야 하는가?

#### 2.1 Walled Garden vs Open Web

* **Walled Garden (네이버 검색광고 등):** 외부 경쟁자가 없으므로 내부 랭킹(pCTR)만 잘하면 됨. (상대평가)
* **Open Web (RTB 환경):** 내가 내부에서 1등을 해도, 외부 DSP(크리테오 등)가 더 비싼 돈(Retargeting 등)을 들고 오면 짐.
  * $\rightarrow$ **"상대방(외부 DSP)이 얼마를 낼지"**를 예측하는 것이 승패의 핵심.

#### 2.2 1가 경매(First-Price Auction)와 Bid Shading

* **시장 변화:** 2가 경매(SPA)에서 1가 경매(FPA)로 트렌드 이동.
* **문제점:** 1가 경매에서는 내가 쓴 돈을 다 내야 하므로, 무턱대고 높게 쓰면 손해(Winner's Curse).
* **해결책 (Bid Shading):**
  * 승률을 유지하면서 입찰가를 깎아야 함.
  * 이를 위해선 **"2등이 얼마를 낼지(Winning Price)"**를 정확히 알아야 그보다 `1원`만 더 비싸게 낼 수 있음.
  * **결론:** 이 논문의 모델(MCNet)이 바로 이 **'2등의 가격 분포'를 예측하는 엔진** 역할을 수행함.

### 3. 기존 예측 모델의 기술적 한계

#### 3.1 점 추정(Point Estimation) vs 분포 추정(Distribution Estimation)

* **pCTR 모델:** 정답이 0/1이므로 확률값 하나(Point)만 추정하면 됨.
* **낙찰가 모델:** 정답이 연속적인 금액(Price)이며 변동성이 큼. 평균값 하나만 믿고 입찰했다간 낭패를 봄. $\rightarrow$ **평균($\mu$)과 분산($\sigma$)을 모두 예측해야 함.**

#### 3.2 중도절단(Censored) 데이터의 딜레마

* **Winning Log:** 낙찰가는 정확히 알 수 있음 ($w = 150\text{원}$).
* **Losing Log:** 낙찰가는 모르고, 내 입찰가보다 비싸다는 것만 앎 ($w > 200\text{원}$).
* **기존 한계:** 기존 모델들은 이 '패찰 로그'를 제대로 활용하지 못하거나, 데이터 분포를 단순 정규분포(Gaussian)로 퉁쳐버림.
  * *현실은 데이터가 여러 군집으로 나뉜 **다봉형(Multi-modal)** 구조임 (논문 Figure 1 참조).*

---

## Part 2. 기술 방법론: Censored Regression과 MCNet

이 섹션에서는 "정답을 모르는 데이터(패찰 로그)"를 학습시키기 위한 수학적 기법과, 이를 딥러닝으로 구현한 **MCNet**의 내부 로직을 해부합니다.

### 1. 핵심 문제 해결: 중도절단 회귀 (Censored Regression)

일반적인 회귀 분석(MSE Loss)을 쓸 수 없는 이유는 **패찰(Lose) 데이터의 정답($y$)이 '특정 값'이 아니라 '범위(Range)'로 존재**하기 때문입니다. 이를 해결하기 위해 **통계적 우도(Likelihood)** 개념을 도입합니다.

#### 1.1 데이터의 두 가지 상태 (State)

모델은 들어오는 데이터의 상태에 따라 계산 방식을 이원화합니다.

1.  **Non-censored Data (낙찰, Win):**
    * 정확한 관측값 $w_i$ 존재.
    * **목표:** 모델이 예측한 확률 분포에서 $w_i$ 지점의 **높이(Density, PDF)**를 최대화.
2.  **Right-censored Data (패찰, Lose):**
    * 정확한 값은 모르나, 입찰가 $b_i$보다는 크다는 하한선(Lower Bound) 존재 ($W_i \ge b_i$).
    * **목표:** 모델이 예측한 확률 분포에서 $b_i$보다 큰 구간의 **면적(Probability, CDF)**을 최대화.

#### 1.2 수학적 직관 (Intuition)

> "이긴 데이터는 **과녁의 정중앙**을 맞추도록 영점을 조절하고, 진 데이터는 최소한 **담장 너머**로 화살을 보내도록 힘을 조절한다."

### 2. 모델의 진화 과정 (Evolution of Models)

논문은 기존 방식의 한계를 단계적으로 극복하며 최종 모델(MCNet)로 나아갑니다.

#### 2.1 Baseline: CR (Standard Censored Regression)

* **가정:** $W \sim \mathcal{N}(\beta^T x, \sigma^2)$
* **구조:** 입력($x$) $\rightarrow$ 평균($\mu$) 예측. **분산($\sigma$)은 상수(Constant)로 고정.**
* **한계:** 시장 상황(시간, 지면 등)에 따라 가격 변동성이 달라지는데(이분산성), 이를 무시하고 "모든 상황의 불확실성은 똑같다"고 가정함.

#### 2.2 Upgrade: P-CR (Fully Parametric CR)

* **가정:** $W \sim \mathcal{N}(\beta^T x, \exp(\alpha^T x))$
* **구조:** 입력($x$) $\rightarrow$ 평균($\mu$)과 **분산($\sigma$)을 동시에 예측.**
* **발전:** "새벽 시간대(안정적)는 분산을 작게, 뉴스 지면(불안정)은 분산을 크게" 예측 가능해짐.
* **한계:** 여전히 **단일 정규분포(Unimodal)**만 사용. 낙찰가가 두 가격대에 뭉쳐 있는(Multi-modal) 현실 데이터(Figure 1)를 설명 불가.

#### 2.3 Final: MCNet (Mixture Density Censored Network)

* **핵심 아이디어:** **MDN (Mixture Density Network)** 도입.
* **가정:** $P(w|x) = \sum_{k=1}^{K} \pi_k(x) \mathcal{N}(w | \mu_k(x), \sigma_k(x))$
* **의미:** $K$개의 정규분포를 가중치($\pi$)를 두어 섞음으로써, 어떤 복잡한 형태의 분포도 근사(Approximation) 가능.

### 3. MCNet 아키텍처 상세 (Architecture Detail)

pCTR 모델러의 관점에서 Input/Output 레이어의 변화를 분석합니다.

#### 3.1 Input & Hidden Layer

* **Input ($x$):** 고차원 희소 벡터 (User ID, AD Slot ID, Time, Device 등).
* **Hidden:** Fully Connected Layer + ReLU Activation. (Feature Representation 학습)

#### 3.2 Output Layer (핵심 차별점)

일반적인 pCTR 모델은 1개의 노드(Sigmoid)를 갖지만, MCNet은 혼합 성분 개수($K$)에 따라 **$3 \times K$개**의 노드를 출력합니다.

| 출력 변수            | 개수 | 활성화 함수 | 역할 및 의미                      |
| :------------------- | :--- | :---------- | :-------------------------------- |
| **$\pi$ (Priors)**   | $K$  | `Softmax`   | 각 분포가 선택될 확률. (총합 = 1) |
| **$\sigma$ (Scale)** | $K$  | `Exp`       | 각 분포의 표준편차. (항상 양수)   |
| **$\mu$ (Location)** | $K$  | `Linear`    | 각 분포의 중심(평균) 가격.        |

> **예시 ($K=2$):**
>
> * **분포 1 (짠돌이 그룹):** 평균 100원, 분산 10, 비중 70%
> * **분포 2 (큰손 그룹):** 평균 500원, 분산 50, 비중 30%
> * $\rightarrow$ 결과적으로 100원과 500원에 봉우리가 두 개 있는 쌍봉 낙찰가 분포가 완성됨.

### 4. 손실 함수: Censored NLL (Equation 6)

이 모델의 학습 엔진인 **Negative Log-Likelihood (NLL)** 수식의 구조입니다. pCTR의 `LogLoss`와 구조적으로 동일합니다.

$$
\mathcal{L} = - \sum_{i} \log \left( \text{Probability}(Data_i) \right)
$$

이 $\text{Probability}$ 부분이 데이터 타입에 따라 갈라집니다.

#### 4.1 이긴 경우 (Win Term)

$$
\sum_{k=1}^{K} \pi_k \cdot \frac{1}{\sigma_k} \phi \left( \frac{w_i - \mu_k}{\sigma_k} \right)
$$

* **해석:** $K$개의 정규분포들이 합심하여 실제 낙찰가($w_i$) 위치에서의 **확률 밀도(Density)**를 계산.
* **학습 방향:** 실제 가격($w_i$) 근처에 분포의 중심($\mu$)을 맞추고, 폭($\sigma$)을 좁혀서 높이를 키우도록 유도.

#### 4.2 진 경우 (Lose Term)

$$
\sum_{k=1}^{K} \pi_k \cdot \left( 1 - \Phi \left( \frac{b_i - \mu_k}{\sigma_k} \right) \right)
$$

* **해석:** $K$개의 정규분포들이 합심하여 입찰가($b_i$)보다 **오른쪽(더 비싼 영역)에 있는 면적(Survival Probability)**을 계산.
* **학습 방향:** "정답은 모르지만 적어도 200원보단 비싸다"는 신호를 받아, 분포를 200원보다 오른쪽으로 밀어내거나(Shift), 꼬리를 길게 늘리도록(Variance Increase) 유도.

---

## Part 3. 실험 결과 및 결론

마지막 파트에서는 MCNet이 실제 데이터에서 기존 방법론들을 어떻게 압도했는지 정량적으로 분석하고, 현업 적용 시 고려해야 할 한계점과 미래 연구 방향을 제시합니다.

### 1. 실험 설정 (Experimental Setup)

#### 1.1 데이터셋 (Datasets)

* **iPinYou (Public):** 학계 표준 벤치마크 데이터. (Season 2, 3 활용)
* **Adobe AdCloud (Proprietary):** 실제 상용 DSP의 대규모 로그 데이터 (약 3,100만 건).
  * *의의:* 실제 리얼월드 트래픽에서의 유효성을 검증함.

#### 1.2 평가 지표 (Metric): ANLP

* **ANLP (Average Negative Log Probability):**
  * 점 추정 오차(MSE)는 분포 예측 모델을 평가할 수 없으므로, **"모델이 예측한 분포가 실제 데이터를 얼마나 잘 설명하는가(Likelihood)"**를 측정.
  * 값이 **낮을수록(0에 가까울수록)** 성능이 우수함.

### 2. 성능 분석 (Performance Analysis)

#### 2.1 전체 모델 비교 (Leaderboard)

실험 결과(Table 2)에 따르면, 모델 간 성능 서열은 다음과 같이 명확하게 정리됩니다.

1.  **MCNet (Proposed):** 압도적 1위.
    * 기존 Baseline(CR) 대비 **30% 이상** 성능 향상.
    * 강력한 경쟁자(Survival Tree) 대비 **10% 이상** 성능 향상.
2.  **P-CR (Parametric-CR):** 2위권.
    * 분산($\sigma$)을 예측하는 것만으로도 고정 분산 모델(CR)보다 5~10% 우수함.
3.  **ST (Survival Tree):** 2~3위권.
    * 비모수적 방법 중 가장 강력하지만, 데이터 희소성 문제에 취약함.
4.  **CR (Standard CR):** 하위권.
    * 단일 가우시안 가정의 한계 노출.
5.  **KM (Kaplan-Meier):** 최하위권.
    * 개인화(Feature 활용)가 전혀 안 되므로 예측력이 가장 낮음.

#### 2.2 확장성 대결 (Scalability): MCNet의 승리 요인

이 논문의 가장 중요한 발견점 중 하나는 **"Feature 소화 능력"**의 차이입니다.

* **핸디캡 매치 (Feature Trimming 적용):**
  * ST(트리 모델)는 연산량 한계로 빈도수 $10^7$ 이하의 Feature를 모두 버려야 했음.
  * 이 조건($*$)에서 MCNet과 ST의 성능은 비슷함. (둘 다 유연하므로)
* **무제한 매치 (Full Features):**
  * **MCNet**은 딥러닝 기반이므로 수만 개의 Feature를 전부 학습에 활용 가능.
  * **결과:** 모든 정보를 활용한 MCNet이 제한된 정보만 쓴 ST를 압도함.
  * **결론:** **"Big Data & High-dimensional Sparse Vector 환경에서는 딥러닝 기반의 MCNet이 유일한 대안이다."**

### 3. 한계점 및 제언 (Limitations & Discussion)

#### 3.1 식별 불가능성 (Non-identifiability)

* **문제:** 학습 데이터에 존재하는 **'최대 입찰가(Max Bid)' 이상의 영역**에 대해서는 모델이 학습할 기회가 없음.
* **현상:** MCNet은 유연하기 때문에, 이 미지의 영역에 대해 평평하게 그리든 꼬리를 길게 그리든 수학적으로는 모두 정답 처리가 됨. (실제 분포와 다를 수 있음)
* **해결책 (Hybrid Approach):**
  * 데이터가 있는 구간 $\rightarrow$ **MCNet** (Data-driven)
  * 데이터가 없는 꼬리 구간 $\rightarrow$ **통계적 가정** (Rule-based, e.g., Exponential Decay)을 결합하는 것이 안전함.

#### 3.2 하이퍼파라미터 민감도

* **Mixture 개수 ($K$):** $K$가 너무 작으면 복잡한 분포를 표현 못 하고, 너무 크면 과적합(Overfitting) 발생. 실험적으로 $K=4 \sim 8$ 정도가 적절함.

### 4. 최종 결론 (Conclusion)

본 연구는 RTB 환경의 고질적인 문제인 **중도절단(Censored) 데이터**와 **다봉형(Multi-modal) 가격 분포**를 해결하기 위해 **MCNet**을 제안했습니다.

1.  **정확성:** 딥러닝과 혼합 모델(Mixture Model)을 결합하여 업계 최고 수준의 예측 정확도 달성.
2.  **실용성:** pCTR 모델과 유사한 아키텍처 및 Loss 구조를 가져, 기존 파이프라인에 통합하기 용이하며 대용량 데이터 처리가 가능함.
3.  **활용:** 이 모델은 1가 경매(First-Price Auction) 하에서의 **Bid Shading** 전략 수립 및 예산 최적화(Budget Pacing)의 핵심 엔진으로 즉시 활용 가능함.

---

## [Appendix] 광고 시스템 생태계 및 기술 방법론 Q&A

본 문서는 논문 *"Scalable Bid Landscape Forecasting in Real-time Bidding"*의 이해를 돕기 위해, 광고 기술(AdTech)의 비즈니스 구조와 핵심 모델링 개념을 질의응답(Q&A) 형태로 정리한 기술 부록입니다.

### A. 광고 생태계와 '기울어진 운동장' (Business Context)

#### 1. Walled Garden(폐쇄형 플랫폼)과 DSP 경쟁 구도

카카오, 네이버, 구글과 같은 거대 플랫폼은 매체(Publisher)이자 거래소(Exchange), 그리고 DSP 역할을 동시에 수행합니다. 이로 인해 내부 DSP와 외부 DSP 간에는 구조적인 불공정함이 존재합니다.

| 비교 항목          | **내부 DSP (예: Kakao Moment)**                              | **외부 DSP (예: Criteo, Google DV360)**                      |
| :----------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **역할 비유**      | **홈 팀 (Home Team)**                                        | **원정 팀 (Away Team)**                                      |
| **데이터 (Data)**  | **1st Party Data (Raw Level)**<br>로그인 정보, 쇼핑 이력, 관심사 등 유저를 100% 식별 가능. | **3rd Party / Bid Request (Limited)**<br>암호화된 ID(ADID/IDFA)와 제한적 정보만 수신. |
| **비용 (Cost)**    | **No Tech Tax**<br>수수료 상쇄 가능. <br>$\rightarrow$ **같은 100원 예산으로 95원 입찰 가능.** | **High Tech Tax**<br>DSP/Exchange 수수료 차감.<br>$\rightarrow$ **같은 100원 예산으로 70~80원 입찰 가능.** |
| **속도 (Latency)** | **Zero Latency**<br>타임아웃(Time-out) 리스크 없음.          | **Network Latency**<br>100ms 내 응답 못 하면 기회 박탈.      |
| **경쟁 방식**      | **Selection (선발전)**<br>가장 적합한 내부 광고를 골라내는 랭킹 시스템. | **Valuation (본선 경쟁)**<br>외부의 다른 DSP들과 '돈(Price)'으로 싸우는 경매. |

#### 2. Project Bernanke (프로젝트 버냉키)

* **정의:** 구글이 자사 광고 거래소(AdX)의 독점적 지위를 이용해, 자사 DSP(Google Ads)에게 경쟁사들의 과거 입찰 데이터를 몰래 제공한 프로젝트.
* **시사점:** 내부 DSP는 '정답지(낙찰 예상가)'를 미리 알고 입찰하는 불공정한 이점(Self-Preferencing)을 누릴 수 있음이 밝혀짐.

### B. 경매 방식과 MCNet의 필요성 (Auction Mechanism)

왜 낙찰가 예측 모델(MCNet)이 필요한지는 **경매 방식의 변화**와 밀접한 관련이 있습니다.

#### 1. 2가 경매 (Second-Price Auction, SPA)

* **규칙:** 1등이 낙찰받되, 가격은 **2등이 부른 가격(+1원)**만 지불.
* **전략:** 내 가치(Valuation)를 그대로 입찰하는 것이 최적 전략. (MCNet 필요성 낮음)

#### 2. 1가 경매 (First-Price Auction, FPA) - *Current Trend*

* **규칙:** 1등이 낙찰받고, **자기가 부른 가격 그대로** 지불.
* **문제점:** 2등이 500원인데 내가 1000원을 내면 500원 손해 (**Winner's Curse**).
* **전략 (Bid Shading):** 승률을 유지하는 선에서 입찰가를 깎아야 함.
  * *Shaded Bid = Valuation $\times$ (1 - Shading Factor)*
* **MCNet의 역할:** 입찰가를 얼마나 깎을지 결정하려면 **"2등이 얼마를 낼지(Winning Price)"**를 알아야 함. MCNet은 바로 이 **Winning Price의 분포(Landscape)**를 예측해 주는 핵심 엔진임.

### C. 기술 방법론 심층 Q&A (Technical Deep Dive)

모델링 과정에서 pCTR 모델러가 헷갈리기 쉬운 개념들을 정리했습니다.

#### Q1. pCTR 모델은 점 추정인가요? (Point vs. Distribution)

* **pCTR 모델 (Point Estimation):**
  * 정답($y$)이 0(노출) 아니면 1(클릭)인 이진 분류 문제입니다.
  * 베르누이 분포 특성상 확률값 $p$ 하나만 알면 되므로, $p$라는 **값 하나(Point)**만 추정합니다.
* **낙찰가 모델 (Distribution Estimation):**
  * 정답($y$)이 10원, 100원, 1000원 등 연속적이며 변동 폭이 큽니다.
  * 평균($\mu$) 하나만으로는 "얼마나 위험한지(Risk)"를 알 수 없습니다. 따라서 **평균($\mu$)과 분산($\sigma$), 더 나아가 분포의 모양($\pi$)** 전체를 추정해야 합니다.

#### Q2. 왜 Loss Function을 Win/Lose로 나누나요? (Censored Data)

데이터의 **관측 상태(Observability)**가 다르기 때문입니다.

1.  **Win Data (이김):**
    * **상황:** 정확한 낙찰가($w$)를 압니다.
    * **목표:** "내 예측 분포에서 **$w$ 지점의 높이(PDF)**를 최대화해라." (영점 사격)
2.  **Lose Data (짐):**
    * **상황:** 내 입찰가($b$)보다 비싸다($w > b$)는 것만 압니다.
    * **목표:** "내 예측 분포에서 **$b$보다 오른쪽 구간의 면적(Survival Probability, CDF)**을 최대화해라." (장외 홈런)

#### Q3. Parametric vs. Non-parametric vs. MCNet

논문에서 비교하는 세 가지 모델군의 차이점입니다.

| 구분     | **Parametric (예: CR)**       | **Non-parametric (예: Survival Tree)** | **MCNet (제안 모델)**                |
| :------- | :---------------------------- | :------------------------------------- | :----------------------------------- |
| **정의** | 파라미터 개수가 고정됨.       | 데이터에 따라 모델 크기가 변함.        | **Parametric** (구조적 분류)         |
| **가정** | "분포는 정규분포 하나다."     | "가정 없음. 데이터 모양 그대로 그림."  | "정규분포 여러 개를 섞으면 뭐든 됨." |
| **장점** | 빠르고 가벼움. (Scalable)     | 정확하고 유연함. (Flexible)            | **빠르고 유연함.** (둘 다 잡음)      |
| **단점** | 복잡한 현실 데이터를 못 맞춤. | 데이터/Feature가 많으면 느려서 못 씀.  | 하이퍼파라미터($K$) 튜닝 필요.       |

### D. 평가지표: ANLP (Average Negative Log Probability)

RMSE나 AUC 같은 일반적인 지표를 쓰지 못하는 이유와 ANLP의 의미입니다.

#### 1. 왜 RMSE를 못 쓰는가?

RMSE는 예측값($\hat{y}$)과 정답($y$)의 차이를 계산합니다. 하지만 패찰(Lose) 데이터는 **정답($y$)이 얼만지 모르기 때문에** 오차를 계산할 수 없습니다.

#### 2. ANLP의 원리

**"모델이 실제 데이터를 보고 얼마나 덜 놀랐는가(Surprise)?"**를 측정합니다.

$$
ANLP = - \frac{1}{N} \sum \log(P(\text{Data}|\text{Model}))
$$

* **Win 데이터:** 모델이 "$150$원일 확률이 높아요!"라고 했는데 진짜 $150$원이면 $\rightarrow$ 확률($P$) 높음 $\rightarrow$ Loss 작음 $\rightarrow$ **Good.**
* **Lose 데이터:** 모델이 "$200$원보단 비쌀 확률이 높아요!"라고 했는데 진짜 졌으면(비쌌으면) $\rightarrow$ 확률($P$) 높음 $\rightarrow$ Loss 작음 $\rightarrow$ **Good.**

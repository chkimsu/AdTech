import numpy as np

class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0):
        """
        n_arms: 후보 광고의 개수
        n_features: Context Vector의 차원 수 (위 예시에서는 7)
        alpha: 탐색(Exploration) 가중치. 클수록 모험을 즐김.
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # 각 광고(Arm)별로 학습해야 할 행렬 초기화
        # A: 공분산 행렬 (데이터의 정밀도) -> Identity로 초기화
        # b: 보상 벡터 (클릭 여부 누적) -> 0으로 초기화
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def select_arm(self, context_vectors):
        """
        현재 Context에서 가장 점수(UCB Score)가 높은 광고 선택
        context_vectors: 각 광고별 Feature Vector 리스트
        """
        best_arm = -1
        max_ucb = -float('inf')
        
        scores = []
        
        for arm_idx in range(self.n_arms):
            x = context_vectors[arm_idx].reshape(-1, 1) # (d, 1) 형태
            A_inv = np.linalg.inv(self.A[arm_idx])     # 역행렬 계산
            
            # 1. Exploitation: 예측 CTR (Theta * x)
            theta = np.dot(A_inv, self.b[arm_idx])
            prediction = np.dot(theta.T, x).item()
            
            # 2. Exploration: 불확실성 (Confidence Interval)
            # 데이터가 적은 방향(x)일수록 이 값이 큼
            uncertainty = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x).item())
            
            # 최종 점수 = 예측값 + 불확실성 보너스
            ucb_score = prediction + uncertainty
            scores.append(ucb_score)
            
            if ucb_score > max_ucb:
                max_ucb = ucb_score
                best_arm = arm_idx
                
        return best_arm, scores

    def update(self, arm_idx, context_vector, reward):
        """
        유저의 반응(클릭/노클릭)을 보고 모델 업데이트
        reward: 1 (클릭), 0 (노출되었으나 클릭 안함)
        """
        x = context_vector.reshape(-1, 1)
        
        # A 행렬 업데이트 (Outer Product 더하기)
        self.A[arm_idx] += np.dot(x, x.T)
        
        # b 벡터 업데이트 (Reward 가중치 더하기)
        self.b[arm_idx] += reward * x

# --- [시뮬레이션] ---

# 설정: 광고 2개, Feature 3개 (단순화: [매칭타입점수, 유저성별, 시즌성])
model = LinUCB(n_arms=2, n_features=3, alpha=0.5)

# 상황: 유저가 "남자 청바지" 검색
# 광고 0 (Exact Match): 매칭점수 높음(1), 남성타겟(1), 시즌(1) -> [1, 1, 1]
# 광고 1 (Broad Match): 매칭점수 낮음(0.2), 남성타겟(1), 시즌(1) -> [0.2, 1, 1]
context_ad_0 = np.array([1.0, 1.0, 1.0])
context_ad_1 = np.array([0.2, 1.0, 1.0])
contexts = [context_ad_0, context_ad_1]

print("--- 1. 초기 상태 (학습 전) ---")
chosen_arm, scores = model.select_arm(contexts)
print(f"선택된 광고: {chosen_arm} (점수: A={scores[0]:.2f}, B={scores[1]:.2f})")
# 초기에는 불확실성이 커서 랜덤하거나 Context가 뚜렷한 쪽이 선택됨

print("\n--- 2. 학습 진행 (광고 0번 클릭 발생 가정) ---")
# 광고 0번이 노출되었고, 클릭됨(Reward=1)
model.update(arm_idx=0, context_vector=context_ad_0, reward=1)

chosen_arm, scores = model.select_arm(contexts)
print(f"업데이트 후 선택: {chosen_arm} (점수: A={scores[0]:.2f}, B={scores[1]:.2f})")
# 광고 0의 예측값(Exploitation)이 상승하여 점수가 높아짐

print("\n--- 3. 상황 변화 (다른 Context) ---")
# 광고 1만 유리한 상황 (예: 여성 유저가 검색해서 Feature가 바뀜)
# 광고 0: [1, 0, 1], 광고 1: [0.2, 0, 1] -> 하지만 광고 1이 안 가본 길이라면?
context_new_0 = np.array([1.0, 0.0, 1.0])
context_new_1 = np.array([0.2, 0.0, 1.0]) 
# 실제로는 광고 1에 대해 탐색 보너스가 여전히 높게 작용할 수 있음
chosen_arm, scores = model.select_arm([context_new_0, context_new_1])
print(f"새로운 상황 선택: {chosen_arm}")
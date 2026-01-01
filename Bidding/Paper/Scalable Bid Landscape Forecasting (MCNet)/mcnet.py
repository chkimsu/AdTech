import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class MCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_mixtures=4):
        """
        Mixture Density Censored Network (MCNet) 구현
        논문: Scalable Bid Landscape Forecasting in Real-time Bidding (KDD 2019)
        
        Args:
            input_dim (int): 입력 Feature의 차원 수
            hidden_dim (int): 은닉층 노드 수 (논문에서는 64 사용)
            num_mixtures (int): 섞을 가우시안 분포의 개수 K (논문에서는 2~8 사용)
        """
        super(MCNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        
        # 1. Feature Representation Layer (Shared)
        # 논문에서는 fully connected hidden layer 사용
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        # 2. Mixture Density Head
        # 출력은 K개의 분포 각각에 대한 (mu, sigma, pi)를 내뱉어야 하므로 3 * K개 노드 필요
        self.output_layer = nn.Linear(hidden_dim, 3 * num_mixtures)

    def forward(self, x):
        """
        Forward Pass
        
        Returns:
            pi (Tensor): 각 혼합 성분의 가중치 (Batch, K), Sum=1 (Softmax)
            sigma (Tensor): 각 혼합 성분의 표준편차 (Batch, K), Positive (Exp)
            mu (Tensor): 각 혼합 성분의 평균 (Batch, K)
        """
        x = self.activation(self.layer1(x))
        output = self.output_layer(x)
        
        # 출력값을 mu, sigma, pi로 분리 (Split)
        # output shape: (Batch, 3 * K) -> mu, sigma, pi 각각 (Batch, K)
        mu, log_sigma, log_pi = torch.chunk(output, 3, dim=1)
        
        # 제약 조건 적용
        # 1. Sigma는 항상 양수여야 함 -> exp 적용
        sigma = torch.exp(log_sigma)
        
        # 2. Pi는 확률이므로 합이 1이어야 함 -> softmax 적용
        pi = F.softmax(log_pi, dim=1)
        
        # Mu는 제약 없음 (실수 전체)
        
        return pi, sigma, mu

def censored_nll_loss(pi, sigma, mu, target, is_win):
    """
    MCNet의 핵심인 Censored Negative Log-Likelihood Loss 함수 (식 6번)
    
    Args:
        pi, sigma, mu: 모델의 예측값 (Batch, K)
        target (Tensor): 정답 값 (Batch, 1). 
                         - Win인 경우: 실제 낙찰가 (w_i)
                         - Lose인 경우: 내 입찰가 (b_i) -> Lower Bound
        is_win (Tensor): 승패 여부 (Batch, 1). True(1)면 Win, False(0)면 Lose
    
    Returns:
        loss (Scalar): 배치의 평균 NLL
    """
    
    # 편의를 위해 target 차원 맞춤
    target = target.view(-1, 1) # (Batch, 1)
    
    # 정규분포 객체 생성 (배치 처리 지원)
    m = Normal(mu, sigma)
    
    # --- Part 1: Winning Auctions (이긴 경우) ---
    # 목표: Log Sum ( pi * PDF(w) )
    # PDF 계산: exp(log_prob) -> 확률 밀도 값
    pdf_values = torch.exp(m.log_prob(target)) # (Batch, K)
    
    # Mixture 확률 합산: Sum (pi_k * PDF_k)
    weighted_pdf = torch.sum(pi * pdf_values, dim=1, keepdim=True)
    
    # Log Likelihood for Wins (수치적 안정을 위해 1e-9 추가)
    log_likelihood_win = torch.log(weighted_pdf + 1e-9)

    # --- Part 2: Losing Auctions (진 경우) ---
    # 목표: Log Sum ( pi * P(W > b) ) = Log Sum ( pi * (1 - CDF(b)) )
    # Normal.cdf(x)는 P(X <= x)이므로, P(X > b)는 1 - CDF(b)임.
    
    survival_prob = 1.0 - m.cdf(target) # P(W > b)
    
    # Mixture 확률 합산
    weighted_survival = torch.sum(pi * survival_prob, dim=1, keepdim=True)
    
    # Log Probability for Loss (Lower Bound)
    log_likelihood_lose = torch.log(weighted_survival + 1e-9)
    
    # --- Combine ---
    # 각 데이터가 Win인지 Lose인지에 따라 Loss를 선택 (Conditional Loss)
    # is_win이 1이면 win term 사용, 0이면 lose term 사용
    
    final_log_likelihood = torch.where(is_win.bool(), log_likelihood_win, log_likelihood_lose)
    
    # Negative Log Likelihood (Minimize해야 하므로 - 붙임)
    loss = -torch.mean(final_log_likelihood)
    
    return loss

# ==========================================
# 실행 예시 (Example Usage)
# ==========================================
if __name__ == "__main__":
    # 1. 하이퍼파라미터 설정
    BATCH_SIZE = 1024
    INPUT_DIM = 20  # 예시 Feature 차원
    HIDDEN_DIM = 64
    K = 4           # Mixture 개수
    LR = 1e-3
    
    # 2. 모델 및 옵티마이저 초기화
    model = MCNet(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_mixtures=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6) # L2 reg
    
    print("Model Structure:")
    print(model)
    
    # 3. 더미 데이터 생성
    # x: Feature 벡터 (Batch, Input_dim)
    x_batch = torch.randn(BATCH_SIZE, INPUT_DIM)
    
    # target: Win인 경우 실제 가격 w, Lose인 경우 입찰가 b
    # (실제로는 로그 스케일링 등을 해서 넣는 것이 일반적입니다)
    target_batch = torch.abs(torch.randn(BATCH_SIZE, 1) * 100 + 200) 
    
    # is_win: 1이면 낙찰(Win), 0이면 패찰(Lose) (랜덤 생성)
    is_win_batch = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    
    # 4. Training Step
    model.train()
    
    # Forward Pass
    pi, sigma, mu = model(x_batch)
    
    # Loss Calculation
    loss = censored_nll_loss(pi, sigma, mu, target_batch, is_win_batch)
    
    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\nTraining Step 완료!")
    print(f"Loss 값: {loss.item():.4f}")
    
    # 5. 예측값 확인 (Inference)
    # 첫 번째 샘플의 예측 분포 파라미터 확인
    print(f"\n[Sample 0 Prediction]")
    print(f"Weights (pi): {pi[0].detach().numpy()}")
    print(f"Means (mu):   {mu[0].detach().numpy()}")
    print(f"StdDevs (sigma): {sigma[0].detach().numpy()}")
    print(f"Target: {target_batch[0].item():.2f}, Win?: {bool(is_win_batch[0].item())}")

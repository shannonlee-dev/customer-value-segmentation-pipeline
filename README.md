# H&M 고객가치 세분화 파이프라인

## 목표와 미션 대응

이 프로젝트는 H&M 구매 이력, 고객 속성, 상품 텍스트·카테고리, 상품 이미지를 결합해 고객가치 세그먼트를 설명하는 재현 가능한 분석이다. 결정적 표본화, 결측치 처리, 이미지·텍스트 특징, IQR 이상치 점검, 고유 구매일 기준 RFM, 실행 완료 노트북, 개인정보·재배포 경계를 하나의 작업 흐름으로 남긴다. `price`는 데이터셋의 상대값이며 특정 통화 금액으로 해석하지 않는다.

## 데이터를 찾은 방법

Kaggle, AI Hub, UCI Online Retail, Olist, 이미지 전용 자료를 먼저 비교했다. 거래 이력, 고객과 날짜, 상품 텍스트·카테고리, 실제 이미지를 **동시에** 제공하고 자연스럽게 결합 가능한지를 기준으로 걸렀다. AI Hub와 이미지 전용 자료는 구매·고객 시계열이 부족했고, UCI Online Retail과 Olist는 이 프로젝트에 필요한 상품 이미지 결합 조건을 충족하지 못했다.

## 선택한 이유

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)는 거래·상품·고객·이미지를 `article_id` 하나로 자연스럽게 연결한다. 따라서 구매일과 상대 가격으로 RFM을 만들고, 상품명 길이·카테고리·이미지 통계를 같은 분석 표본에서 다룰 수 있다. 여러 출처를 억지로 결합하거나 행동 값을 추정하지 않아도 된다는 점이 선택의 핵심이다.

## 라이선스 및 사용 조건

데이터를 받기 전에 Kaggle 로그인 상태에서 공식 [대회 페이지](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)와 [competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules)를 검토하고 대회 조건을 수락해야 한다. 사용 범위는 대회 규칙의 비상업적·학술적 경계 안에 둔다. 원본 파일, 처리된 행 단위 파일, 이미지, 자격 증명은 내려받은 환경에만 두며 원본/처리 데이터를 재배포하지 않는다. 이 데이터에 CC 또는 MIT 라이선스를 주장하지 않는다.

## 출처 매핑과 결정적 코호트

| 분석 열 | 공식 원천 열/규칙 |
| --- | --- |
| 구매일, 상대 가격, 판매 채널 | `transactions_train.csv`의 `t_dat`, `price`, `sales_channel_id` |
| 고객 속성 | `customers.csv`의 `age`, `club_member_status`, `fashion_news_frequency` |
| 상품 텍스트와 카테고리 | `articles.csv`의 `prod_name`, `product_group_name` |
| 이미지 경로 | `article_id`에서 계산한 로컬 이미지 경로 |

활성 고객 전체에서 `SHA-256(seed + ":" + customer_id)`의 오름차순 앞 **500**명을 고른다(기본 seed 42). 구매액·최근성·빈도를 보고 뽑지 않으므로 재실행해도 같은 고객 집합이 나오며 선택 편향을 가치 기준으로 추가하지 않는다. 상품·고객 메타데이터의 누락 또는 키 중복은 **fail-fast 검증 오류**로 처리한다. 메타데이터 결함을 조용히 제외하지 않으며, 선택된 거래 중 로컬 이미지 파일이 없거나 원본 shape가 `(1750, 1166, 3)`이 아닌 행은 제외하고 각각의 수와 비율을 로컬 요약에 기록한다. 실제 식별자, 상품명, 원본 행은 저장소와 문서에 노출하지 않는다.

## 재현 워크플로

대회 조건을 수락한 뒤 로컬에만 내려받는다.

```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw/h-and-m
unzip data/raw/h-and-m/h-and-m-personalized-fashion-recommendations.zip -d data/raw/h-and-m
```

```text
data/
├── raw/h-and-m/{articles.csv,customers.csv,transactions_train.csv,images/}
└── processed/{hm_customer_cohort.csv,hm_customer_cohort.summary.json}
```

코호트를 만들고 노트북을 재생성·검증한 다음 테스트한다. `data/raw/`, `data/processed/`, 이미지와 `kaggle.json`은 추적하지 않는다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/prepare_hm_data.py \
  --raw-dir data/raw/h-and-m --output data/processed/hm_customer_cohort.csv
HM_RAW_DATA_DIR=data/raw/h-and-m .venv/bin/python scripts/build_notebook.py
MPLCONFIGDIR=/tmp/hm-matplotlib-cache HM_RAW_DATA_DIR=data/raw/h-and-m \
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/analysis_report.ipynb --ExecutePreprocessor.timeout=600
.venv/bin/python scripts/verify_notebook.py notebooks/analysis_report.ipynb
.venv/bin/python -m unittest discover -s tests -v
```

## 분석 방법과 범위

전체 9,351개 거래에는 Pandas 문자열 연산으로 상품명 길이를 계산한다. 나이는 회원 상태 **그룹별** 중앙값으로 먼저 채우고, 해당 그룹에 값이 없을 때만 전체 중앙값으로 보완한다. 가격은 IQR 1.5배 울타리 밖을 표시하되 원본을 삭제하지 않고 비교용 복사본만 필터링한다. RFM의 Frequency는 행 수가 아니라 **고유 구매일** 수라서 주문 식별자가 없는 원천에서 같은 날의 여러 행을 과대계수하지 않는다.

전처리에서 원본 shape가 `(1750, 1166, 3)`인 거래만 남긴 뒤, 이미지는 결정적으로 선택한 64개 상품에만 Matplotlib로 읽고 NumPy 슬라이싱(`::35`) 뒤 `np.stack`으로 묶어 Mean/Std를 축 방향으로 계산한다. 이 상품별 특징을 해당 상품이 등장한 **81개 거래 행에 다시 결합**하므로 이미지 요약과 상관계수는 64개 상품을 동일 가중한 값이 아니라 **거래 가중(transaction-weighted)** 통계다. 원본 shape 검증으로 축소 후 텐서는 모두 `(50, 34, 3)`이며 서로 다른 축소 형상은 계속 엄격히 거부한다. 표본은 이미지 I/O와 대형 텐서의 메모리 비용을 제한하기 위한 것이며, 전체 거래·텍스트·IQR·RFM은 shape 필터 후 전체 범위를 사용한다. 슬라이싱 뷰가 큰 디코딩 배열을 붙잡지 않도록 compact-copy로 **연속 메모리**를 만든 뒤 적층한다. JPEG를 하나씩 읽는 I/O와 텐서 적층은 여전히 주요 **병목**이다.

## 집계 근거

아래 값은 원본 shape 필터 후 전체 9,351개 거래·6,538개 상품을 대상으로 한 표/텍스트/IQR/RFM과, 무작위 상태 42로 결정적으로 고른 64개 상품 이미지 표본을 분리해 계산했다. 이미지 특징은 81개 거래 행에 다시 결합되어 이미지 요약과 상관계수가 거래 가중된다. 전체 분석 프레임의 형상은 **(9,351, 12)**이며, 기간은 2018-09-20부터 2020-09-22까지다. 나이 결측은 대치 전 72건이고 IQR 밖은 478건이며 경계는 -0.010440677966101701와 0.060474576271186305이다.

| RFM 세그먼트 | 고객 수 | 평균 최근성 | 평균 고유 구매일 수 | 평균 상대 Monetary | Monetary 비중 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Churned | 180 | 453.8556 | 1.2833 | 0.1083 | 0.0748 |
| Loyal | 88 | 281.2727 | 4.9318 | 0.3924 | 0.1325 |
| New | 20 | 30.8500 | 1.3500 | 0.1240 | 0.0095 |
| Potential | 49 | 118.5714 | 1.3878 | 0.1250 | 0.0235 |
| VIP | 161 | 56.5217 | 12.8075 | 1.2294 | 0.7596 |

다음 상관행렬은 결정적으로 선택한 **64개 상품 이미지 표본**의 특징을 **81개 거래 행**에 다시 결합해 계산했다. 따라서 이미지 관련 계수는 전체 코호트 이미지 추정치가 아니라 표본 범위의 거래 가중 기술값이다.

|  | 상대 가격 | 나이 | 이미지 평균 | 이미지 표준편차 | 상품명 길이 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 상대 가격 | 1.0000 | -0.1129 | -0.3082 | 0.2479 | 0.1280 |
| 나이 | -0.1129 | 1.0000 | 0.1136 | -0.1878 | 0.0753 |
| 이미지 평균 | -0.3082 | 0.1136 | 1.0000 | -0.9410 | -0.1484 |
| 이미지 표준편차 | 0.2479 | -0.1878 | -0.9410 | 1.0000 | 0.1636 |
| 상품명 길이 | 0.1280 | 0.0753 | -0.1484 | 0.1636 | 1.0000 |

이미지 평균과 상품명 길이는 -0.1484로 약한 음의 선형 신호이고, 상대 가격과 이미지 평균은 -0.3082로 중간 정도의 음의 선형 신호다. 두 값 모두 관찰적·결정적 64개 상품 표본 범위의 해석일 뿐 인과관계를 뜻하지 않는다.

## 비즈니스 인사이트

1. VIP는 161명(32.3%)이지만 Monetary의 0.7596을 차지하고 평균 고유 구매일 수는 12.8075다. 대상은 VIP이며, 최근 구매 문맥을 이용한 재입고·연관 제안을 적용해 재방문 방향의 변화를 본다. 무작위 A/B 홀드아웃에서 재방문률과 상대 Monetary가 대조군보다 오르지 않으면 이 제안의 가설을 **반증**한다.
2. Churned는 180명(36.1%)으로 가장 크고 평균 최근성은 453.8556일, 평균 고유 구매일 수는 1.2833이다. 대상은 Churned이며, 일괄 할인 대신 동의 기반 재접촉의 타이밍·메시지를 실험해 복귀 방향을 측정한다. A/B 홀드아웃에서 90일 복귀율이 개선되지 않거나 접촉 피로 지표가 악화되면 중단한다.
3. Loyal은 88명으로 Monetary 비중 0.1325, 평균 고유 구매일 수 4.9318을 보인다. 대상은 Loyal이며, VIP 전환용 혜택을 소규모로 시험해 다음 구매일과 상대 Monetary의 상승 방향을 본다. 추가로 필요한 데이터는 노출·쿠폰·재고·반품 정보이며, 이를 보정한 대조 분석 또는 A/B에서 차이가 사라지면 전환 가설을 반증한다.

## 한계와 다음 단계

IQR은 비대칭 가격 분포에서 유효한 정상값도 이상치로 표시할 수 있고, 그룹별 대치는 나이의 분산을 축소한다. 500명 결정 표본은 모집단을 대표한다고 보장하지 않으며, 원본 shape 필터가 비표준 이미지 상품을 체계적으로 제외한다. 이미지 통계는 64개 상품을 81개 거래 행에 다시 결합한 거래 가중 표본이라 자주 구매된 상품의 영향이 더 크고 색상·형태의 의미도 모두 담지 않는다. 가격은 정규화된 상대값이고 주문 ID가 없어 같은 날 주문을 완전히 구분할 수 없다. 인과 효과는 관찰 분석만으로 결론낼 수 없다.

다음 단계는 90일 **이탈**을 목표 변수로 정의하고, 최근성·고유 구매일 빈도·상대 Monetary·회원 상태·채널·카테고리·이미지/텍스트 특징과 함께 노출·재고·반품·동의 기반 마케팅 이력을 추가로 필요한 데이터로 받아 시간 분할 검증하는 것이다.

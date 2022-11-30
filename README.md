# Recommend-Model
k-NN과 그래프 신경망을 이용한 하이브리드 시퀀셜 추천시스템
https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11035715

## 개요
2021년 한국소프트웨어종합학술대회에 발표된 "k-NN과 그래프 신경망을 이용한 하이브리드 시퀀셜 추천시스템"논문의 실험 코드입니다.  
  
사용자의 기록 session 데이터에 기반하여 SR-GNN모델의 결과값과 IKNN의 결과값을 더하여 아이템을 추천합니다.  
  
Session-Based Recommendation with Graph Neural Networks(SR-GNN)  
https://github.com/CRIPAC-DIG/SR-GNN  
데이터에서 아이템 수가 많은 경우 전처리과정과 모델 연산 내부에서 memory낭비가 일어나기 때문에 dict 형태로 저장하여 사용합니다.  
  
Item KNN  
https://github.com/leonvking0/Recommendation_Algos  
코사인 유사도 계산하는 부분이 잘못되어 있어 수정되었습니다.  

## 사용방법
SR-GNN+SKNN.ipynd 파일로 실행 할 수 있습니다.  
논문에서 제시한 추가 방법이 각각 Method 1,2,3으로 나누어져 있습니다.

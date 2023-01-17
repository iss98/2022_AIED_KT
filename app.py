import streamlit as st
import torch
import pandas as pd
import numpy as np

from model import *

st.set_page_config(
  page_title = "2022_학생연구",
  layout = "wide",
)

st.title("T-RKT를 이용한 학습진단 예시")
st.write("**팀원** : 수학교육과 김래영, 송무호, 신인섭")

model = RKT(400)
model.load_state_dict(torch.load("best_model.pt", map_location = "cpu"))
model.eval()
df = pd.read_csv("sample_text.csv", index_col = 0)

item1 = torch.tensor(df.iloc[0,:].astype('float32').values).unsqueeze(dim = 0)
item2 = torch.tensor(df.iloc[1,:].astype('float32').values).unsqueeze(dim = 0)
item3 = torch.tensor(df.iloc[2,:].astype('float32').values).unsqueeze(dim = 0)
item4 = torch.tensor(df.iloc[3,:].astype('float32').values).unsqueeze(dim = 0)
item5 = torch.tensor(df.iloc[4,:].astype('float32').values).unsqueeze(dim = 0)

cr = torch.ones(200).unsqueeze(dim = 0)
incr = torch.zeros(200).unsqueeze(dim = 0)

st.subheader("문항1")
st.markdown("어떤 고등학교 3학년 남학생 수는 여학생 수의 1.5배이다. 대학수학능력시험 모의고사 통계에 따르면 남학생의 평균 점수는 400점 만점에 225점이고 여학생의 평균 점수는 235점이다. 3학년 전체 학생의 평균 점수는 몇 점인가?")
response1 = None
response1 = st.radio("정답을 골라주세요", ("229", "230", "231", "232", "233"))

st.subheader("문항2")
st.markdown("세 자료")
st.markdown("* $A$ : 1부터 50까지의 자연수")
st.markdown("* $B$ : 51부터 100까지의 자연수")
st.markdown("* $C$ : 1부터 100까지의 짝수  ")
st.markdown("의 표준편차를 순서대로 $a, ~b, ~c$라 할 때, $a, ~b, ~c$의 대소관계를 바르게 나타낸 것은?")
response2 = None
response2 = st.radio("정답을 골라주세요", ("a=b=c", "a=b<c", "a<b=c", "a<b<c", "a<c<b"))

st.subheader("문항3")
st.markdown("다음은 첫째 항이 $a-15d$, 공차가 $d$, 항의 개수가 31인 등차수열이다.")
st.markdown("$a-15d, ~\cdots,~a-d,~a,~a+d,~\cdots,~a+15d$")
st.markdown("위 항들의 값이 표준편차를 $\sigma$라고 할 때, $\frac{\sigma}{d}$의 값을 소수점 아래 둘째 자리까지 구하시오.")
st.markdown("(단, $d > 0$이고 $\sqrt{5} = 2.25$로 계산한다.) ")
response3 = st.number_input("정답을 적어주세요", min_value = 7.00, max_value = 10.00, step = 0.01)

st.subheader("문항4")
st.markdown("주사위를 한 번 던져 나오는 눈의 수를 4로 나눈 나머지를 확률변수 $X$라 하자. $X$의 평균은?")
st.markdown("(단, 주사위의 각 눈이 나올 확률은 모두 같다.)")
response4 = None
response4 = st.radio("정답을 골라주세요", ("2", "5/3", "3/2", "4/3", "1"))

if st.button("학습진단 시작"):
    if response1 == "229":
        inter1 = torch.cat((item1, cr), dim = -1)
    else : 
        inter1 = torch.cat((item1, incr), dim = -1)

    if response2 == "a=b<c":
        inter3 = torch.cat((item3, cr), dim = -1)
    else : 
        inter3 = torch.cat((item3, incr), dim = -1)

    if response3 == 8.96:
        inter4 = torch.cat((item4, cr), dim = -1)
    else :
        inter4 = torch.cat((item4, incr), dim = -1)

    if response4 == "3/2":
        inter2 = torch.cat((item2, cr), dim = -1)
    else :
        inter2 = torch.cat((item2, incr), dim = -1)

    input_inter = torch.cat([inter1,inter3, inter4, inter2], dim = 0).unsqueeze(dim = 0)
    input_text = torch.cat([item1, item3, item4, item2], dim = 0).unsqueeze(dim = 0)
    target_text1 = torch.cat([item3, item4, item2, item5], dim = 0).unsqueeze(dim = 0)
    out1 = model(target_text1, input_inter, input_text)[:,-1,:].squeeze().detach().cpu().numpy()  
    predict = round(out1[622],2)

    st.subheader("문항5")
    st.markdown("어떤 학생이 오랜만에 방문하는 인터넷 사이트에 접속하기 위하여 비밀번호 여섯 자리를 입력하려고 한다. 이 학생은 비밀번호를 지정할 때, 앞의 네 자리는 항상 자신의 생일 숫자인을 사용하고 뒤의 두 자리는  중에서 서로 다른 두 숫자를 택하여 사용하는데, 뒤의 두 자리 수가 전혀 기억나지 않는다. 비밀번호 입력을 시작하여 맞는지 확인하는 데 걸리는 시간은 초이고, 접속에 실패한 비밀번호는 다시 입력하지 않는다. 처음 입력할 때부터 접속될 때까지 소요되는 시간의 기댓값은?")

    if st.button("이 문제를 맞출 확률은?"):
        if predict > 0.5:
            st.success(f"문제를 맞출 확률 :  {predict}")
        else : 
            st.error(f"문제를 맞출 확률 :  {predict}")
else : 
    st.write("문항을 모두 풀고 학습 진단을 시작해주세요")

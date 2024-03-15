import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint
from langchain.llms import OpenAI
from config import *
import os

st.title('📧 Testimony Writer Assistant App')

openai_api_key = st.sidebar.text_input('OpenAI API Key')
huggingface_api_key = st.sidebar.text_input('HuggingFaceHub API Key')
# huggingface_repo_id = st.sidebar.text_input('HuggingFace Repo ID', "mistralai/Mistral-8x7B-Instruct-v0.1")

option = st.selectbox(
    'Which model that you want to test?',
    ('Gemini', 'GPT4'))

st.write('You selected:', option)

def generate_response(input_text):

  # os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
  # os.environ["LINE_CHATBOT_API_KEY"] = LINE_CHATBOT_API_KEY

  # 1. setup prompt
  prompt_1 = ChatPromptTemplate.from_template(
      """
      คุณคือตำรวจที่มีหน้าที่ในการแปลสำนวนบันทึกประจำวันเหล่านี้ให้เป็นคำให้การในชั้นศาล สังกัดสถานีตำรวจนครบาลบาลดอนเมือง 

      รูปแบบที่คุณจะสรุปในรายงานมีดังนี้

      ข้าพเจ้า ร้อยตำรวจเอก สมชาย ใจดี สังกัดสถานีตำรวจนครบาลดอนเมือง  ขอรายงานการสอบสวน  (หัวข้อคดีความ)

      **ผู้แจ้ง** : (ข้อมูลผู้แจ้ง)

      **วันเวลาที่แจ้ง** : (วันเวลาที่แจ้ง)

      **สถานที่แจ้ง**: (สถานที่แจ้ง)

      **เหตุการณ์**: (รายละเอียดของเหตุการณ์ที่เกิดขึ้น เขียนโดยสรุปเป็นลำดับขั้นตอนตามเวลา)

      **พยาน**: (ข้อมูลผู้พบเห็นเหตุการณ์)

      สำนวนบันทึกประจำวัน: {topic}
      """
  )

  # 2. chain invocation
  output_parser = StrOutputParser()
  chain_1 = prompt_1 | llm | output_parser

  reply_message = chain_1.invoke({"topic": input_text})

  st.info(reply_message)

with st.form('my_form'):
  # Call Large Langauge Model (LLM) Function Depending on User
  if option == 'GPT4': 
    llm = OpenAI(model_name="gpt-4", api_key=openai_api_key)
  elif option == 'Gemini': 
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision")
  # elif option == 'HuggingFace': 
  #   llm = HuggingFaceEndpoint(repo_id=huggingface_repo_id, temperature=0.5, token=huggingface_api_key)

  # prompt
  text = st.text_area('How do you want your testimony to be written?', 'นาย ก. ซึ่งอาศัยอยู่เขตจตุจักร กรุงเทพมหานคร ทำบัตรประชาชนหาย เมื่อวันที่ 15 มีนาคม พ.ศ. 2567 เวลาประมาณ 15:00 สถานที่ที่คาดว่าหายคือ สนามบินดอนเมือง มีพยานที่อยู่ด้วยกันคือนาง ข. ซึ่งเป็นคุณแม่ของนาย ก. และอาศัยอยู่ด้วยกันกับนาย ก. และมีการเข้าแจ้งความในวันถัดไป')
  
  submitted = st.form_submit_button('Submit')

  # Generate Reponse
  ## Gemini
  if option == 'Gemini' and submitted:
    generate_response(text)
  
  ## GPT 4
  if option == 'GPT4' and not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if option == 'GPT4' and submitted and openai_api_key.startswith('sk-'):
    generate_response(text)

  # ## HuggingFaceHub
  # if option == 'HuggingFace' and not huggingface_api_key.startswith('hf_'):
  #   st.warning('Please enter your HuggingFaceHub API key!', icon='⚠')
  # if option == 'HuggingFace' and submitted and huggingface_api_key.startswith('hf_'):
  #   generate_response(text)
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from openai import OpenAI
from config import *
import os

st.title('📧 Testimony Writer Assistant App')

def generate_response(input_text):

  os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
  os.environ["LINE_CHATBOT_API_KEY"] = LINE_CHATBOT_API_KEY

  llm = ChatGoogleGenerativeAI(model="gemini-pro")
  llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision")

    # 1. setup prompt
  prompt_1 = ChatPromptTemplate.from_template(
      """
      คุณคือตำรวจที่มีหน้าที่ในการแปลสำนวนบันทึกประจำวันเหล่านี้ให้เป็นคำให้การในชั้นศาล

      สำนวนบันทึกประจำวัน: {topic}
      """
  )

  # 2. chain invocation
  output_parser = StrOutputParser()
  chain_1 = prompt_1 | llm | output_parser

  reply_message = chain_1.invoke({"topic": input_text})

  st.info(reply_message)

with st.form('my_form'):
  # prompt
  text = st.text_area('How do you want your testimony to be written?', 'นาย ก. ทำบัตรประชาชนหาย เมื่อวันที่ 15 มีนาคม พ.ศ. 2567 เวลาประมาณ 15:00 สถานที่ที่คาดว่าหายคือ สนามบินดอนเมือง มีพยานที่อยู่ด้วยกันคือนาง ข. ซึ่งเป็นคุณแม่ของนาย ก.')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text)
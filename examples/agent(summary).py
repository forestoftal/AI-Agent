import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
from twilio.rest import Client

# Load environment variables
load_dotenv("openaikey.env")

# Get keys from .env
openai_api_key = os.getenv("OPENAI_API_KEY")
twilio_sid = os.getenv("TWILIO_SID")
twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
to_whatsapp = os.getenv("TO_WHATSAPP")
from_whatsapp = "whatsapp:+14155238886"  # Twilio Sandbox number

# Initialize LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0.3)

# === Tool 1: Summarizer ===
def summarize_text(text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="Summarize the following meeting transcript clearly in 3-5 bullet points:\n\n{transcript}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"transcript": text})

# === Tool 2: WhatsApp Sender ===
def send_whatsapp(message: str) -> str:
    try:
        client = Client(twilio_sid, twilio_token)
        sent_msg = client.messages.create(
            body=message,
            from_=from_whatsapp,
            to=to_whatsapp
        )
        return f"✅ WhatsApp message sent! SID: {sent_msg.sid}"
    except Exception as e:
        return f"❌ Failed to send WhatsApp message: {str(e)}"

# Wrap tools
tools = [
    Tool(name="Summarizer", func=summarize_text, description="Summarizes meeting transcripts into bullet points."),
    Tool(name="WhatsAppSender", func=send_whatsapp, description="Sends a message to WhatsApp."),
]

# Initialize LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

                                                                     
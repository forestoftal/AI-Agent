import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from studyflow_agent_langgraph import app as langgraph_app

# Load environment variables
load_dotenv()

st.set_page_config(page_title="📚 StudyFlow Agent", layout="centered")
st.title("📖 StudyFlow LangGraph Agent")

st.markdown("Upload a lecture PDF and select your review start date. The AI agent will create summaries, flashcards, and a review schedule.")

# File upload
pdf_file = st.file_uploader("Upload your lecture PDF", type=["pdf"])
start_date = st.date_input("Select review start date", value=datetime.today())

# Run button
if st.button("Run StudyFlow Agent 🚀"):
    if not pdf_file:
        st.warning("Please upload a PDF.")
    else:
        with st.spinner("Processing... please wait"):
            # Save file to disk
            save_path = "molecularbio.pdf"
            with open(save_path, "wb") as f:
                f.write(pdf_file.read())

            # Run the LangGraph agent
            result = langgraph_app.invoke({
                "pdf_path": save_path,
                "start_date": start_date.strftime("%Y-%m-%d")
            })

        st.success("✅ Done! Your outputs are ready.")

        # Display outputs
        st.subheader("📄 Summary Preview")
        with open("output_langgraph/summaries_langgraph.txt", "r", encoding="utf-8") as f:
            st.text(f.read()[:3000] + "\n...")

        st.subheader("🧠 Flashcards")
        import pandas as pd
        flashcards_df = pd.read_csv("output_langgraph/flashcards_langgraph.csv")
        st.dataframe(flashcards_df)

        st.subheader("📅 Download Schedule (.ics)")
        with open("output_langgraph/review_schedule_langgraph.ics", "rb") as f:
            st.download_button("Download .ics file", f, file_name="review_schedule.ics", mime="text/calendar")

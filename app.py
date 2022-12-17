import torch
import streamlit as st

from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(
        "stanford-crfm/pubmed_gpt_tokenizer")
    model = GPT2LMHeadModel.from_pretrained(
        "stanford-crfm/pubmedgpt").to(device)
    return tokenizer, model


tokenizer, model = load_model()

st.title("PubMedGPT Demo")


# define 2 tabs for the app
tab = st.sidebar.radio("Choose a tab", ["Text Generation", "MedQA"])
st.sidebar.write(
    "Blog [Stanford](https://crfm.stanford.edu/2022/12/15/pubmedgpt.html)")

if tab == "Text Generation":
    prompt = st.text_input("Prompt", "Photosynthesis is")

    max_length = st.slider("Max length", 10, 100, 50)

    num_return_sequences = st.slider("Num return sequences", 1, 5, 1)

    generated = []

    if st.button("Generate"):
        try:
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

            for output in outputs:
                generated.append(tokenizer.decode(
                    output, skip_special_tokens=True))
        except Exception as e:
            st.write(e)
    for text in generated:
        # write text with animation typing effect
        st.write(text, use_column_width=True, key=text)

elif tab == "MedQA":
    # load model pubmedgpt for MedQA
    tokenizer, model = load_model()

    question = st.text_input("Question", "What is the treatment for COVID-19?")
    context = st.text_input(
        "Context", "COVID-19 is a disease caused by SARS-CoV-2. The most common symptoms are fever, dry cough, and tiredness. The most common treatment is hydroxychloroquine.")

    if st.button("Answer"):
        try:
            input_ids = tokenizer.encode(
                question, context, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids,
                max_length=100,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated = tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            st.write(generated)
        except Exception as e:
            st.write(e)

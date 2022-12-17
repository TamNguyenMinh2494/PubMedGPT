import torch
import streamlit as st

from transformers import GPT2Tokenizer, GPT2LMHeadModel


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model


tokenizer, model = load_model()

st.title("GPT-2 Demo")

prompt = st.text_input("Prompt", "Photosynthesis is")

max_length = st.slider("Max length", 10, 100, 50)

num_return_sequences = st.slider("Num return sequences", 1, 5, 1)

generated = []
if st.button("Generate"):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
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
st.write("Generated text:")
for text in generated:
    # show generated text with number of tokens
    st.write(f"{text}")


st.write(
    "Blog [Stanford](https://crfm.stanford.edu/2022/12/15/pubmedgpt.html)")

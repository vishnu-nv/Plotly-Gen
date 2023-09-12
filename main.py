import gradio as gr
from text_generation import Client

API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-13b-hf"
API_TOKEN = "YOUR_TOKEN"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

#Prompt
def generate_prompt(visualization_type, description, metadata):
    prompt = f"Create a {visualization_type} Visualization for {description}."
    prompt += f" Metadata: {metadata}"
    generated_text = generate_text(prompt)
    return generated_text

# #Prompt
# def generate_prompt(visualization_type, description, metadata):
#     prompt = f"Create a {visualization_type} Visualization for {description}."
#     prompt += f" Metadata: {metadata}"
#     generated_text = generate_text(prompt)
#     return generated_text



def generate_text(prompt):
    client = Client(API_URL)
    text = []
    for response in client.generate_stream(prompt, max_new_tokens=900):
        if not response.token.special:
            text.append(response.token.text)
    final_text = "".join(text)
    return final_text

#Input
visualization_type = gr.inputs.Dropdown(['Bar Chart', 'Pie Chart'], label="Visualization Type")
description = gr.inputs.Textbox(label="Visualization Description")
metadata = gr.inputs.Textbox(label="Metadata Information")

#Gradio interface
interface = gr.Interface(
    fn=generate_prompt,
    inputs=[visualization_type, description, metadata],
    outputs="text",
    title="Plotly-Gen",
    description="Enter the input parameters"
)

if __name__ == "__main__":
    interface.launch()

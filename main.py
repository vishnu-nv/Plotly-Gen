import gradio as gr
from text_generation import Client

API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-13b-hf"
API_TOKEN = "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"

FIM = "<FILL_ME>"
EOS = "</s>"
EOT = "<EOT>"

theme = gr.themes.Monochrome(
  primary_hue="indigo",
  secondary_hue="blue",
  neutral_hue="slate",
  radius_size=gr.themes.sizes.radius_sm,
  font=[
    gr.themes.GoogleFont("Open Sans"),
    "ui-sans-serif",
    "system-ui",
    "sans-serif",
  ],
)

client = Client(
  API_URL,
  headers={"Authorization": f"Bearer {API_TOKEN}"},
)

def generate(prompt, temp=0.9, max_tokens=1000, top_p=0.95, rep_penalty=1.0):
  temp = max(float(temp), 1e-2)
  top_p = float(top_p)
  fim = FIM in prompt

  gen_kwargs = dict(
    temperature=temp,
    max_new_tokens=max_tokens,
    top_p=top_p,
    repetition_penalty=rep_penalty,
    do_sample=True,
    seed=42,
  )

  if fim:
    try:
      prefix, suffix = prompt.split(FIM)
      prompt = f"<PRE> {prefix} <SUF>{suffix} <MID>"
    except ValueError:
      raise ValueError(f"Only one {FIM} allowed in prompt!")

  stream = client.generate_stream(prompt, **gen_kwargs)
  output = prefix if fim else prompt

  for response in stream:
    if any([end_token in response.token.text for end_token in [EOS, EOT]]):
      if fim:
        output += suffix
        yield output
      else:
        return output
    else:
      output += response.token.text
    yield output

description = """
<div style="text-align: center;">
  <h1> Plotly- Gen </h1>
</div>
"""

with gr.Blocks(theme=theme, analytics_enabled=False) as demo:
  with gr.Column():
    gr.Markdown(description)
    with gr.Row():
      with gr.Column():
        instruction = gr.Textbox(
          placeholder="Enter your code here",
          lines=5,
          label="Input",
          elem_id="q-input",
        )
        submit = gr.Button("Generate", variant="primary")
        output = gr.Code(elem_id="q-output", lines=30, label="Output")
  
  submit.click(
    generate,
    inputs=[instruction],
    outputs=[output],
  )
demo.queue(concurrency_count=16).launch(debug=True)

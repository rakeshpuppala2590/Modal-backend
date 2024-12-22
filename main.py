import modal
import torch
import io
from fastapi import Query, Response, HTTPException, Request
import requests
import os
from datetime import datetime, timezone

app = modal.App("example-text-to-image")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "fastapi[standard]",
)

with image.imports():
    from diffusers import AutoPipelineForText2Image
    import torch
    from fastapi import Response

@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("custom-secret")],
)

class StableDiffusion:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo"
                                                              ,
                                                              torch_dtype=torch.float16,variant="fp16")
        self.pipe.to("cuda")
        self.CLIENT_XT_BAR_1 = os.environ["CLIENT_XT_BAR_1"]


    @modal.web_endpoint()
    def generate_endpoint(self, request: Request, prompt: str= Query(..., title="Prompt", description="The text prompt to generate an image from")):
        api_key = request.headers.get("X-API-KEY")
        if api_key != self.CLIENT_XT_BAR_1:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        image = self.pipe(prompt,inference_steps=1, guidance_scale=0.0).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    @modal.web_endpoint()
    def health_endpoint(self):
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("custom-secret")],
)

def keep_warm():

    health_url = "https://rakeshpuppala2591--example-text-to-image-stablediffusion-bf7236.modal.run/"
    generate_url = "https://rakeshpuppala2591--example-text-to-image-stablediffusion-da8a9a.modal.run"
    health_response = requests.get(health_url)

    print(health_response.json()['timestamp'])
    headers = {
        "X-API-KEY": os.environ["CLIENT_XT_BAR_1"]
    }
    generate_response = requests.get(generate_url, headers==headers)
    print(datetime.now(timezone.utc).isoformat())
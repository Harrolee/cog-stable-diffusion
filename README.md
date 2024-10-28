# Stable Diffusion v2 Cog model

I deployed Sayak Paul's cartoonizer to Replicate. Read his [blog post](https://huggingface.co/blog/instruction-tuning-sd)

# Use Locally

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"

## Apple Silion?

replace 'cuda' with 'mps'

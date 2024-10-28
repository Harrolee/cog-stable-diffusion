from typing import List
from PIL import Image, ImageOps
import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionInstructPix2PixPipeline


MODEL_ID = "instruction-tuning-sd/cartoonizer"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="The image you want to cartoonize",
        ),
        prompt: str = Input(
            description="This model was finetuned to use the default prompt",
            default="Cartoonize the following image",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        image = Image.open(str(image))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        output = self.pipe(
            prompt=prompt,
            image=image,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths

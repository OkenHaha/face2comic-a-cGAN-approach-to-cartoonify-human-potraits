# face2comic — cGAN Approach to Cartoonify Human Portraits

Final semester project for Master of Computer Applications (MCA), submitted 2024.

A conditional Generative Adversarial Network (cGAN) trained on the Pix2Pix framework to perform unpaired image-to-image translation — converting real human portrait photographs into comic/cartoon-style renderings.

**Trained model on HuggingFace:** [okenk/face2comic](https://huggingface.co/okenk/face2comic) *(if applicable)*

---

## What it does

Given a real portrait image as input, the model generates a stylized comic-book version of the face — preserving facial structure, expression, and identity while applying a learned artistic transformation.

This is an image-to-image translation task, not a filter. The generator learns the mapping from photo → comic style from paired training data, conditioned on the input image via the discriminator's adversarial signal.

---

## Architecture

**Generator:** U-Net with skip connections
- Encoder-decoder structure preserves spatial information
- Skip connections prevent information bottleneck at the latent representation

**Discriminator:** PatchGAN
- Classifies overlapping image patches as real or fake rather than the full image
- Encourages high-frequency detail and sharper local structure in generated outputs

**Loss:** Combined adversarial loss + L1 reconstruction loss
- Adversarial loss: pushes generator to fool the discriminator
- L1 loss: keeps generated output grounded to the input structure

```
Input Portrait ──► Generator (U-Net) ──► Generated Comic Image
                        │
                   Discriminator (PatchGAN)
                        │
               Real/Fake classification
                   (per patch)
```

---

## Training

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Architecture | Pix2Pix cGAN |
| Generator | U-Net |
| Discriminator | PatchGAN |
| Loss | Adversarial + L1 |
| Hardware | Google Colab (GPU) |

---

## Results

The model successfully learns the photo → comic style transformation, producing outputs with coherent facial structure and stylized textures. Edge sharpness and color abstraction improve significantly over baseline epochs.

Sample outputs available in `/samples/`.

---

## Setup

```bash
git clone https://github.com/OkenHaha/face2comic-a-cGAN-approach-to-cartoonify-human-potraits
cd face2comic-a-cGAN-approach-to-cartoonify-human-potraits
pip install -r requirements.txt
```

**Inference:**
```bash
python inference.py --input path/to/portrait.jpg --output output.jpg
```

**Training:**
```bash
python train.py --dataroot ./data --epochs 200
```

---

## Why cGAN for this task?

Standard GANs generate images unconditionally. For style transfer tasks, we need the output to be *conditioned* on the input — the generated comic should correspond to the specific face in the photo, not a random face in comic style. Pix2Pix's conditional framework enforces this correspondence through paired training and the discriminator's conditioning on the input image alongside the generated output.

---

## Academic Context

Submitted as final semester MCA project. The project explores learned artistic style transfer as an application of conditional generative modelling, with a focus on understanding how adversarial training dynamics affect output quality in image translation tasks.

---

## Citation

```
@misc{oken2024face2comic,
  author = {Keithellakpam, Oken},
  title  = {face2comic: A cGAN Approach to Cartoonify Human Portraits},
  year   = {2024},
  url    = {https://github.com/OkenHaha/face2comic-a-cGAN-approach-to-cartoonify-human-potraits}
}
```

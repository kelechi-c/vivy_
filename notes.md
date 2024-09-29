## */well, here we are*..

##### I have always wanted to build/train an audio/music generation model. 
(Probably cus it's not so popular/rampant)

I thought about it first in August, but then I decided to implement from a paper(Stable Audio). Yes, I got overwhelmed(stopped at the autoencoder). 
I then took a step back, did basic audio models(for classification), did some extensive research..
also thought of(and started) **modifying a pretrained LLM to generate audio**.
But I still wanted to build mine from scratch, *just to feel good/satisfied*.
So I started, and this time, it wasn't one paper. I was drawing knowledge and insights from different papers and sources.
And my aim was to do this, while tweaking it to produce the best possible results with as little compute as possible.

Some inspiring papers/research I visited include => 
[AudioLDM, Stable audio, MusicGen, ERNIE-Music, MelFusion, SpeechTokenizer, Encodec, MusicLM, AudioCraft]
+ [Text to Music Audio Generation using Latent Diffusion Model, A re-engineering of AudioLDM Model by Ernan Wang]

However my progress will be slow due to exams. But my first step would be to train a **VAE for audio waveform compression**. 
Then the **CLAP fine-tuning** and the **diffusion model** part.
**OR**
I could just use a **transformer** architecture instead, 
way simpler[just Encodec for tokenization, cross attention for text conditioning, **decoder-only transformer** (as in MusicGen). 
You could even use just a **GPT2/Llama** architecture]

I estimate this would be done by **November**...if I don't divert from this again.

However I **will** be switching between image and audio generation from time to time, but still centering around **diffusion and AR/transformer models**.

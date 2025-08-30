# Claim Verification Report

## Summary

- Total Claims: 40
- Supported: 17 (42.5%)
- Partially Supported: 17 (42.5%)
- Contradicted: 2 (5.0%)
- Undetermined: 4 (10.0%)

## Claims by Reviewer

### Reviewer_5w24

- Total Claims: 12
- Supported: 2 (16.7%)
- Partially Supported: 6 (50.0%)
- Contradicted: 2 (16.7%)
- Undetermined: 2 (16.7%)

### Reviewer_ZhX4

- Total Claims: 13
- Supported: 7 (53.8%)
- Partially Supported: 5 (38.5%)
- Contradicted: 0 (0.0%)
- Undetermined: 1 (7.7%)

### Reviewer_B66v

- Total Claims: 7
- Supported: 5 (71.4%)
- Partially Supported: 1 (14.3%)
- Contradicted: 0 (0.0%)
- Undetermined: 1 (14.3%)

### Reviewer_d8D3

- Total Claims: 8
- Supported: 3 (37.5%)
- Partially Supported: 5 (62.5%)
- Contradicted: 0 (0.0%)
- Undetermined: 0 (0.0%)

## Detailed Results

### Claim 1

**Claim:** Summary: In the present paper, the authors introduce RVQGAN, a neural audio codec that uses a convolutional encoder / decoder along with Residual Vector Quantization as a bottleneck, with a multi scale mel reconstruction loss and different adversarial losses.

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about RVQGAN's architecture and performance improvements.

**Evidence:**
1. wavenet [27] decoder. SoundStream [46] is one of the first universal compression models capable of handling diverse audio types, while supporting varying bitrates using a single model. They use a full...
2. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
3. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...
4. High fidelity neural audio synthesis : Recently, generative adversarial networks (GANs) have emerged as a solution to generate high-quality audio with fast inference speeds, due to the feedforward (pa...

--------------------------------------------------

### Claim 2

**Claim:** They show state of the art performance from 3 to 8kbps, compared with the EnCodec model [8].

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Contradicted
**Confidence:** 0.95
**Justification:** The claim states the model performs at 3-8kbps, but evidence shows performance metrics at higher bitrates, contradicting the claim's bitrate range.

**Evidence:**
1. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...
2. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
3. | Codec    |   Bitrate (kbps) |   Bandwidth (kHz) |   Mel distance ↓ |   STFT distance ↓ |   ViSQOL ↑ |   SI-SDR ↑ |
|----------|----------|----------|----------|----------|----------|----------|
| Pr...
4. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...

--------------------------------------------------

### Claim 3

**Claim:** The key novelties are:
- in each VQ layer, the authors perform the retrieval of the nearest codebook entry into a lower dimension space, and use cosine similarity instead of L2 distance to boost the utilization of the codebooks.

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence mentions using cosine similarity via L2-normalization but does not explicitly state retrieval in a lower dimension space.

**Evidence:**
1. To address this issue, we use two key techniques introduced in the Improved VQGAN image model[44] to improve codebook usage: factorized codes and L2-normalized codes. Factorization decouples code look...
2. ## A Appendix

Modified codebook learning algorithm In our work, we use a modified quantization operation, given by:

<!-- formula-not-decoded -->

Here, W in and W out are projection matrices, with W...
3. Quantization setup: we find that using exponential moving average as the codebook learning method, as in EnCodec[8], results in worse metrics especially for SI-SDR. It also results in poorer codebook ...
4. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...

--------------------------------------------------

### Claim 4

**Claim:** - the author notice that the original technique from Soundstream [45] to select a varying number of quantizers can hurt the full bandwidth performance, and thus select 50% of the time all the quantizers in RVQ.

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence mentions quantizer dropout and its impact on audio quality but does not explicitly confirm the 50% selection of quantizers as described in the claim.

**Evidence:**
1. | Codec       | Sampling rate (kHz)   | Target bitrate (kbps)   | Striding factor   | Frame rate (Hz)   | # of 10-bit codebooks   | Compression factor   |    |
|----------|----------|----------|------...
2. Quantization setup: we find that using exponential moving average as the codebook learning method, as in EnCodec[8], results in worse metrics especially for SI-SDR. It also results in poorer codebook ...
3. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...
4. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...

--------------------------------------------------

### Claim 5

**Claim:** Strengths: - great execution and illustration of the various issues tackled here and the proposed solutions.

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The claim mentions strengths related to execution and solutions, but the evidence lacks direct alignment with these specific points.

**Evidence:**
1. |          |          | ✓          | 1.10          | 1.97          | 3.79       |          |          |          |          |
|          | 1536          | snake        | ✓          | ✗          | 5   ...
2. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
3. For our ablation study, we train each model with a batch size of 12 for 250 k iterations. In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch si...
4. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...

--------------------------------------------------

### Claim 6

**Claim:** Weaknesses: - incremental improvement over previous work: overall method is coming from [45], adversarial losses are a combination of the one from [45] and [8].

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Partially Supported
**Confidence:** 0.70
**Justification:** The claim mentions adversarial losses combining [45] and [8], but the evidence only references [19] (MelGAN) and does not mention [45] or [8].

**Evidence:**
1. | [17]   | Jaehyeon Kim, Jungil Kong, and Juhee Son. Conditional variational autoencoder with adversar- ial learning for end-to-end text-to-speech. In International Conference on Machine Learning , pa...
2. We find that adversarial losses are critical to both the quality of the output audio, as well as the bitrate efficiency. When training with only reconstruction loss, the bitrate efficiency drops from ...
3. High fidelity neural audio synthesis : Recently, generative adversarial networks (GANs) have emerged as a solution to generate high-quality audio with fast inference speeds, due to the feedforward (pa...
4. neural network activations (such as Leaky ReLUs) are known to struggle with extrapolating periodic signals, and exhibit poor out-of-distribution generalization for audio synthesis [21].

To add a peri...

--------------------------------------------------

### Claim 7

**Claim:** The authors however claim novelty: l.59, "we make impactful design changes [...]: multi scale stft discriminator, multi scale losses".

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence mentions multi-scale STFT discriminator and multi-scale losses, but lacks direct confirmation of the claim's specific novelty.

**Evidence:**
1. Architecture: We find that varying the decoder dimension has some effect on performance, with smaller models having consistently worse metrics. However, the model with decoder dimension 1024 has simil...
2. Frequency domain reconstruction loss: while the mel-reconstruction loss [18] is known to improve stability, fidelity and convergence speed, the multi-scale spectral losses[42, 11, 15] encourage modeli...
3. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...
4. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...

--------------------------------------------------

### Claim 8

**Claim:** In Section 3.4, the architecture for the multi scale discriminator is missing.

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Contradicted
**Confidence:** 1.00
**Justification:** The evidence explicitly describes the multi-scale discriminator design in Section 3.4.

**Evidence:**
1. |
| [9]   | Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. arXiv preprint arXiv:2005.00341 , 2020. |
| [10]  | ...
2. Architecture: We find that varying the decoder dimension has some effect on performance, with smaller models having consistently worse metrics. However, the model with decoder dimension 1024 has simil...
3. IEEE Signal Processing Letters , 22(8):1006-1010, 2014. |
| [27]   | Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Ko...
4. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...

--------------------------------------------------

### Claim 9

**Claim:** Paragraph starting 193: this insight has been noted and motivated before in [45], [5] and [40], it doesn't seem like the authors bring any new material evidence here?

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Undetermined
**Confidence:** 0.95
**Justification:** The evidence does not provide information about the novelty of the authors' contribution or prior work cited.

**Evidence:**
1. IEEE Signal Processing Letters , 22(8):1006-1010, 2014. |
| [27]   | Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Ko...
2. | [35]   | Jonathan Shen, Ruoming Pang, Ron J Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, Rj Skerrv-Ryan, et al. Natural tts synthesis by conditioning wav...
3. | [1]   | Andrea Agostinelli, Timo I Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music f...
4. For our ablation study, we train each model with a batch size of 12 for 250 k iterations. In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch si...

--------------------------------------------------

### Claim 10

**Claim:** I'm trying to get a better sense of how exactly the model is utilizing this feature.

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Undetermined
**Confidence:** 0.95
**Justification:** The claim seeks clarification on model feature usage, but evidence focuses on technical specifications and performance metrics without direct insight into internal mechanisms.

**Evidence:**
1. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...
2. We first split our dataset into data sources that we know to be full-band - they are confirmed to contain energy in frequencies up to the desired Nyquist frequency (22.05kHz) of the codec - and data s...
3. Architecture: We find that varying the decoder dimension has some effect on performance, with smaller models having consistently worse metrics. However, the model with decoder dimension 1024 has simil...
4. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...

--------------------------------------------------

### Claim 11

**Claim:** Limitations: authors properly address societal impact.

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The evidence discusses audio compression and model improvements but does not explicitly address societal impact.

**Evidence:**
1. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
2. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
3. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...
4. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...

--------------------------------------------------

### Claim 12

**Claim:** Soundness: 3 good

Presentation: 3 good

Contribution: 2 fair

Ethics Review Flagged: ['No ethics review needed.']

**Reviewer:** Reviewer_5w24 (ID: 2GorIEtexk)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the claim's metrics and comparisons with baselines.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
3. We train our model on a large dataset compiled of speech, music, and environmental sounds. For speech, we use the DAPS dataset [26], the clean speech segments from DNS Challenge 4 [10], the Common Voi...
4. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...

--------------------------------------------------

### Claim 13

**Claim:** Summary: This paper introduces a novel high-fidelity neural audio compression algorithm that achieves impressive compression ratios while maintaining audio quality.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about high-fidelity audio compression with impressive ratios and quality maintenance.

**Evidence:**
1. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...
2. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
3. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...
4. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...

--------------------------------------------------

### Claim 14

**Claim:** The authors combine advancements in high-fidelity audio generation with improved vector quantization techniques from the image domain, along with enhanced adversarial and reconstruction losses.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about combining audio generation, vector quantization, and improved losses.

**Evidence:**
1. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...
2. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...
3. wavenet [27] decoder. SoundStream [46] is one of the first universal compression models capable of handling diverse audio types, while supporting varying bitrates using a single model. They use a full...
4. High fidelity neural audio synthesis : Recently, generative adversarial networks (GANs) have emerged as a solution to generate high-quality audio with fast inference speeds, due to the feedforward (pa...

--------------------------------------------------

### Claim 15

**Claim:** Their approach achieves a remarkable 90x compression of 44.1 KHz audio into tokens at just 8kbps bandwidth.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The abstract and introduction directly mention 90x compression of 44.1 KHz audio at 8kbps bandwidth.

**Evidence:**
1. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...
2. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
3. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
4. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...

--------------------------------------------------

### Claim 16

**Claim:** The authors conduct a thorough comparison with competing audio compression algorithms and demonstrate the superior performance of their method.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence states the method outperforms competing codecs in objective and subjective metrics.

**Evidence:**
1. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
2. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
3. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...
4. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...

--------------------------------------------------

### Claim 17

**Claim:** Furthermore, they provide detailed ablations for each design choice, allowing readers to gain insights into the effectiveness of different components.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence mentions ablation studies and model configurations but does not explicitly state that detailed ablations provide insights into design choices.

**Evidence:**
1. For our ablation study, we train each model with a batch size of 12 for 250 k iterations. In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch si...
2. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...
3. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...
4. We first split our dataset into data sources that we know to be full-band - they are confirmed to contain energy in frequencies up to the desired Nyquist frequency (22.05kHz) of the codec - and data s...

--------------------------------------------------

### Claim 18

**Claim:** Strengths: - **Impressive compression performance**: The proposed algorithm achieves a 90x compression ratio for 44.1 KHz audio at just 8kbps bandwidth, demonstrating its effectiveness in reducing data size while preserving audio quality.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The claim mentions 90x compression at 8kbps, which is supported by the abstract but lacks specific metrics from the evidence snippets.

**Evidence:**
1. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
2. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
3. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
4. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...

--------------------------------------------------

### Claim 19

**Claim:** - **Novel Method**: The proposed "codebook collapse" and "quantizer dropout" effectively address the issues in lossy audio compression.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about codebook collapse and quantizer dropout addressing issues in lossy audio compression.

**Evidence:**
1. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
2. Quantization setup: we find that using exponential moving average as the codebook learning method, as in EnCodec[8], results in worse metrics especially for SI-SDR. It also results in poorer codebook ...
3. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...
4. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...

--------------------------------------------------

### Claim 20

**Claim:** - **Comprehensive evaluation**: The authors compare their method against existing audio compression algorithms, demonstrating its superiority in terms of performance.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the method outperforms existing codecs in performance metrics across all bitrates.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
3. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...
4. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...

--------------------------------------------------

### Claim 21

**Claim:** - **Thorough ablations**: The paper provides detailed insights into the impact of design choices, allowing readers to understand the effectiveness of different components and their contributions to the overall results.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The evidence describes ablation studies but does not explicitly state that the insights provide a full understanding of component effectiveness.

**Evidence:**
1. For our ablation study, we train each model with a batch size of 12 for 250 k iterations. In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch si...
2. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...
3. We first split our dataset into data sources that we know to be full-band - they are confirmed to contain energy in frequencies up to the desired Nyquist frequency (22.05kHz) of the codec - and data s...
4. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...

--------------------------------------------------

### Claim 22

**Claim:** Weaknesses: - The novelty of the proposed model structure is a combination of existing models: 
  - factorized codes and L2-normalized codes are from Improved VQGAN image model;
  - Snake activation function from BigVGAN
- This paper presents a strong audio compression technique.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The claim mentions combining existing models but the evidence shows specific improvements and innovations in audio compression techniques.

**Evidence:**
1. neural network activations (such as Leaky ReLUs) are known to struggle with extrapolating periodic signals, and exhibit poor out-of-distribution generalization for audio synthesis [21].

To add a peri...
2. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
3. To address this issue, we use two key techniques introduced in the Improved VQGAN image model[44] to improve codebook usage: factorized codes and L2-normalized codes. Factorization decouples code look...
4. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...

--------------------------------------------------

### Claim 23

**Claim:** However, since the proposed novel points are specifically tailored for a narrow domain, their impact may be limited to the machine learning community and other domains like computer vision/NLP

Questions: - Have you attempted to apply a similar architecture to the vocoder in TTS?

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Undetermined
**Confidence:** 0.75
**Justification:** No evidence directly addresses applying the architecture to vocoder in TTS.

**Evidence:**
1. | [35]   | Jonathan Shen, Ruoming Pang, Ron J Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, Rj Skerrv-Ryan, et al. Natural tts synthesis by conditioning wav...
2. IEEE Signal Processing Letters , 22(8):1006-1010, 2014. |
| [27]   | Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Ko...
3. Language modeling of natural signals : Neural language models have demonstrated great success in diverse tasks such as open-ended text generation [6] with in-context learning capabilities. A key-compo...
4. In 2021 IEEE Spoken Language Technology Workshop (SLT) , pages 492-498. IEEE, 2021. |
| [44]   | Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason B...

--------------------------------------------------

### Claim 24

**Claim:** Limitations: The authors have adequately addressed the limitations.

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The claim mentions addressing limitations, but the evidence focuses on model improvements without explicitly stating all limitations were resolved.

**Evidence:**
1. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
2. IEEE Signal Processing Letters , 22(8):1006-1010, 2014. |
| [27]   | Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Ko...
3. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...
4. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...

--------------------------------------------------

### Claim 25

**Claim:** Soundness: 3 good

Presentation: 3 good

Contribution: 3 good

Ethics Review Flagged: ['No ethics review needed.']

**Reviewer:** Reviewer_ZhX4 (ID: YjQqdH8Z78)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the claim's metrics and performance across all categories with no contradictions.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
3. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...
4. wavenet [27] decoder. SoundStream [46] is one of the first universal compression models capable of handling diverse audio types, while supporting varying bitrates using a single model. They use a full...

--------------------------------------------------

### Claim 26

**Claim:** Summary: This paper introduces a RVQGAN-based neural audio codec method, demonstrating superior audio reconstruction quality, a high compression rate, and generalization across diverse audio domains.

**Reviewer:** Reviewer_B66v (ID: 3GDoZw7Tv9)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about superior audio quality, high compression, and domain generalization.

**Evidence:**
1. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
2. ## High-Fidelity Audio Compression with Improved RVQGAN

Rithesh Kumar*

Descript, Inc.

Prem Seetharaman* Descript, Inc.

Alejandro Luebs

Descript, Inc.

Ishaan Kumar

Descript, Inc.

Kundan Kumar

...
3. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
4. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...

--------------------------------------------------

### Claim 27

**Claim:** The authors substantiate the significant performance superiority of their model over alternatives through extensive and thorough qualitative and quantitative experiments.

**Reviewer:** Reviewer_B66v (ID: 3GDoZw7Tv9)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence shows the model outperforms baselines in objective and subjective metrics across bitrates.

**Evidence:**
1. | 1.0          | 8          | ✓          | 1.07 1.07        | 1.80 1.81         | 3.98 3.97  | 9.07 9.04  | 99%          | 99%          | 99%          |
|          | 1536          | snake        | ✗  ...
2. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
3. For our ablation study, we train each model with a batch size of 12 for 250 k iterations. In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch si...
4. |          |          | ✓          | 1.10          | 1.97          | 3.79       |          |          |          |          |
|          | 1536          | snake        | ✓          | ✗          | 5   ...

--------------------------------------------------

### Claim 28

**Claim:** They present and validate their technique to fully utilize residual vector quantization, alongside model, discriminator, and loss design choices for enhanced performance.

**Reviewer:** Reviewer_B66v (ID: 3GDoZw7Tv9)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence discusses RVQ and design choices but does not explicitly confirm full utilization of residual vector quantization as claimed.

**Evidence:**
1. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...
2. ## A Appendix

Modified codebook learning algorithm In our work, we use a modified quantization operation, given by:

<!-- formula-not-decoded -->

Here, W in and W out are projection matrices, with W...
3. Quantization setup: we find that using exponential moving average as the codebook learning method, as in EnCodec[8], results in worse metrics especially for SI-SDR. It also results in poorer codebook ...
4. wavenet [27] decoder. SoundStream [46] is one of the first universal compression models capable of handling diverse audio types, while supporting varying bitrates using a single model. They use a full...

--------------------------------------------------

### Claim 29

**Claim:** Weaknesses: * The authors derived the proposed methods from existing studies and experimentally validate them in the neural audio codec domain.

**Reviewer:** Reviewer_B66v (ID: 3GDoZw7Tv9)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the authors built on existing studies and validated their methods in neural audio codecs.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
3. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
4. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...

--------------------------------------------------

### Claim 30

**Claim:** This approach seems to compromise the scientific novelty of the research.

**Reviewer:** Reviewer_B66v (ID: 3GDoZw7Tv9)

**Result:** Undetermined
**Confidence:** 0.85
**Justification:** The evidence does not directly address the scientific novelty of the research approach.

**Evidence:**
1. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
2. For our ablation study, we train each model with a batch size of 12 for 250 k iterations. In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch si...
3. Moreover, we provide additional insight into the practical behavior of quantizer dropout and it's interaction with RVQ. Firstly, we find that these techniques put together lead the quantized codes to ...
4. neural network activations (such as Leaky ReLUs) are known to struggle with extrapolating periodic signals, and exhibit poor out-of-distribution generalization for audio synthesis [21].

To add a peri...

--------------------------------------------------

### Claim 31

**Claim:** Limitations: The authors have adequately addressed both the limitations of their research and its possible societal impacts.

**Reviewer:** Reviewer_B66v (ID: 3GDoZw7Tv9)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence shows the authors addressed limitations and societal impacts through performance comparisons and model improvements.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
3. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
4. Architecture: We find that varying the decoder dimension has some effect on performance, with smaller models having consistently worse metrics. However, the model with decoder dimension 1024 has simil...

--------------------------------------------------

### Claim 32

**Claim:** Soundness: 4 excellent

Presentation: 3 good

Contribution: 4 excellent

Ethics Review Flagged: ['No ethics review needed.']

**Reviewer:** Reviewer_B66v (ID: 3GDoZw7Tv9)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the claim's metrics and performance across multiple evaluations.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
3. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...
4. We train our model on a large dataset compiled of speech, music, and environmental sounds. For speech, we use the DAPS dataset [26], the clean speech segments from DNS Challenge 4 [10], the Common Voi...

--------------------------------------------------

### Claim 33

**Claim:** Summary: The authors propose a neural audio codec model that demonstrates superior performance compared to previous works, and present experimental results.

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the claim about superior performance and experimental results of the proposed codec.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
3. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
4. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...

--------------------------------------------------

### Claim 34

**Claim:** Strengths: - The authors appropriately explain the problem they aim to address.

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence shows the model outperforms baselines in some metrics but lacks specific details on all comparisons.

**Evidence:**
1. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...
2. |          |          | ✓          | 1.10          | 1.97          | 3.79       |          |          |          |          |
|          | 1536          | snake        | ✓          | ✗          | 5   ...
3. | 1.0          | 8          | ✓          | 1.07 1.07        | 1.80 1.81         | 3.98 3.97  | 9.07 9.04  | 99%          | 99%          | 99%          |
|          | 1536          | snake        | ✗  ...
4. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...

--------------------------------------------------

### Claim 35

**Claim:** Weaknesses: - For a neural audio codec to be utilized like traditional audio codecs, it should not fail in any patterns.

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence shows improvements in audio compression but does not explicitly address failure patterns in neural audio codecs.

**Evidence:**
1. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
2. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
3. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
4. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...

--------------------------------------------------

### Claim 36

**Claim:** Although the authors divided the original dataset into a training set and evaluation set, it is necessary to validate whether the proposed audio codec works well on more diverse and completely different audio data.

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The evidence shows the codec works well on diverse data but mentions specific challenges with environmental sounds and musical instruments.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. We train our model on a large dataset compiled of speech, music, and environmental sounds. For speech, we use the DAPS dataset [26], the clean speech segments from DNS Challenge 4 [10], the Common Voi...
3. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
4. We have presented a high-fidelity universal neural audio compression algorithm that achieves remarkable compression rates while maintaining audio quality across various types of audio data. Our method...

--------------------------------------------------

### Claim 37

**Claim:** Additionally, finding failure cases of previous works and comparing them can serve as strong evidence supporting the superiority of the authors' proposed method.

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence shows performance metrics but lacks direct comparison of failure cases between methods.

**Evidence:**
1. | 1.0          | 8          | ✓          | 1.07 1.07        | 1.80 1.81         | 3.98 3.97  | 9.07 9.04  | 99%          | 99%          | 99%          |
|          | 1536          | snake        | ✗  ...
2. For our ablation study, we train each model with a batch size of 12 for 250 k iterations. In practice, this takes about 30 hours to train on a single GPU. For our final model, we train with a batch si...
3. |          |          | ✓          | 1.10          | 1.97          | 3.79       |          |          |          |          |
|          | 1536          | snake        | ✓          | ✗          | 5   ...
4. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...

--------------------------------------------------

### Claim 38

**Claim:** Questions: Based on the MUSHRA scores curves, it appears that higher bitrates yield better scores, and the highest quality that this method can achieve remains unconfirmed.

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Partially Supported
**Confidence:** 0.95
**Justification:** The evidence shows higher bitrates yield better MUSHRA scores, but the highest quality remains unconfirmed as stated in the claim.

**Evidence:**
1. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...
2. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
3. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
4. | Ablation on         | Decoder dim. | Activation   | Multi-period   | Single-scale   | # of STFT bands   | Multi-scale mel. | Latent dim   | Quant. method   | Quant. dropout   | Bitrate (kbps)   | Ba...

--------------------------------------------------

### Claim 39

**Claim:** Limitations: Limitations have been well described.

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about limitations being well described.

**Evidence:**
1. Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencod...
2. - We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality...
3. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
4. Frequency domain reconstruction loss: while the mel-reconstruction loss [18] is known to improve stability, fidelity and convergence speed, the multi-scale spectral losses[42, 11, 15] encourage modeli...

--------------------------------------------------

### Claim 40

**Claim:** Soundness: 3 good

Presentation: 3 good

Contribution: 3 good

Ethics Review Flagged: ['No ethics review needed.']

**Reviewer:** Reviewer_d8D3 (ID: VRIhZMXFoC)

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the claim's metrics and performance across all categories with no contradictions.

**Evidence:**
1. We now compare the performance of our final model with competitive baselines: EnCodec [8], Lyra [46], and Opus [37], a popular open-source audio codec. For EnCodec, Lyra, and Opus, we use publicly ava...
2. 3. STFT distance: distance between log magnitude spectrograms of the reconstructed and ground truth waveforms. We use window lengths [2048 , 512] . This metric captures the fidelity in higher frequenc...
3. In , we show the result of our MUSHRA study, which compares EnCodec to our proposed codec at various bitrates. We find that our codec achieves much higher MUSHRA scores than EnCodec at all bitrates. H...
4. wavenet [27] decoder. SoundStream [46] is one of the first universal compression models capable of handling diverse audio types, while supporting varying bitrates using a single model. They use a full...

--------------------------------------------------


# Methods and models reference

This file lists **all registered detectors** and **built-in benchmark dataset classes** implemented in DetectZoo. Every detector follows `detector.predict(input) → DetectionResult` (see the main [README](README.md)).

Load a detector by name:

```python
from detectzoo import load_detector

det = load_detector("fast_detectgpt")
```

List names programmatically with `list_detectors()` and `list_detectors("text" | "image" | "audio")`.

---

## Detectors

### Text

Detectors for identifying LLM-generated text. Each typically accepts a string (or file path).

**Zero-shot statistical methods**

| Name | Class | Method |
|------|-------|--------|
| `log_likelihood` | `LogLikelihoodDetector` | Average token log-probability under a causal LM. Lower perplexity → higher score. |
| `log_rank` | `LogRankDetector` | Average log-rank of observed tokens in the predicted distribution. |
| `rank` | `RankDetector` | Average raw token rank (no log transform). Distinct from log-rank. |
| `entropy` | `EntropyDetector` | Average predictive entropy. Machine text tends to have lower entropy. |
| `lrr` | `LRRDetector` | Log-Likelihood Ratio: −LL / LogRank. Combines two signals into one score. |
| `lastde` | `LastdeDetector` | Multiscale Distribution Entropy of token log-probability sequences. Training-free; measures regularity of the probability landscape via orbit cosine-similarity histograms. |
| `gecscore` | `GECScoreDetector` | Grammar Error Correction scoring. Corrects grammar with a GEC model and measures ROUGE-2 similarity — AI text needs fewer corrections. (Wu et al., COLING 2025) |
| `irm` | `IRMDetector` | Implicit Reward Model. Log-likelihood ratio between an instruction-tuned model and its base counterpart via DPO theory. (Liu et al., NeurIPS 2025) |
| `biscope` | `BiScopeDetector` | Bidirectional cross-entropy. Measures both forward (next-token) and backward (memorisation) CE signals from a causal LM. (Guo et al., NeurIPS 2024) |
| `tocsin` | `TOCSINDetector` | Token Cohesiveness. Measures semantic difference after random token deletion via BARTScore, combined with Fast-DetectGPT curvature. (Ma & Wang, EMNLP 2024) |
| `ipad` | `IPADDetector` | Inverse Prompt for AI Detection. Inverts the likely prompt and scores prompt-text consistency. (Fan et al., 2025) |

**Perturbation / distribution-based methods**

| Name | Class | Method |
|------|-------|--------|
| `detectgpt` | `DetectGPTDetector` | Perturbation-based probability curvature. Uses T5 to generate perturbations and measures log-prob curvature. |
| `fast_detectgpt` | `FastDetectGPTDetector` | Perturbation-free curvature. Estimates curvature from the model's own conditional distribution without generating perturbations. |
| `adadetectgpt` | `AdaDetectGPTDetector` | Adaptive DetectGPT. Extends Fast-DetectGPT with a learned B-spline witness function for improved detection power. (Jin et al., NeurIPS 2025) |
| `npr` | `NPRDetector` | Normalized Perturbation Rank. Like DetectGPT but uses log-rank instead of log-probability. |
| `lastde_pp` | `LastdePPDetector` | Distribution-based extension of Lastde — samples alternative tokens from the model's distribution and computes a normalised discrepancy (like Fast-DetectGPT but using MDE as the scoring function). |
| `glimpse` | `GlimpseDetector` | Probability Distribution Estimation + Fast-DetectGPT. Estimates full token distributions from top-K log-probs using a geometric tail model. (Bao et al., ICLR 2025) |

**Multi-model / generation-based methods**

| Name | Class | Method |
|------|-------|--------|
| `binoculars` | `BinocularsDetector` | Two-model perplexity ratio. Compares an observer and performer model. |
| `dna_gpt` | `DNAGPTDetector` | Divergent N-Gram Analysis. Truncates text, regenerates continuations, compares log-probs of original vs. re-generated. |
| `dna_detectllm` | `DNADetectLLMDetector` | DNA-inspired mutation-repair paradigm. Constructs an ideal AI sequence and measures repair effort via perplexity and cross-perplexity. (Zhu et al., NeurIPS 2025) |
| `revise_detect` | `ReviseDetector` | Revision-based detection. Uses a seq2seq model to revise text and computes BARTScore similarity — AI text changes less when revised. |
| `raidar` | `RaidarDetector` | Rewriting-invariance detection. Rewrites text with multiple prompts and measures edit distance — AI text is more invariant under rewriting. (Mao et al., ICLR 2024) |
| `ghostbuster` | `GhostbusterDetector` | Multi-model probability features. Uses multiple LMs to extract per-token probability vectors and computes structured features for classification. (Verma et al., NAACL 2024) |

**Layer / representation analysis methods**

| Name | Class | Method |
|------|-------|--------|
| `text_fluoroscopy` | `TextFluoroscopyDetector` | Layer-wise KL divergence analysis. Projects each transformer layer's hidden state to vocabulary space and measures KL divergence between intermediate and final layers. Human text shows higher max-KL. |
| `coco` | `CoCoDetector` | Measures inter-sentence coherence via cosine similarity of hidden-state embeddings. Human text shows more varied coherence patterns than machine text. |
| `phd` | `PHDDetector` | Persistent Homology Dimension. Estimates intrinsic dimension of token embeddings via MST weight scaling — human text has higher intrinsic dimension. (Tulchinskii et al., NeurIPS 2023) |
| `mle_ide` | `MLEDetector` | Maximum Likelihood intrinsic dimension estimation (Levina-Bickel). Uses k-NN distance ratios on token embeddings. (Tulchinskii et al., NeurIPS 2023) |

**Supervised / reward-model methods**

| Name | Class | Method |
|------|-------|--------|
| `roberta_base` | `RobertaBaseDetector` | Pre-trained [RoBERTa Base OpenAI Detector](https://huggingface.co/openai-community/roberta-base-openai-detector). Classifies text as Real/Fake using a RoBERTa-base model fine-tuned on GPT-2 outputs. Also available as `"roberta_openai_base"`. |
| `roberta_large` | `RobertaLargeDetector` | Pre-trained [RoBERTa Large OpenAI Detector](https://huggingface.co/openai-community/roberta-large-openai-detector). Same approach as base but with a larger backbone. Also available as `"roberta_openai_large"`. |
| `radar` | `RADARDetector` | RoBERTa-large fine-tuned jointly with a paraphraser for robustness against paraphrase attacks. |
| `imbd` | `ImBDDetector` | Imitate Before Detect. Fine-tunes GPT-Neo-2.7B with Style Preference Optimization (SPO) to learn machine writing preferences, then uses the analytic sampling discrepancy as the detection score. |
| `remodetect` | `ReMoDetectDetector` | Reward Model detection. Uses a pre-trained reward model (DeBERTa-v3-Large) to score text — aligned LLMs produce text with higher reward scores. (Lee et al., NeurIPS 2024) |
| `detective` | `DeTeCtiveDetector` | Multi-level contrastive learning. Learns embeddings via a 3-level contrastive hierarchy (model → family → label) with KNN inference. (He et al., NeurIPS 2024) |

**OOD-based methods**

| Name | Class | Method |
|------|-------|--------|
| `dsvdd` | `DSVDDDetector` | Deep SVDD. Learns a hypersphere around LLM text embeddings; distance from centre indicates human text (OOD). (Zeng et al., NeurIPS 2025) |
| `hrn` | `HRNDetector` | Holistic Regularised Network. Per-model one-class classifiers with gradient penalty, averaged at inference. (Zeng et al., NeurIPS 2025) |
| `energy_detector` | `EnergyDetector` | Energy-based OOD detection. Uses log-sum-exp of multi-class classifier logits as the energy score. (Zeng et al., NeurIPS 2025) |

### Image

Detectors for AI-generated images (diffusion, GAN, etc.). Each accepts an image file path or a PIL `Image`.

**Artifact / frequency-based methods**

| Name | Class | Method |
|------|-------|--------|
| `cnnspot` | `CNNSpotDetector` | ResNet-50 detector from CNNDetection. Learns generator artifacts that are especially visible in frequency statistics. (Wang et al., CVPR 2020) |
| `lgrad` | `LGradDetector` | Learning on Gradients. Converts RGB images to gradient-domain images before classification to emphasize generator-agnostic high-frequency traces. (Tan et al., CVPR 2023) |
| `npr_deepfake` | `NPRDeepfakeDetector` | Neighboring Pixel Relationships. Detects upsampling artifacts using a residual map between the image and a down-upsampled version. (Tan et al., CVPR 2024) |
| `freqnet` | `FreqNetDetector` | Frequency-aware ResNet detector designed to improve cross-generator generalization through frequency-space learning. (Tan et al., AAAI 2024) |
| `ladeda` | `LaDeDaDetector` | Locally Aware Deepfake Detection Algorithm. Scores local 9×9 patches and pools them into an image-level decision. (Cavia et al., 2024) |
| `safe` | `SAFEDetector` | Simple Preserved and Augmented FEatures. Applies DWT high-pass preprocessing before a lightweight ResNet classifier. (Li et al., KDD 2025) |

**CLIP / representation-based methods**

| Name | Class | Method |
|------|-------|--------|
| `univfd` | `UnivFDDetector` | Universal Fake Image Detector. Uses CLIP ViT-L/14 image features with a linear classifier for broad generator transfer. (Ojha et al., CVPR 2023) |
| `fatformer` | `FatFormerDetector` | Forgery-aware Adaptive Transformer. Adds spatial/frequency adapters and language-guided alignment on top of CLIP. (Liu et al., CVPR 2024) |
| `c2p_clip` | `C2PCLIPDetector` | Category Common Prompt CLIP. Uses CLIP visual features with a trained binary head for deepfake detection. (Tan et al., AAAI 2025) |
| `d3` | `D3Detector` | Discrepancy Deepfake Detector. Compares CLIP features from intact and patch-shuffled image views. (Yang et al., CVPR 2025) |
| `cospy` | `CoSpyDetector` | Combines semantic and pixel features with pretrained CO-SPY weights trained on ProGAN. (Cheng et al., CVPR 2025) |
| `cospy_sd_v1_4` | `CoSpySDV14Detector` | CO-SPY variant with weights for Stable Diffusion v1.4 style data. |
| `aide` | `AIDEDetector` | Hybrid detector combining SRM/DCT artifact features with OpenCLIP ConvNeXt semantic features. (Yan et al., ICLR 2025) |
| `drct` | `DRCTDetector` | Diffusion Reconstruction Contrastive Training. Uses a ConvNeXt detector trained with diffusion-reconstructed contrastive samples. (Chen et al., ICML 2024) |
| `patchcraft` | `PatchCraftDetector` | Rich-vs-poor texture contrast. Detects AI images from texture patch contrast features. (Zhong et al., 2023) |

**Training-free / calibration-based methods**

| Name | Class | Method |
|------|-------|--------|
| `aeroblade` | `AerobladeDetector` | Training-free latent diffusion detector. Scores images by VAE reconstruction error; generated images tend to reconstruct more cleanly. (Ricker et al., CVPR 2024) |
| `manifold_bias` | `ManifoldBiasDetector` | Manifold Induced Biases. Zero-/few-shot detector that estimates alignment with the natural image manifold. (Brokman et al., ICLR 2025) |

### Audio

Detectors for synthetic speech and deepfake audio. Each accepts an audio file path or a `(waveform, sample_rate)` tuple unless noted otherwise.

**End-to-end / graph-based methods**

| Name | Class | Method |
|------|-------|--------|
| `rawnet2` | `RawNet2Detector` | End-to-end sinc-filter front-end with residual blocks and GRU. (Tak et al., Interspeech 2021) |
| `aasist` | `AASISTDetector` | Integrated spectro-temporal graph attention network. (Jung et al., ICASSP 2022) |
| `rawgat_st` | `RawGATSTDetector` | End-to-end spectro-temporal graph attention on raw waveform. (Tak et al., Interspeech 2021) |
| `res_tssdnet` | `ResTSSDNetDetector` | Residual time-domain and spectral-domain dilated network. (Hua et al., 2021) |
| `samo` | `SAMODetector` | Speaker attractor multi-center one-class learning. (Ding et al., 2023) |
| `ast_asvspoof` | `ASTASVspoofDetector` | Audio Spectrogram Transformer fine-tuned on ASVspoof 2019. (Gong et al., 2021) |

**SSL / self-supervised methods**

| Name | Class | Method |
|------|-------|--------|
| `anti_deepfake_wav2vec` | `AntiDeepfakeWav2VecDetector` | SSL post-training of Wav2Vec2-Large on 74k hrs speech. (Ge et al., 2022) |
| `anti_deepfake_hubert` | `AntiDeepfakeHubertDetector` | SSL post-training of HuBERT-XLarge on 74k hrs speech. (Ge et al., 2022) |
| `anti_deepfake_xlsr2b` | `AntiDeepfakeXLSR2BDetector` | SSL post-training of XLS-R-2B on 74k hrs speech. (Ge et al., 2022) |
| `xlsr_sls` | `XLSRSLSDetector` | Sensitive layer selection over XLS-R backbone. (Zhang et al., 2022) |

---

## Built-in datasets

Datasets integrate with `load_dataset(name, ...)` (see registry via `list_datasets()`). Data is downloaded and cached under `.detectzoo_data/` on first use where applicable.

### Text datasets

| Dataset | Class | Description | Auto-download source |
|---------|-------|-------------|----------------------|
| HC3 | `HC3Dataset` | Human vs. ChatGPT answers across multiple domains | Hugging Face (`Hello-SimpleAI/HC3`) |
| HC3 Plus | `HC3PlusDataset` | Extends HC3 with semantic-invariant tasks (summarisation, translation, paraphrase) | GitHub |
| CHEAT | `CHEATDataset` | 35k ChatGPT-written academic abstracts (generation, polish, fusion) | GitHub |
| OpenLLMText | `OpenLLMTextDataset` | ~340k samples from human + GPT-3.5, PaLM, LLaMA, GPT-2 | Zenodo |
| MAGE | `MAGEDataset` | Multi-LLM text detection testbed for in- and out-of-distribution evaluation | Hugging Face (`yaful/MAGE`) |
| M4 | `M4Dataset` | Multi-generator, multi-domain, multi-lingual MGT detection (EACL'24 best resource) | GitHub (`mbzuai-nlp/M4`) |
| RAID | `RAIDDataset` | 10M+ documents, 11 LLMs × 11 genres × 12 adversarial attacks (ACL'24 shared benchmark). **Note:** the original repo (`liamdugan/raid`) withholds test-set labels for leaderboard evaluation; DetectZoo uses a labeled re-split (`Shengkun/Raid_split`) so that test-set ground truth is available for offline evaluation. | Hugging Face (`Shengkun/Raid_split`) |
| L2R | `L2RDataset` | 21-domain human/LLM corpus (GPT-3.5/4o, Gemini 1.5 Pro, Llama-3-70B) from ACL'25 | GitHub (`ranhli/l2r_data`) |
| TuringBench | `TuringBenchDataset` | Human vs. 19 neural generators — binary TT and 20-way AA tasks (EMNLP'21) | Hugging Face (`turingbench/TuringBench`) |
| WritingPrompts | `WritingPromptsDataset` | ~303k human-written stories from r/WritingPrompts | Hugging Face (`euclaise/writingprompts`) |
| XSum | `XSumDataset` | BBC article summaries — human-written source corpus for detection benchmarks | Hugging Face (`EdinburghNLP/xsum`) |

### Image datasets

| Dataset | Class | Description | Auto-download source |
|---------|-------|-------------|----------------------|
| CNNDetection / ForenSynths | `CNNDetectionDataset` | Train/val/test benchmark for CNN-generated image detection. Uses `split="train"`, `"val"`, or `"test"` and optional `partitions=[...]`. Registry name: `cnn_detection` (alias `foren_synths`). | Hugging Face (`sywang/CNNDetection`) / upstream CNNDetection archives |
| AIGCDetect | `AIGCDetectDataset` | PatchCraft / AIGCDetect benchmark with 16 different GAN and diffusion generator partitions. Registry: `aigcdetect`. | ModelScope (`aemilia/AIGCDetectionBenchmark`) |
| GenImage | `GenImageDataset` | Million-scale AI-generated image benchmark with generator partitions such as ADM, BigGAN, Midjourney, VQDM, GLIDE, Stable Diffusion, and Wukong. Registry: `genimage`. | Hugging Face (`ENSTA-U2IS/GenImage`) |
| DRCT-2M | `DRCT2MDataset` | Large-scale diffusion real/fake image pairs (ICML 2024 DRCT paper companion data). Registry: `drct2m`. | ModelScope (`BokingChen/DRCT-2M`) |
| UnivFD Diffusion | `UnivFDDataset` | Univ-FD diffusion evaluation partitions including ADM, LDM, GLIDE, and DALL-E. Registry: `univfd_diffusion`. | Google Drive |
| Self-Synthesis | `SelfSynthesisDataset` | GANGen-Detection benchmark with nine GAN partitions such as AttGAN, BEGAN, SNGAN, STGAN, and others. Registry: `self_synthesis`. | Google Drive |
| Chameleon | `ChameleonDataset` | AIDE paper testset for sanity-checking AI-generated image detectors. Registry: `chameleon`. | Google Drive |

### Audio datasets

| Dataset | Class | Description | Auto-download source |
|---------|-------|-------------|----------------------|
| ASVspoof 2019 | `ASVspoof2019Dataset` | Logical Access (LA) spoofing attacks benchmark — standard anti-spoofing evaluation corpus. Registry: `asvspoof2019`. | Official website |
| FoR | `FoRDataset` | Fake-or-Real speech corpus covering a range of TTS and voice conversion systems. Registry: `for`. | Official website |
| Deepfake-Eval-2024 | `DeepfakeEval2024Dataset` | Multi-modal in-the-wild deepfakes from social media and TrueMedia.org (2024); audio split ~40k clips. Registry: `deepfake_eval_2024`. | Hugging Face (`nuriachandra/Deepfake-Eval-2024`, gated) |

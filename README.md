## A survey on image-text multimodal models
This is the repository of **A survey on image-text multimodal models**, the article offers a thorough review of the current state of research concerning the application of large pretrained models in image-text tasks and provide a perspective on its future development trends. For details, please refer to:

**Vision-Language Models for Vision Tasks: A Survey**  
[Paper](www.baidu.com)
 
[![arXiv](https://img.shields.io/badge/arXiv-2304.00685-b31b1b.svg)](www.baidu.com) 
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)


*Feel free to contact us or pull requests if you find any related papers that are not included here.*



## Abstract

From the perspective of the multimodal, the field of artificial intelligence has
made significant breakthroughs, notably the advent and evolution of large-scale
pretrained models. In the domain of Natural Language Processing (NLP) and
Computer Vision (CV), these models leverage extensive data learning and vast
parameters, operating in a data-driven manner, thereby reducing the reliance on
intricate feature engineering and task-specific model designs. However, the poten-
tial of these large pretrained models is progressively being revealed in image-text
tasks. Numerous studies indicate that the joint representation of images and text
can provide a richer semantic context for information. Hence, the application of
large pretrained models in image-text tasks has garnered widespread attention.
Despite the remarkable success of large pretrained models in image-text tasks,
there remain numerous challenges and issues. This article aims to provide a com-
prehensive review of the development of large pretrained models in image-text
tasks and suggest potential solutions to relevant challenges and problems. We
first delves into the fundamental concepts and developmental history of large pre-
trained models, and then demonstrates the actual effectiveness and application
value of pretrained models in image-text tasks through a series of specific use
cases. The paper will probe deeply into the challenges and limitations faced by
pretrained models. On this basis, potential research directions for large pretrained
models are explored. we hope that the article will offer a thorough review of the
current state of research concerning the application of large pretrained models in
image-text tasks and provide a perspective on its future development trends.

## Citation
If you find our work useful in your research, please consider citing:
```
引用论文
```

## Menu
- [Evolution of Multimodal Models](#evolution-of-multimodal-models)
  - [Early Pre-trained Model](#early-pre-trained-model)
  - [Recent Pre-trained Models](#recent-Pre-trained-models)
  - [Current Pre-trained Models](#current-pre-trained-models)
- [Applications of Multimodal Modelss in Image- Text Tasks](#applications-of-multimodal-modelss-in-image-text-tasks)
  - [Image Caption Generation](#image-caption-generation)
  - [Image-Text Matching](#image-text-matching)
  - [VQA/Visual Reasoning](#vqavisual-reasoning)
  - [Visuall Grounding](#visuall-grounding)
  - [Text-to-Image Generation](#text-to-image-generation)
- [Challenges and future directions of multimodal models in image-text tasks](#)
## Evolution of Multimodal Models
### Early Pre-trained Model
|Paper|Published in|
|---|:---:|
|[Distributed representations of sentences and documents](http://arxiv.org/abs/1405.4053v2)|ICML 2014|
|[Going deeper with convolutions](http://dx.doi.org/10.1109/cvpr.2015.7298594)|CVPR 2015|
|[Deep residual learning for image recognition](http://dx.doi.org/10.1109/cvpr.2016.90)|CVPR 2016|
|[Placing images with refined language models and similarity search with pca-reduced VGG features](https://www.semanticscholar.org/paper/Placing-Images-with-Refined-Language-Models-and-VGG-Kordopatis-Zilos-Popescu/7f05df12dff3defee495507abd4870a0a30c3590)|2016|
|[Learned in translation: Contextualized word vectors](http://arxiv.org/abs/1708.00107v2)|NeurIPS 2017|
|[Attention is all you need](http://arxiv.org/abs/1906.02792v1)|NeurIPS 2017|
|[Enriching word vectors with subword information](http://dx.doi.org/10.1162/tacl_a_00051)|TACL 2018|
|[Deep contextualized word representations](http://dx.doi.org/10.18653/v1/n18-1202)|NAACL 2018|
|[Prediction of user loyalty in mobile applications using deep contextualized word representations](http://dx.doi.org/10.1080/24751839.2021.1981684)|Journal of Information and Telecommunication 2021|
|[Alexnet classifier and support vector regressor for scheduling and power control in multimedia heterogeneous networks](http://dx.doi.org/10.1109/tmc.2021.3123200)|IEEE Trans. on Mobile Comput. 2021|
|[On the performance of googlenet and alexnet applied to sketches](http://dx.doi.org/10.1609/aaai.v30i1.10171)|AAAI 2022|
|[A novel approach for early recognition of cataract using VGG-16 and custom user-based region of interest](http://dx.doi.org/10.1145/3512353.3512356)|APIT 2022|
|[Robust zero-watermarking scheme based on a depthwise overparameterized VGG network in healthcare information security](http://dx.doi.org/10.1016/j.bspc.2022.104478)|Biomedical Signal Processing and Control 2022|
|[Study of adam and adamax optimizers on alexnet architecture for voice biometric authentication system](http://dx.doi.org/10.1109/imcom56909.2023.10035592)|IEEE 2023|
|[Dropout alexnet-extreme learning optimized with fast gradient descent optimization algorithm for brain tumor classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.7543)|2023|
### Recent Pre-trained Models
|Paper|Published in|
|---|:---:|
|[BERT: pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)|NAACL 2019|
|[Roberta: A robustly optimized bert pretraining approach](http://arxiv.org/abs/1907.11692v1)|ICLR 2020|
|[Albert: A lite bert for self-supervised learning of language representations](https://arxiv.org/abs/1909.11942)|ICLR 2020|
|[Spanbert: Improving pre-training by representing and predicting spans](http://dx.doi.org/10.1162/tacl_a_00300)|Transactions of the Association for Computational Linguistics 2020|
|[Revisiting pre-trained models for chinese natural language processing](http://dx.doi.org/10.18653/v1/2020.findings-emnlp.58)|ACL 2020|
|[Language models are unsupervised multitask learners](http://arxiv.org/abs/2010.11855v1)|arXiv 2020|
|[End-to-End Object Detection with Transformers](http://dx.doi.org/10.1007/978-3-030-58452-8_13)|ECCV 2020|
|[An image is worth 16x16 words: Transformers for image recognition at scale](http://arxiv.org/abs/2010.11929v2)|ICLR 2021|
|[Swin transformer: Hierarchical vision transformer using shifted windows](http://dx.doi.org/10.1109/iccv48922.2021.00986)|ICCV 2021|
|[Pre-training with whole word masking for chinese bert](http://dx.doi.org/10.1109/taslp.2021.3124365)|IEEE/ACM Trans. Audio Speech Lang. Process. 2021|
|[Llama: Open and efficient foundation language models](http://arxiv.org/abs/2302.13971v1)|arXiv 2023|
|[Llama 2: Open foundation and fine-tuned chat models](https://arxiv.org/abs/2307.09288)|arXiv 2023|
### Current Pre-trained Models
|Paper|Published in|
|---|:---:|
|[Xlnet: Generalized autoregressive pretraining for language understanding](http://arxiv.org/abs/1906.08237v2)|NeurIPS 2019|
|[Lxmert: Learning cross-modality encoder representations from transformers](http://dx.doi.org/10.18653/v1/d19-1514)|ACL 2019|
|[Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks](https://arxiv.org/abs/1908.02265)|NeurIPS 2019|
|[Pixel-bert: Aligning image pixels with text by deep multi-modal transformers](https://arxiv.org/abs/2004.00849)|arXiv 2020|
|[An image is worth 16x16 words: Transformers for image recognition at scale](http://arxiv.org/abs/2010.11929v2)|ICLR 202q|
|[Vinvl: Revisiting visual representations in vision-language models](http://dx.doi.org/10.1109/cvpr46437.2021.00553)|CVPR 2021|
|[Unifying vision-and-language tasks via text generation](https://arxiv.org/abs/2102.02779)|ICML 2021|
|[Scaling up visual and vision-language representation learning with noisy text supervision](https://arxiv.org/abs/2102.05918)| PMLR 2021|
|[Learning transferable visual models from natural language supervision](http://arxiv.org/abs/2103.00020v1)|PMLR 2021|
|[Mdetrmodulated detection for end-to-end multi-modal understanding](https://arxiv.org/abs/2104.12763)|ICCV 2021|
|[Simvlm: Simple visual language model pretraining with weak supervision](http://arxiv.org/abs/2108.10904v3)|ICLR 2022|
|[Glm: General language model pretraining with autoregressive blank infilling](http://dx.doi.org/10.18653/v1/2022.acl-long.26)|ACL 2022|
|[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)|PMLR 2022|
|[An empirical study of training end-to-end vision-and-language transformers](http://dx.doi.org/10.1109/cvpr52688.2022.01763)|CVPR 2022|
|[Coca: Contrastive captioners are image-text foundation models](https://arxiv.org/abs/2205.01917)|arXiv 2022|
|[Scaling instruction-finetuned language models](https://arxiv.org/abs/2210.11416)|arXiv 2022|
|[Opt: Open pre-trained transformer language models](https://arxiv.org/abs/2205.01068)|arXiv 2022|
|[Palm: Scaling language modeling with pathways](http://arxiv.org/abs/2204.02311v5)|arXiv 2022|
|[Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155v1)|NeurIPS 2022|
|[OFA: unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework](https://arxiv.org/abs/2202.03052)|PMLR 2022|
|[Image as a foreign language: Beit pretraining for all vision and vision-language tasks](https://arxiv.org/abs/2208.10442)|CVPR 2023|
|[Visual instruction tuning](http://arxiv.org/abs/2308.13437v2)|arXiv 2023|
|[Bloom: A 176b-parameter open-access multilingual language model](https://arxiv.org/abs/2211.05100)|arXiv 2023|
|[Sparks of artificial general intelligence: Early experiments with gpt-4](http://arxiv.org/abs/2303.12712v5)|arXiv 2023|
|[Pali: A jointly-scaled multilingual language-image model](https://arxiv.org/abs/2209.06794)|arXiv 2023|
|[Palm 2 technical report](http://arxiv.org/abs/2305.10403v3)|arXiv 2023|
|[Lima: Less is more for alignment](http://arxiv.org/abs/2305.11206v1)|arXiv 2023|
|[mplug2: A modularized multi-modal foundation model across text"](https://arxiv.org/abs/2302.00402)|arXiv 2023|
|[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)|arXiv 2023|
|[Exploring the limits of transfer learning with a unified text-to-text transformer](https://arxiv.org/abs/1910.10683)|arXiv 2023|
|[Glm-130b: An open bilingual pre-trained model](https://arxiv.org/abs/1909.11942)|ICLR 2023|
|[EVA: exploring the limits of masked visual representation learning at scale](http://dx.doi.org/10.1109/cvpr52729.2023.01855)|CVPR2023|
|[Alpaca: A strong, replicable instruction-following model](https://crfm.stanford.edu/2023/03/13/alpaca.html)|2023|

## Applications of Multimodal Modelss in Image- Text Tasks
### Image Caption Generation
|Paper|Published in|
|---|:---:|
|[mplug: Effective and efficient vision-language learning by cross-modal skip-connections](http://dx.doi.org/10.18653/v1/2022.emnlp-main.488)|EMNLP 2022|
|[Visualgpt: Data-efficient adaptation of pretrained language models for image captioning](http://dx.doi.org/10.1109/cvpr52688.2022.01750)|CVPR 2022|
|[BSAM: research on image-text matching method based on bert and self-attention mechanism](http://dx.doi.org/10.1109/smc53654.2022.9945109)|SMC 2022|
|[Haav: Hierarchical aggregation of augmented views for image captioning](http://dx.doi.org/10.1109/cvpr52729.2023.01062)|CVPR 2023|
|[BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models](https://arxiv.org/abs/2301.12597)|arXiv 2023|
|[Toward building general foundation models for language, vision, and vision-language understanding tasks](https://arxiv.org/abs/2301.05065)|arXiv 2023|
|[Momo: A shared encoder model for text, image and multi-modal representations](https://arxiv.org/abs/2304.05523)|arXiv 2023|
### Image-Text Matching
|Paper|Published in|
|---|:---:|
|[Ask me anything: Free-form visual question answering based on knowledge from external sources](http://dx.doi.org/10.1109/cvpr.2016.500)|CVPR 2016|
|[Stacked cross attention for image-text matching](http://dx.doi.org/10.1007/978-3-030-01225-0_13)|ECCV 2018|
|[Cross-modal semantic matching generative adversarial networks for text-to-image synthesis](http://dx.doi.org/10.1109/tmm.2021.3060291)|IEEE Trans. Multimedia 2021|
|[Contrastive cross-modal pre-training: A general strategy for small sample medical imaging](http://dx.doi.org/10.1109/jbhi.2021.3110805)|IEEE J. Biomed. Health Inform. 2021|
|[Neural network decision-making criteria consistency analysis via inputs sensitivity](http://dx.doi.org/10.1109/icpr56361.2022.9956394)|ICPR 2022|
|[Visual relationship detection: A survey](http://dx.doi.org/10.1109/tcyb.2022.3142013)|IEEE Trans. Cybern. 2022|
|[Fine-grained bidirectional attention-based generative networks for image-text matching](https://link.springer.com/chapter/10.1007/978-3-031-26409-2_24)|ECML PKDD 2022|
|[Image-text matching with fine-grained relational dependency and bidirectional attention-based generative networks](http://dx.doi.org/10.1145/3503161.3548058)|MM 2022|
|[Vision-language matching for text-to-image synthesis via generative adversarial networks](http://dx.doi.org/10.1109/tmm.2022.3217384)|IEEE Trans. Multimedia 2022|
|[Similarity reasoning and filtration for image-text matching](http://dx.doi.org/10.1609/aaai.v35i2.16209)|AAAI 2022|
|[Generative label fused network for image-text matching](https://dl.acm.org/doi/10.1016/j.knosys.2023.110280)|2023|
### VQA/Visual Reasoning
|Paper|Published in|
|---|:---:|
|[Fvqa: Fact-based visual question answering](http://dx.doi.org/10.1109/tpami.2017.2754246)|IEEE Trans. Pattern Anal. Mach. Intell. 2017|
|[Towards context-aware interaction recognition for visual relationship detection](http://dx.doi.org/10.1109/iccv.2017.71)|ICCV 2017|
|[Scene graph generation by iterative message passing](http://dx.doi.org/10.1109/cvpr.2017.330)|CVPR 2017|
|[Modeling relationships in referential expressions with compositional modular networks](http://dx.doi.org/10.1109/cvpr.2017.470)|CVPR 2017|
### Visuall Grounding
|Paper|Published in|
|---|:---:|
|[A fast and accurate one-stage approach to visual grounding](http://dx.doi.org/10.1109/iccv.2019.00478)|ICCV 2019|
|[Learning to compose and reason with language tree structures for visual grounding](http://dx.doi.org/10.1109/tpami.2019.2911066)|IEEE Trans. Pattern Anal. Mach. Intell. 2019|
|[Transvg: End-to-end visual grounding with transformers](http://dx.doi.org/10.1109/iccv48922.2021.00179)|ICCV 2021|
|[High-resolution image synthesis with latent diffusion models](http://dx.doi.org/10.1109/cvpr52688.2022.01042)|CVPR 2022|
### Text-to-Image Generation
|Paper|Published in|
|---|:---:|
|[Conditional variational autoencoder for neural machine translation](http://arxiv.org/abs/1812.04405v1)|arXiv 2018|
|[Pre-trained models for natural language processing: A survey](https://arxiv.org/abs/2003.08271)|arXiv 2020|
|[Detectgan: Gan-based text detector for camera-captured document images](http://dx.doi.org/10.1007/s10032-020-00358-w)|IJDAR 2020|
|[Diffusion models beat gans on image synthesis](http://arxiv.org/abs/2105.05233v4)|NeurIPS 2021 workshop|
|[Classifier-free diffusion guidance](https://arxiv.org/abs/2207.12598)|NeurIPS 2021|
|[Face0: Instantaneously conditioning a text-to-image model on a face](https://arxiv.org/abs/2306.06638)|arXiv 2023|
|[Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models](https://arxiv.org/abs/2307.06949)|arXiv 2023|





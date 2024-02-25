## A survey on image-text multimodal models
This is the repository of **A survey on image-text multimodal models**, the article offers a thorough review of the current state of research concerning the application of large pretrained models in image-text tasks and provide a perspective on its future development trends. For details, please refer to:

**A Survey on Image-text Multimodal Models**  
[Paper](https://arxiv.org/abs/2309.15857)
 
[![arXiv](https://img.shields.io/badge/arXiv-2309.15857-b31b1b.svg)](https://arxiv.org/abs/2309.15857) 
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
@misc{guo2023survey,
      title={A Survey on Image-text Multimodal Models}, 
      author={Ruifeng Guo and Jingxuan Wei and Linzhuang Sun and Bihui Yu and Guiyong Chang and Dawei Liu and Sibo Zhang and Zhengbing Yao and Mingjun Xu and Liping Bu},
      year={2023},
      eprint={2309.15857},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Menu
- [Development Process](#evolution-of-multimodal-models)
  - [Technical Evoluation](#early-pre-trained-model)
  - [Evolution of Application Technology](#evolution-of-application-technology)
- [Applications of multimodal Models in Image-Text Tasks](#applications-of-multimodal-models-in-image--text-tasks)
  - [Tasks](#tasks)
  - [Generic Model](#generic-model)
  - [Medical Model](#medical-model)
- [Challenges and future directions of multimodal models in image-text tasks](#challenges-and-future-directions-of-multimodal-models-in-image-text-tasks)
  - [External Factor](#external-factor)
  - [Intrinsic Factor](#intrinsic-factor)

## Development Process

### Technical Evoluation

#### Initial Stage and Early Stage

|Paper|Published in|
|---|:---:|
|[Framing image description as a ranking task: Data, models and evaluation metrics](https://www.ijcai.org/Proceedings/15/Papers/593.pdf)|IJCAI 2015|
|[Mind’s eye: A recurrent visual representation for image caption generation](https://openaccess.thecvf.com/content_cvpr_2015/papers/Chen_Minds_Eye_A_2015_CVPR_paper.pdf)|CVPR 2015|
|[Deep visual-semantic alignments for generating image descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)|CVPR 2015|
|[Show, attend and tell: Neural image caption generation with visual attention](https://proceedings.mlr.press/v37/xuc15.html)|PMLR 2015|
|[Show and tell: A neural image caption generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)|CVPR 2015|

### Attention Mechanism and the Rise of Transformers
|Paper|Published in|
|---|:---:|
|[Bilinear attention networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/96ea64f3a1aa2fd00c72faacf0cb8ac9-Paper.pdf)|NeurIPS 2018|
|[Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks](https://arxiv.org/abs/1908.02265)|NeurIPS 2019|
|[Lxmert: Learning cross-modality encoder representations from transformers](https://aclanthology.org/D19-1514.pdf)|ACL 2019|
|[Visualbert:A simple and performant baseline for vision and languag](https://arxiv.org/abs/1908.03557)|arXiv2019|
|[Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training](https://arxiv.org/abs/1908.06066)|AAAI 2020|
|[VL-BERT: pre-training of generic visual-linguistic representations](https://openreview.net/forum?id=SygXPaEYvH)|ICLR 2020|

### Recent Image-text Multimodal Models
|Paper|Published in|
|---|:---:|
|[Vilt: Vision-and-language transformer without convolution or region supervision](https://proceedings.mlr.press/v139/kim21k/kim21k.pdf)|PMLR 2021|
|[Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)|PMLR 2021|
|[An image is worth 16x16 words: Transformers for image recognition at scale](https://openreview.net/forum?id=YicbFdNTTy)|ICLR 2021|
|[Vlmo: Unified vision-language pre-training with mixture-of-modality-expert](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d46662aa53e78a62afd980a29e0c37ed-Abstract-Conference.html)|NeurlPS 2022|
|[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)|PMLR 2022|
|[OFA: unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework](https://arxiv.org/abs/2202.03052)|PMLR 2022|
|[Learning from fm communications: Toward accurate, efficient, all-terrain vehicle localization](https://dl.acm.org/doi/10.1109/TNET.2022.3187885)|IEEE 2022|
|[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)|arXiv 2023|
|[InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://openreview.net/forum?id=vvoWPYqZJA&referrer=%5Bthe%20profile%20of%20Wenliang%20Dai%5D(%2Fprofile%3Fid%3D~Wenliang_Dai1))|NeurlPS 2023|
|[mplug2: A modularized multi-modal foundation model across text"](https://arxiv.org/abs/2302.00402)|arXiv 2023|
|[Mmap: Multi-modal alignment prompt for cross-domain multi-task learning](https://arxiv.org/pdf/2312.08636.pdf)|arXiv 2023|
|[Image as a foreign language: Beit pretraining for all vision and vision-language tasks](https://arxiv.org/abs/2208.10442)|CVPR 2023|
|[Visual Instruction Tuning](https://openreview.net/forum?id=w0H2xGHlkw)|NeulPS2023|
|[Sparks of artificial general intelligence: Early experiments with gpt-4](http://arxiv.org/abs/2303.12712v5)|arXiv 2023|
|[Minigpt-4: Enhancing vision-language understanding with advanced large language models](https://arxiv.org/abs/2304.10592)|arXiv 2023|
|[Minigpt-5: Interleaved vision-and-language generation via generative vokens](https://openreview.net/forum?id=HKJfSd5hcb)|ICLR 2024|
|[Structure-clip: Enhance multi-modal language representations with structure knowledg](https://arxiv.org/abs/2305.06152)|AAAI 2024|
|[m-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronize](https://arxiv.org/abs/2401.10208)|arXiv 2024|

### Evolution of Application Technology

#### Initial Stage and Early Stage
|Paper|Published in|
|---|:---:|
|[A combined convolutional and recurrent neural network for enhanced glaucoma detection](https://www.nature.com/articles/s41598-021-81554-4)|Nature 2021|

#### Attention Mechanism and the Rise of Transformers
|Paper|Published in|
|---|:---:|
|[ Gloria: A multimodal global-local representation learning framework for label-efficient medical image recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf)|IEEE 2021|
|[Mmbert: Multimodal bert pretraining for improved medical vqa](https://arxiv.org/abs/2104.01394)|IEEE 2021|


#### Recent Image-text Multimodal Models
|Paper|Published in|
|---|:---:|
|[Generalized radiograph representation learning via cross-supervision between images and free-text radiology reports](https://www.nature.com/articles/s42256-021-00425-9)|Nature 2022|
|[Medclip: Contrastive learning from unpaired medical images and text](https://aclanthology.org/2022.emnlp-main.256.pdf)|EMNLP 2022|
|[Roentgen: vision-language foundation model for chest x-ray generatio](https://arxiv.org/abs/2211.12737)|arXiv 2022|
|[Lvit: language meets vision transformer in medical image segmentation](https://ieeexplore.ieee.org/document/10172039/)|IEEE 2023|
|[MMTN: Multi-Modal Memory Transformer Network for Image-Report Consistent Medical Report Generation](https://ojs.aaai.org/index.php/AAAI/article/view/25100)|AAAI 2023|
|[LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://openreview.net/forum?id=GSuP99u2kR)|NeurIPS 2023|
|[XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models](https://arxiv.org/abs/2306.07971)|arXiv 2023|
|[Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data](https://arxiv.org/abs/2308.02463)|arXiv 2023|

## Applications of multimodal Models in Image- Text Tasks
### Tasks
#### Pre-training Task
|Paper|Published in|
|---|:---:|
|[Every picture tells a story: Generating sentences from images](https://link.springer.com/book/10.1007/978-3-642-15561-1)|ECCV 2010|
|[Similarity reasoning and filtration for image-text matching](https://ojs.aaai.org/index.php/AAAI/article/view/16209)|AAAI 2021|
|[Visual relationship detection: A survey](https://ojs.aaai.org/index.php/AAAI/article/view/16209)|AAAI 2021|

#### Model components
|Paper|Published in|
|---|:---:|
|[Very deep convolutional networks for large-scale image recognitio](https://ojs.aaai.org/index.php/AAAI/article/view/16209)|AAAI 2021|
|[Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://openreview.net/forum?id=gZ9hCDWe6ke)|ICLR 2021|

### Generic Model
##### model architecture
|Paper|Published in|
|---|:---:|
|[Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)|PMLR 2021|
|[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)|PMLR 2022|
|[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)|arXiv 2023|
|[Minigpt-4: Enhancing vision-language understanding with advanced large language models](https://arxiv.org/abs/2304.10592)|arXiv 2023|
|[Pandagpt: One model to instruction-follow them all](https://aclanthology.org/2023.tllm-1.2.pdf)|ACL 2023|
|[Mobilevlm: A fast, reproducible and strong vision language assistant for mobile device](https://arxiv.org/abs/2312.16886)|arXiv 2023|
|[Qwen-vl: A frontier large vision-language model with versatile abilities](https://arxiv.org/abs/2308.12966)|arXiv 2023|
|[Minigpt-v2: large language model as a unified interface for vision-language multi-task learning](https://openreview.net/forum?id=nKvGCUoiuW)|ICLR 2024|
|[SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://arxiv.org/abs/2401.12168)|arXiv 2024|
|[Mobilevlm v2: Faster and stronger baseline for vision language mode](https://arxiv.org/abs/2402.03766)|arXiv 2024|
|[Llava-plus: Learning to use tools for creating multimodal agents](https://openreview.net/forum?id=IB1HqbA2Pn)|ICLR 2024|

#### data
|Paper|Published in|
|---|:---:|
|[Im2text: Describing images using 1 million captioned photographs](https://papers.nips.cc/paper_files/paper/2011/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)|NeurlPS 2011|
|[Gqa: A new dataset for real-world visual reasoning and compositional question answerin](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.pdf)|CVPR 2019|
|[The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scal](https://link.springer.com/article/10.1007/s11263-020-01316-z)|IJCV 2020|
|[Fashion iq: A new dataset towards retrieving images by natural
language feedback](https://users.cs.utah.edu/~ziad/papers/cvpr_2021_fashion_iq.pdf)|CVPR 2021|
|[Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts](https://openaccess.thecvf.com/content/CVPR2021/papers/Changpinyo_Conceptual_12M_Pushing_Web-Scale_Image-Text_Pre-Training_To_Recognize_Long-Tail_Visual_CVPR_2021_paper.pdf)|CVPR 2021|
|[Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learnin](https://aclanthology.org/2021.acl-long.202/)|ACL 2021|
|[Wenlan: Bridging vision and language by large-scale multi-modal pre-training](https://arxiv.org/abs/2103.06561)|arXiv 2021|
|[RedCaps: Web-curated image-text data created by the people, for the people](https://openreview.net/forum?id=VjJxBi1p9zh)|NeurlPS 2021|
|[Wit:Wikipedia-based image text dataset for multimodal multilingual machine learning](https://arxiv.org/abs/2103.01913)|arXiv 2021|
|[Flava: A foundational language and vision alignment model](https://openaccess.thecvf.com/content/CVPR2022/papers/Singh_FLAVA_A_Foundational_Language_and_Vision_Alignment_Model_CVPR_2022_paper.pdf)|CVPR 2022|
|[Unimo-2: End-to-end unified vision-language grounded learning](https://aclanthology.org/2022.findings-acl.251/)|ACL 2022|
|[Laion-5b: An open large-scale dataset for training next generation image-text models](https://openreview.net/pdf?id=M3Y74vmsMcY)|NeurlPS 2022|

### Medical Model
#### model architecture
|Paper|Published in|
|---|:---:|
|[Medblip: Bootstrapping language-image pre-training from 3d medical images and text](https://arxiv.org/abs/2305.10799)|arXiv 2023|
|[Med-flamingo: a multimodal medical few-shot learner, in: Machine Learning for Health (ML4H)](https://proceedings.mlr.press/v225/moor23a/moor23a.pdf)|PMLR 2023|
|[Pmc-vqa: Visual instruction tuning for medical visual question answering](https://arxiv.org/abs/2305.10415)|arXiv 2023|
|[Masked vision and language pre-training with unimodal and multimodal contrastive losses for medical visual question answering](https://arxiv.org/abs/2307.05314)|MICCAI 2023|
|[Pmc-clip: Contrastive language-image pre-training using biomedical documents](https://arxiv.org/abs/2303.07240)|MICCAI 2023|
|[Pmc-llama: Further finetuning llama on medical paper](https://arxiv.org/abs/2304.14454)|arXiv 2023|
|[MEDITRON-70B: Scaling Medical Pretraining for Large Language Models](https://arxiv.org/abs/2311.16079)|arXiv 2023|
|[ Biomedgpt:Open multimodal generative pre-trained transformer for biomedicine](https://arxiv.org/abs/2308.09442)|arXiv 2023|
|[LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://openreview.net/forum?id=GSuP99u2kR)|NeurIPS 2023|
|[MedJourney: Counterfactual Medical Image Generation by Instruction-Learning from Multimodal Patient Journeys](https://openreview.net/forum?id=ELlBpc0tfb)|ICLR 2024|

##### data
|Paper|Published in|
|---|:---:|
|[Radiology objects in context (roco): a multimodal image dataset](https://link.springer.com/chapter/10.1007/978-3-030-01364-6_20)|MICCAI 2018|
|[A dataset of clinically generated visual questions and answers about radiology images](https://www.nature.com/articles/sdata2018251)|Nature 2018|
|[Chexpert:A large chest radiograph dataset with uncertainty labels and expert comparison](https://arxiv.org/abs/1901.07031)|AAAI 2019|
|[Mimic-cxr-jpg, a large publicly available database of labeled chest radiograph](https://www.nature.com/articles/s41597-019-0322-0)|Nature 2019|
|[Slake: A Semantically-Labeled Knowledge-Enhanced Dataset For Medical Visual Question Answering](https://ieeexplore.ieee.org/document/9434010?denied=)|IEEE 2021|
|[K-PathVQA: Knowledge-Aware Multimodal Representation for Pathology Visual Question Answering](https://ieeexplore.ieee.org/document/10177927)|IEEE 2022|
|[A foundational multimodal vision language ai assistant for human pathology](https://arxiv.org/abs/2312.07814)|arXiv 2023|
|[One model to rule them all: Towards universal segmentation for medical images with text prompt](https://arxiv.org/abs/2312.17183)|arXiv 2023|
|[Towards generalist foundation model for radiology](https://arxiv.org/abs/2308.02463)|arXiv 2023|

## Challenges and future directions of multimodal models in image-text tasks
### External Factor
#### Challenges for Multimodal Dataset
|Paper|Published in|
|---|:---:|
|[Annotation and processing of continuous emotional attributes: Challenges and opportunitie](https://arxiv.org/abs/2007.14886)|IEEE 2013|
|[Multimodal machine learning: A survey and taxonomy](https://arxiv.org/abs/1705.09406)|IEEE 2018|
|[A survey on deep multimodal learning for computer vision: advances, trends, applications, and datasets](https://link.springer.com/article/10.1007/s00371-021-02166-7)|IEEE 2018|
|[A survey on deep multimodal learning for computer vision: advances, trends, applications, and datasets](https://link.springer.com/article/10.1007/s00371-021-02166-7)|-|
|[Between subjectivity and imposition: Power dynamics in data annotation for computer vision](https://arxiv.org/abs/2007.14886)|CSCW 2020|
|[Algorithmic fairness in computational medicine](https://pubmed.ncbi.nlm.nih.gov/36084616/)|-|
|[Bias and Non-Diversity of Big Data in Artificial Intelligence: Focus on Retinal Diseases](https://pubmed.ncbi.nlm.nih.gov/36651834/)|-|

#### Computational Resource Demand
|Paper|Published in|
|---|:---:|
|[Model compression for deep neural networks: A survey](https://www.mdpi.com/2073-431X/12/3/60)|2023|
|[ survey on model compression for large language model](https://arxiv.org/abs/2308.07633)|arXiv 2023|
|[Weakly supervised machine learning]()|CAAI 2023|
|[Semi-supervised and un-supervised clustering: A review and experimental evaluation](https://dl.acm.org/doi/abs/10.1016/j.is.2023.102178)|Information System 2023|
|[Deep learning model compression techniques: Advances, opportunities, and perspective](https://www.ajol.info/index.php/tjet/article/view/250169)|2023|

### Intrinsic Factor
#### Unique Challenges for Image-Text Tasks
|Paper|Published in|
|---|:---:|
|[Cross-domain image captioning via cross-modal retrieval and model adaptation](https://ieeexplore.ieee.org/document/9292444)|IEEE 2020|
|[Transformers in medical image analysis, Intelligent Medicine ](https://www.sciencedirect.com/science/article/pii/S2667102622000717)|Intelligent Medicine 2022|
|[What you see is what you read? improving text-image alignment evaluatio](https://openreview.net/pdf?id=j5AoleAIru)|NeurIPS 2023|
|[Foundational models in medical imaging: A comprehensive survey and future visio](https://arxiv.org/abs/2310.18689)|arXiv 2023|
|[A scoping review on multimodal deep learning in biomedical images and texts](https://arxiv.org/abs/2307.07362)|arXiv 2023|
|[Transformer Architecture and Attention Mechanisms in Genome Data Analysis: A Comprehensive Review](https://www.mdpi.com/2079-7737/12/7/1033)|2023|
|[Transformers in medical imaging: A survey, Medical Image Analysis](https://arxiv.org/abs/2201.09873)|Medical Image Analysis 2023|
|[A survey of multimodal hybrid deep learning for computer vision: Architectures, applications, trends, and challenges, Information Fusion](https://www.researchgate.net/publication/376993734_A_survey_of_multimodal_hybrid_deep_learning_for_computer_vision_Architectures_applications_trends_and_challenges)|2023|
|[Incorporating domain knowledge for biomedical text analysis into deep learning: A survey](https://dl.acm.org/doi/abs/10.1016/j.jbi.2023.104418)|Journal of Biomedical Informatics 2023|
|[Towards electronic health record-based medical knowledge graph construction, completion, and applications: A literature study](https://www.sciencedirect.com/science/article/abs/pii/S1532046423001247)|https://www.sciencedirect.com/science/article/abs/pii/S1532046423001247|
|[ECOFLAP: EFFICIENT COARSE-TO-FINE LAYER-WISE PRUNING FOR VISION-LANGUAGE MODELS](https://openreview.net/pdf?id=iIT02bAKzv#:~:text=To%20overcome%20this%20limitation%20of,score'%20(Coarse)%20and%20then)|ICLR 2024|
|[A novel attention-based cross-modal transfer learning framework for predicting cardiovascular disease](https://www.sciencedirect.com/science/article/abs/pii/S0010482524000611)|Computers in Biology and Medicine 2024|
|[A survey on hallucination in large vision-language models](https://arxiv.org/abs/2402.00253)|arXiv 2024|

#### Multimodel Alignment and Co-learning
|Paper|Published in|
|---|:---:|
|[Aligning temporal data by sentinel events: discovering patterns in electronic health records](https://dl.acm.org/doi/10.1145/1357054.1357129)|2008|
|[Resilient learning of computational models with noisy labels](https://ieeexplore.ieee.org/document/8738840)|IEEE 2019|
|[A label-noise robust active learning sample collection method for multi-temporal urban land-cover classification and change analysis](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300629)|ISPRS 2020|
|[Bayesian dividemix++ for enhanced learning with noisy labels,](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4538538)|2023|
|[A survey on deep learning in medical image registration: New technologies, uncertainty, evaluation metrics, and beyond](https://arxiv.org/pdf/2307.15615.pdf)|arXiv 2023|
|[On the resurgence of recurrent models for long sequences:Survey and research opportunities in the transformer era](https://arxiv.org/abs/2402.08132)|arXiv 2024|
|[Multi-Modal Machine Learning in Engineering Design: A Review and Future Directions](https://asmedigitalcollection.asme.org/computingengineering/article-abstract/24/1/010801/1169855/Multi-Modal-Machine-Learning-in-Engineering-Design?redirectedFrom=fulltext)|2024|
|[A survey of multimodal information fusion for smart healthcare: Mapping the journey from data to wisdom](https://www.sciencedirect.com/science/article/pii/S1566253523003561)|Information Fusion 2024|

#### Catastrophic Forgetting
|Paper|Published in|
|---|:---:|
|[Multiscale Modeling Meets Machine Learning: What Can We Learn?](https://link.springer.com/article/10.1007/s11831-020-09405-5)|2020|
|[Mitigating Forgetting in Online Continual Learning
with Neuron Calibration](https://proceedings.neurips.cc/paper/2021/file/54ee290e80589a2a1225c338a71839f5-Paper.pdf)|NeurlPS 2021|
|[RDFM: An alternative approach for representing, storing, and maintaining meta-knowledge in web of data](https://www.sciencedirect.com/science/article/abs/pii/S095741742100484X)|2021|
|[CNN Models Using Chest X-Ray Images for COVID-19 Detection: A Survey](https://scholar.google.com/scholar?q=Cnn+models+using+chest+x-ray+images+for+covid-19+detection:+A+survey&hl=en&as_sdt=0&as_vis=1&oi=scholart)|2023|
|[Advancing security in the industrial internet of things using deep progressive neural networks](https://link.springer.com/article/10.1007/s11036-023-02104-y)|2023|
|[A progressive neural network for acoustic echo cancellation](https://ieeexplore.ieee.org/document/10096411/)|IEEE 2023|
|[How our understanding of memory replay evolves](https://journals.physiology.org/doi/abs/10.1152/jn.00454.2022)|2023|
|[Replay as context-driven memory reactivation](https://www.biorxiv.org/content/10.1101/2023.03.22.533833v1)|bioRxiv 2023|
|[Unleashing the power of meta-knowledge: Towards cumulative learning in interpreter training](https://www.researchgate.net/publication/373845767_Unleashing_the_power_of_meta-knowledge_Towards_cumulative_learning_in_interpreter_training)|2023|


#### Model Interpretability and Transparency
|Paper|Published in|
|---|:---:|
|[Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)|2019|
|[Human factors in model interpretability: Industry practices,](https://arxiv.org/abs/2004.11440)|CSCW 2020|
|[Interpretation and visualization techniques for deep learning models in medical imaging](https://pubmed.ncbi.nlm.nih.gov/33227719/)|2021|
|[Case studies of clinical decision-making through prescriptive models based on machine learning](https://www.sciencedirect.com/science/article/abs/pii/S0169260723004959?dgcid=rss_sd_all)|2023|
|[Interpreting black-box models: a review on explainable artificial intelligence](https://link.springer.com/article/10.1007/s12559-023-10179-8)|Cognitive Computation 2023|
|[Terminology, Ontology and their Implementations](https://link.springer.com/book/10.1007/978-3-031-11039-9)|2023|
|[AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers](https://arxiv.org/html/2402.05602v1)|arXiv 2024|
|[From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI](https://arxiv.org/abs/2201.08164)|2023|
#### Model Bias and Fairness Issues
|Paper|Published in|
|---|:---:|
|[Towards fairness-aware federated learning](https://arxiv.org/abs/2111.01872)|arXiv 2021|
|[Toward fairness in artificial intelligence for medical image analysis: identification and mitigation of potential biases in the roadmap from data collection to model deployment](https://pubmed.ncbi.nlm.nih.gov/37125409/)|2023|
|[Evaluating and mitigating unfairness in multimodal remote mental health assessments](https://www.medrxiv.org/content/10.1101/2023.11.21.23298803v1.full)|medRxiv 2023|
|[A Unified Approach to Demographic Data Collection for Research With Young Children Across Diverse Cultures](https://psycnet.apa.org/fulltext/2024-17817-001.html)|Developmental Psychology 2024|
|[Bias Detection and Mitigation within Decision Support System: A Comprehensive Survey](https://ijisae.org/index.php/IJISAE/article/view/3162)|2023|
|[Automated monitoring and evaluation of highway subgrade compaction quality using artificial neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0926580522005337)|Automation in Construction 2023|

## Star History
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=i2vec/A-survey-on-image-text-multimodal-models&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=i2vec/A-survey-on-image-text-multimodal-models&type=Date" />
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=i2vec/A-survey-on-image-text-multimodal-models&type=Date" />
</picture>

## Stargazers
[![Stargazers repo roster for @i2vec/A-survey-on-image-text-multimodal-models](https://reporoster.com/stars/i2vec/A-survey-on-image-text-multimodal-models)](https://github.com/i2vec/A-survey-on-image-text-multimodal-models/stargazers)

## Forkers
[![Forkers repo roster for @i2vec/A-survey-on-image-text-multimodal-models](https://reporoster.com/forks/i2vec/A-survey-on-image-text-multimodal-models)](https://github.com/i2vec/A-survey-on-image-text-multimodal-models/network/members)












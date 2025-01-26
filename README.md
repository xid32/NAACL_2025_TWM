# Temporal Working Memory: Query-Guided Segment Refinement for Enhanced Multimodal Understanding [NAACL 2025]

**[Xingjian Diao](https://xid32.github.io/), Chunhui Zhang, Weiyi Wu, Zhongyu Ouyang, Peijun Qing, Ming Cheng, Soroush Vosoughi, Jiang Gui**

[[Full paper]](https://xid32.github.io/images/publications/Temporal_Working_Memory.pdf)

## Introduction
We introduce **temporal working memory** (TWM), which aims to enhance the temporal modeling capabilities of Multimodal foundation models (MFMs). It selectively retains task-relevant information across temporal dimensions, ensuring that critical details are preserved throughout the processing of video and audio content. The TWM uses a query-guided attention approach to focus on the most informative multimodal segments within temporal sequences. By retaining only the most relevant content, TWM optimizes the use of the model's limited capacity, enhancing its temporal modeling ability. This plug-and-play module can be easily integrated into existing MFMs. With our TWM, nine state-of-the-art models exhibit significant performance improvements across **question answering**, **video captioning**, and **video-text retrieval** tasks.



<p align="center">
<img src="/figs/TMW_pipeline.png" alt="Pipeline Figure" width="100%" height="40%">
</p>

## Project Overview

This code repository includes implementations of the Temporal Working Memory (TWM) mechanism algorithm applied to nine different state-of-the-art models. The steps to run the code are as follows:

1. **Download the repository**: Clone this repository to your local environment.
2. **Data Preprocessing**: Prepare data following the preprocessing steps in each original model repository.
3. **Training Temporal Working Memory (TWM)**: For each model, adjust the number of training epochs and relevant model-specific hyperparameters in the `main_alvs.py` file within each model’s directory. Follow the recommendations in each model's original paper for parameter settings, and then train TWM.
4. **Inference**: Set `epochs = 0` in each model's `main_alvs.py` file, and run to utilize TWM.

## References 
### [Paper](Model Name)
1. [Vision Transformers are Parameter-Efficient Audio-Visual Learners](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Vision_Transformers_Are_Parameter-Efficient_Audio-Visual_Learners_CVPR_2023_paper.pdf) (LAVisH)
2. [Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks](https://proceedings.neurips.cc/paper_files/paper/2023/file/af01716e08073368a7c8a62be46dba17-Paper-Conference.pdf) (DG-SCT)
3. [Tackling Data Bias in MUSIC-AVQA: Crafting a Balanced Dataset for Unbiased
Question-Answering](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Tackling_Data_Bias_in_MUSIC-AVQA_Crafting_a_Balanced_Dataset_for_WACV_2024_paper.pdf) (LAST-Att)
4. [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) (Git)
5. [Action knowledge for video captioning with graph neural networks](https://www.sciencedirect.com/science/article/pii/S1319157823000666) (AKGNN)
6. [NarrativeBridge: Enhancing Video Captioning with Causal-Temporal Narrative](https://arxiv.org/abs/2406.06499) (CEN)
7. [VindLU: A Recipe for Effective Video-and-Language Pretraining](https://arxiv.org/abs/2212.05051) (VINDLU)
8. [TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding](https://arxiv.org/abs/2310.19060) (TESTA)
9. [Learning Video Context as Interleaved Multimodal Sequences](https://arxiv.org/abs/2407.21757) (MovieSeq)

## Acknowledgement
We thank the open-sourced paper mentioned above for the authors' outstanding work.

## Citation
If our work has been helpful to your research, we would appreciate it if you could cite the following paper:

```bash
@inproceedings{diao2025twm,
title={Temporal Working Memory: Query-Guided Segment Refinement for
Enhanced Multimodal Understanding},
author={Diao, Xingjian and Zhang, Chunhui and Wu, Weiyi and Ouyang, Zhongyu and Qing, Peijun and Cheng, Ming and Vosoughi, Soroush and Gui, Jiang},
booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
year={2025}
}
```

## Contact
If you have any questions, suggestions, or bug reports, please contact
`xingjian.diao.gr@dartmouth.edu`

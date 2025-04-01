# ACT4IRSpectra

Code Repository for "Analytical-Chemistry-Informed Transformer for Infrared Spectra Modeling", Proceedings of the AAAI Conference on Artificial Intelligence, 2025


# Abstract

Infrared (IR) spectroscopy is a fundamental technique in analytical chemistry. Recently, deep learning (DL) has drawn great interest as the modeling method of infrared spectral data. However, unlike vision or language tasks, IR spectral data modeling is faced with the problem of calibration transfer and has distinctive characteristics. Introducing the prior knowledge of IR spectroscopy could guide the DL methods to learn representations aligned with the domain-invariant characteristics of spectra, and thus improve the performance. Despite such potential, there is a notable absence of DL methods that incorporate such inductive bias. To this end, we propose Analytical-Chemistry-Informed Transformer (ACT) with two modules informed by the field knowledge in analytical chemistry. First, ACT includes learnable spectral processing inspired by chemometrics, which comprises spectral pre-processing, tokenization, and post-processing. Second, a
straightforward yet effective representation learning mechanism, namely spectral-attention, is incorporated into ACT. Spectral-attention utilizes the intra-spectral and inter-spectral correlations to extract spectral representations. Empirical results show that ACT has achieved competitive results in 9 analytical tasks covering applications across pharmacy, chemistry, and agriculture. Compared with existing networks, ACT reduces the root mean square error of prediction (RMSEP) by more than 20% in calibration transfer tasks. These results indicate that DL methods in IR spectroscopy could benefit from the integration of prior knowledge in analytical chemistry.

# Citation

Kindly cite the paper if you use the code. Thanks!
```
@inproceedings{huang2025act，
  title={Analytical-Chemistry-Informed Transformer for Infrared Spectra Modeling},
  author={Huang, Shiluo and Jin, Yining and Jin, Wei and Mu, Ying},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025},                           
}
```

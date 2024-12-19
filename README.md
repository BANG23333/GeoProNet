# GeoProNet

# GeoPro-Net
GeoPro-Net: Learning Interpretable Spatiotemporal Model through Statistically-Guided Geo-Prototyping

# Abstract
The problem of forecasting spatiotemporal events such as crimes and accidents is crucial to public safety and city management. Besides accuracy, interpretability is also a key requirement for spatiotemporal forecasting models to justify the decisions. Interpretation of the spatiotemporal forecasting mechanism is, however, challenging due to the complexity of multi-source spatiotemporal features, the non-intuitive nature of spatiotemporal patterns for non-expert users, and the presence of spatial heterogeneity in the data.
Currently, no existing deep learning model intrinsically interprets the complex predictive process learned from multi-source spatiotemporal features. To bridge the gap, we propose GeoPro-Net, an intrinsically interpretable spatiotemporal model for spatiotemporal event forecasting problems. GeoPro-Net introduces a novel Geo-concept convolution operation, which employs statistical tests to extract predictive patterns in the input as ``Geo-concepts'', and condenses the ``Geo-concept-encoded'' input through interpretable channel fusion and geographic-based pooling. In addition, GeoPro-Net learns different sets of prototypes of concepts inherently, and projects them to real-world cases for interpretation. Comprehensive experiments and case studies on four real-world datasets demonstrate that GeoPro-Net provides better interpretability while still achieving competitive prediction performance compared with state-of-the-art baselines.

![alt text](https://github.com/BANG23333/GeoPro-Net/blob/main/img/framework.jpg)

# Environment
- python 3.7.0
- torch 1.12.1
- matplotlib 3.5.2
- numpy 1.21.5
- sklearn 1.1.1
- seaborn 0.11.2

# Run GeoPro-Net
Please follow the readme page \
Start with GeoPro-Net.py

Data can be downloaded through [NYC](https://drive.google.com/file/d/17Sc5TBSmqqzxI30HfzvJr6xkhVN0ehFm/view?usp=sharing), [Chicago](https://drive.google.com/file/d/1mMhyRK2gDsIFUO9AajWX2cCEntzxB24J/view?usp=sharing)

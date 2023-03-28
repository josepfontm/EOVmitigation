# EOVmitigation

**Introduction:** This repository contains several EOV Procedures which I have used in different publications/works. Additional functions which aid in plotting results are also included.

In the future I intend to merge these methods into the [Pymodal library](https://github.com/grcarmenaty/pymodal).

## Interpreting Environmental Variability from Damage Sensitive Features
Results presented in the 10th ECCOMAS Thematic Conference on Smart Structures and Materials (SMART 2023)

### Abstract:
Mitigation of Environmental and Operational Variabilities (EOVs) remains one of the main challenges to adopt Structural Health Monitoring (SHM) technologies. Its implementation in wind turbines is one of the most challenging due to the adverse weather and operating conditions these structures have to face. This work proposes an EOV mitigation procedure based on Principal Component Analysis (PCA) which uses EOV-Sensitive Principal Components (PCs) as a surrogate of EOVs, which may be hard to measure or correctly quantify in real-life structures. EOV-Sensitive PCs are conventionally disregarded in an attempt to mitigate the effect of Environmental and Operational Variabilities. Instead, this method proposes their use as independent variables in non-linear regression models, similar to how Environmental and Operational Parameters (EOPs) are used in explicit procedures.

The work results are validated under an experimental dataset of a small-scale wind turbine blade with various cracks artificially introduced. Temperature conditions are varied using a climate chamber. The proposed method outperforms the conventional-PCA based approach, implying that directly disregarding Sensitive-EOV PCs is detrimental in the decision-making within a SHM methodology. In addition, the proposed method achieves similar results to an equivalent explicit procedure, suggesting that EOV-Sensitive PCs can replace directly measured EOVs.

### EOV Procedures:
- [Implicit PCA](https://www.sciencedirect.com/science/article/abs/pii/S0888327004001785): Conventionally, the first Principal Components (PCs) are disregarded to correct Damage Sensitive Features (DSFs). The rationale behind Implicit PCA is that PCs can be categorized between EOV-Sensitive, EOV-Insensitive and Noise, in this order [[1]](#1).

- Explicit PCA Regression: A non-linear method is used to find the best fitting polynomial function for the data in the least squares sense. Temperature (or other EOVs) are used as independent variables (predictors), while Principal Components are used as dependent or explained variables. The following papers describes a similar method, but using natural frequencies as DSFs, instead of PCA results.

- PC-Informed Regression (PROPOSED): In this publication, we proposed a method that uses the so-called EOV-Sensitive PCs as a surrogate of the Environmental and Operational variables driving the non-stanionary behaviour in the DSFs. Hence, a regression model using EOV-Sensitive PCs as predictors and remaining PCs as explained variables.

### References:
<a id="1">[1]</a> 
A.M. Yan, G. Kerschen, P. De Boe and J.C. Golinval (2005). 
Structural damage diagnosis under varying environmental conditions - part i: A linear analysis 
Mech. Syst. Signal Porcess.,vol. 19, no. 4,pp. 847-864


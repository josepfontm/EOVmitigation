# EOVmitigation

## Interpreting Environmental Variability from Damage Sensitive Features
Results presented in the [10th ECCOMAS Thematic Conference on Smart Structures and Materials (SMART 2023)](https://www.researchgate.net/publication/373757121_Interpreting_Environmental_Variability_from_Damage_Sensitive_Features). Additional information regarding the project can be found in this [presentation](https://www.researchgate.net/publication/372077222_Interpreting_environmental_variability_from_damage-sensitive_features).
The experimental data used in this work was extracted from the [Small-scale wind turbine blade](https://onlinelibrary.wiley.com/doi/epdf/10.1002/stc.2660/) provided by the [Chair of Structural Mechanics and Monitoring](https://chatzi.ibk.ethz.ch/) from ETH Zürich.

### Abstract:
Mitigation of Environmental and Operational Variabilities (EOVs) remains one
of the main challenges to adopt Structural Health Monitoring (SHM) technologies. Its implementation
in wind turbines is one of the most challenging due to the adverse weather and
operating conditions these structures have to face. This work proposes an EOV mitigation
procedure based on Principal Component Analysis (PCA), which uses EOV-Sensitive Principal
Components (PCs) as a surrogate of EOVs, which may be hard to measure or correctly
quantify in real-life structures. EOV-Sensitive PCs are conventionally disregarded in an attempt
to mitigate the effect of environmental variability. Instead, we postulate to use of these
variables as predictors in non-linear regression models, similar to how Environmental and Operational
Parameters (EOPs) are used in explicit EOV mitigation procedures. 

The work results are validated under an experimental dataset of a small-scale wind turbine blade with various
cracks artificially introduced. Temperature conditions are varied using a climate chamber. The
proposed method outperforms the conventional-PCA-based approach, implying that directly disregarding
Sensitive-EOV PCs is detrimental in the decision-making within a SHM methodology.
In addition, the proposed method achieves similar results to an equivalent explicit procedure,
suggesting that EOV-Sensitive PCs can replace directly measured EOVs.

### EOV Procedures:
The literature regarding EOV Mitigation is extensive. Nonetheless, some examples are presented here for clarity's sake.
The following [review](https://link.springer.com/chapter/10.1007/978-3-030-81716-9_15) serves as a good starting point to delve into the world of EOV Mitigation in Data-Driven SHM.

Specific works on the available approaches:
- Implicit PCA: Conventionally, the first Principal Components (PCs) are disregarded to correct Damage Sensitive Features (DSFs). The rationale behind Implicit PCA is that PCs can be categorized between EOV-Sensitive, EOV-Insensitive and Noise, in this order [[1]](#1).

- Explicit PCA Regression: A non-linear method is used to find the best fitting polynomial function for the data in the least squares sense. Temperature (or other EOVs) are used as independent variables (predictors), while Principal Components are used as dependent or explained variables. The following papers describes a similar method, but using natural frequencies as DSFs, instead of PCA results [[2]](#2).

Our proposal:
- PC-Informed Regression: In this publication, we proposed a method that uses the so-called EOV-Sensitive PCs as a surrogate of the Environmental and Operational variables driving the non-stationary behaviour in the DSFs. Hence, a regression model using EOV-Sensitive PCs as predictors and remaining PCs as explained variables. Previous works have explored the use of natural frequencies as surrogate variables for bridge damage detection [[3]](#3).

### Pymodal:
The EOV Procedures that can be found in this repo have been added to the [Pymodal library](https://github.com/grcarmenaty/pymodal), an on-going project from the [Group of Applied Mechanics and Advanced Manufacturing (GAM)](https://techtransfer.iqs.edu/grupos/applied-mechanics-and-advanced-manufacturing) at IQS School of Engineering-URL.

### References:
<a id="1">[1]</a> 
A.M. Yan, G. Kerschen, P. De Boe and J.C. Golinval (2005). 
[Structural damage diagnosis under varying environmental conditions - part i: A linear analysis](https://www.sciencedirect.com/science/article/abs/pii/S0888327004001785)
Mech. Syst. Signal Porcess.,vol. 19, no. 4,pp. 847-864

<a id="2">[2]</a> 
Roberts C, Cava DG, Avendaño-Valencia LD (2023)
[Addressing practicalities in multivariate nonlinear regression for mitigating environmental and operational variations](https://journals.sagepub.com/doi/10.1177/14759217221091907)
Struct. Health Monit.

<a id="3">[3]</a> 
William Soo Lon Wah, Yung-Tsang Chen, John S Owen (2021)
[A regression-based damage detection method for structures subjected to changing environmental and operational conditions](https://www.sciencedirect.com/science/article/pii/S0141029620340633),
Eng. Struct., Volume 228, 111462




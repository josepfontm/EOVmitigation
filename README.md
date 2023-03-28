# EOVmitigation

**Introduction:** This repository contains several EOV Procedures which I have used in different publications/works. In the future I intend to merge these methods into the Pymodal library.

[**TPMSgen**](https://github.com/albertforesg/TPMSgen) is a powerful program based on **Python** that allows users to easily design and generate Triply Periodic Minimal Surface (TPMS) geometries. It features a user-friendly interface with multiple **design parameters** (TPMS typology, specimen dimensions, unit cell size…) that makes it simple to generate their corresponding 3D model employing their mathematical equations. In addition, the program offers the possibility to export the 3D model in the **.STL** file format, which can be later used for fabrication with additive manufacturing technologies or in finite element simulation studies. This makes [**TPMSgen**](https://github.com/albertforesg/TPMSgen) a versatile tool for architects, engineers, and material scientists who are interested in exploring the unique properties of TPMS and their potential applications.

![TPMSgen](https://user-images.githubusercontent.com/81706331/212754604-6bf67f0f-b447-4496-8e0a-cb3c199b3c98.png)

---

##Introduction


##Interpreting Environmental Variability from Damage Sensitive Features

###Abstract:
Mitigation of Environmental and Operational Variabilities (EOVs) remains one of the main challenges to adopt Structural Health Monitoring (SHM) technologies. Its implementation in wind turbines is one of the most challenging due to the adverse weather and operating conditions these structures have to face. This work proposes an EOV mitigation procedure based on Principal Component Analysis (PCA) which uses EOV-Sensitive Principal Components (PCs) as a surrogate of EOVs, which may be hard to measure or correctly quantify in real-life structures. EOV-Sensitive PCs are conventionally disregarded in an attempt to mitigate the effect of Environmental and Operational Variabilities. Instead, this method proposes their use as independent variables in non-linear regression models, similar to how Environmental and Operational Parameters (EOPs) are used in explicit procedures.

The work results are validated under an experimental dataset of a small-scale wind turbine blade with various cracks artificially introduced. Temperature conditions are varied using a climate chamber. The proposed method outperforms the conventional-PCA based approach, implying that directly disregarding Sensitive-EOV PCs is detrimental in the decision-making within a SHM methodology. In addition, the proposed method achieves similar results to an equivalent explicit procedure, suggesting that EOV-Sensitive PCs can replace directly measured EOVs.

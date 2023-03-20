# SemNav-HRO

This repository contains the codes for our paper, which is titled as "SemNav-HRO: A Target-Driven Semantic Navigation Framework with Human-Robot-Object Ternary Fusion". 

# Abstract

Target-Driven Semantic Navigation (TDSN) shows great potential to be applied in intelligent domestic assistants supporting humans with daily activities. Although numerous methods have been explored to utilize visual and semantic clues to achieve efficient static TDSN, socially aware TDSN in dynamic and crowded scenarios still remains challenging and has not been adequately studied. The main challenge comes from the complex human-agent interaction mechanism and semantic relation exploitation, which requires the agent to understand the surroundings and perform foresighted behaviors. In this paper, a TDSN framework named SemNav-HRO is proposed by considering Human-Robot-Object (HRO) ternary feature fusion. To be specific, a Deep Reinforcement Learning (DRL) based dual-channel value estimation network is first proposed by integrating multi-granularity map features and social awareness to learn crowded TDSN strategies. Then, the tricky and socially aware TDSN problem is slackened by eliminating the dependence on costly features (e.g., pedestrian speed) and introducing pedestrian trajectory prediction. Finally, a semantic-rich simulator with complex layouts is constructed for TDSN strategy learning and evaluation in crowded situations based on realistic domestic scenes. In the experimental phase, numerous comparative analyses and benchmark tests demonstrate the superiority of our approach in terms of success rate, collision rate, cumulative rewards et al. In addition, we investigate the effect of semantic map size on navigation performance and verify the generalization and interpretability of navigation inference.

# Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library.
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library.
3. Install crowd_sim and crowd_nav into pip:
```
pip install -e .
```

# Dateset
Coming soon...

# Getting Started
This repository are organized in two parts: crowd_sim/ folder contains the simulation environment and crowd_nav/ folder contains codes for training and testing the policies. Below are the instructions for training and testing policies, and they should be executed inside the crowd_nav/ folder.

Coming soon...

# Examples and Demos

Examples of point navigation and target-driven navigation.
<div align="center">
	<img src="./Fig1.png" alt="Editor" width="800">
</div>

Visualization of the navigation process.
<div align="center">
	<img src="./Fig2.png" alt="Editor" width="800">
</div>

|                Demo of point navigation.             |                 Demo of target-driven navigation.             |
|:----------------------------------------------------:|:-------------------------------------------------------------:|
| <img src="./Demo-Point-Navigation.gif" width="500"/> | <img src="./Demo-Target-driven-Navigation.gif" width="500"/>  |


## Acknowledge
This work is based on [RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning). The authors thank Changan Chen, Sha Hu, Payam Nikdel, Greg Mori and Manolis Savva for their works.

# AMICL & residual attention streams

This repository contains code to simulate the AMICL model and residual attention streams modification of Transformer models, as described in:
**"Associative memory inspires improvements for in-context learning using a novel attention residual stream architecture"**
*by Thomas F Burns, Tomoki Fukai, and Christopher J Earls*

## AMICL

The `AMICL` folder contains a single Jupyter notebook, which, when run in order, contains cells which build the data and model, and generate all plots.

## 2-layer Transformer

In the `toy-Transformer` folder, there is code for the 2-layer Transformer model. This was built using JAX, with the codebase originating from Gautam Reddy, and used in their paper titled, "The mechanistic basis of data dependence and abrupt learning in an in-context classification task" ICLR 2024.

File `datasets_v2.py` specifies the dataset generation, files beginning with `transformer_v2_v_residuals` specify the models, and files beginning with `ic_vs_iw_v3` train and run the models.

## Small LM

Code for simulating the small LMs is contained in the `small-LM` folder and is based on the implementations of Raymond Van and Andrej Karpathy. The modified model is specified and trained in the filenames appended with `_Vpass`. File `IOI_task.py` runs the indirect object indentification tests on model files named therein.

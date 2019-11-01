# IRNet
Code for our ACL'19 accepted paper: [Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation](https://arxiv.org/pdf/1905.08205.pdf)

<p align='center'>
  <img src='https://zhanzecheng.github.io/architecture.png' width="91%"/>
</p>

## Environment Setup

* `Python3.6`
* `Pytorch 0.4.0` or higher

Install Python dependency via `pip install -r requirements.txt` when the environment of Python and Pytorch is setup.

## Running Code

#### Data preparation


* Download [Glove Embedding](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and put `glove.42B.300d` under `./data/` directory
* Download [Pretrained IRNet](https://drive.google.com/open?id=1VoV28fneYss8HaZmoThGlvYU3A-aK31q) and put `
IRNet_pretrained.model` under `./saved_model/` directory
* Download preprocessed train/dev datasets from [here](https://drive.google.com/open?id=1YFV1GoLivOMlmunKW0nkzefKULO4wtrn) and put `train.json`, `dev.json` and 
`tables.json` under `./data/` directory

##### Generating train/dev data by yourself
You could process the origin [Spider Data](https://drive.google.com/uc?export=download&id=11icoH_EA-NYb0OrPTdehRWm_d7-DIzWX) by your own. Download  and put `train.json`, `dev.json` and 
`tables.json` under `./data/` directory and follow the instruction on `./preprocess/`

#### Training

Run `train.sh` to train IRNet.

`sh train.sh [GPU_ID] [SAVE_FOLD]`

#### Testing

Run `eval.sh` to eval IRNet.

`sh eval.sh [GPU_ID] [OUTPUT_FOLD]`


#### Evaluation

You could follow the general evaluation process in [Spider Page](https://github.com/taoyds/spider)


## Results
| **Model**   | Dev <br /> Exact Set Match <br />Accuracy | Test<br /> Exact Set Match <br />Accuracy |
| ----------- | ------------------------------------- | -------------------------------------- |
| IRNet    | 53.2                        | 46.7                      |
| IRNet+BERT(base) | 61.9                          | **54.7**                      |


## Citation

If you use IRNet, please cite the following work.

```
@article{GuoIRNet2019,
  author={Jiaqi Guo and Zecheng Zhan and Yan Gao and Yan Xiao and Jian-Guang Lou and Ting Liu and Dongmei Zhang},
  title={Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation},
  journal={arXiv preprint arXiv:1905.08205},
  year={2019},
  note={version 1}
}
```

## Thanks
We would like to thank [Tao Yu](https://taoyds.github.io/) and [Bo Pang](https://www.linkedin.com/in/bo-pang/) for running evaluations on our submitted models.
We are also grateful to the flexible semantic parser [TranX](https://github.com/pcyin/tranX) that inspires our works.

# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
## Examples in Fig
- See ```sft_dpo_positive_example.ipynb``` and ```negative_toy_example.ipynb```

## BLO PEFT learning
- Our code for BLO PEFT Learning experiment is adapted from bilevel post fine-tuning LLM library BIPOST: [https://github.com/Post-LLM/BIPOST](https://github.com/Post-LLM/BIPOST)

- To run the code, it requires to set up a conda enviroment bipost. Follow the instructions from: [https://github.com/Post-LLM/BIPOST](https://github.com/Post-LLM/BIPOST)

- After setting up the enviroment, run the scripts e.g. ```bilevel_dpo_sft_test.sh``` in ```bipost/examples``` directory. Notably, here, our PBGD-Free algorithm is noted as VaFF (value function free).

- Important code files include: ```bilevel_trainer.py``` in ```bipost/trainer``` and ``train_bilevel.py``  in ```bipost/cli```.

## NLSY79 representation Learning
- Our code is adapted from Fair Representation learning: [https://github.com/cjshui/fair-path](https://github.com/cjshui/fair-path)

- To download the NLSY79 dataset, follow the instructions from: [https://github.com/jkomiyama/fairregresion#3-datasets](https://github.com/jkomiyama/fairregresion#3-datasets)

- After data preprocesssing, run ```main.py```

## BiDoRA
- Run the code files for two dataset we considered on Jupyter Notebook

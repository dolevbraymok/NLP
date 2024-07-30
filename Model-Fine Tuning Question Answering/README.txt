# T5 Fine-Tuning on SQuAD v2
note that the only purpose of this project is learning



## Overview
This repository contains the code and resources for fine-tuning the T5 model on the SQuAD v2 dataset. The goal of this project is to create a question-answering system

## Model
- **Model**: T5-small
- **Dataset**: SQuAD v2
- **Framework**: Hugging Face Transformers


Datasets : SQuAD_v2 for training and eval, and for testing i used validation of SQuAD, some other datasets that could be used are mentioned in the notebook but wasnt used
Both the dataset and the T5 i fine tuned , are from huggingFace dataset and Transformer Library respectivly
as for scores i used Rouge, BLEU and Average Cosine Similarity, the results are in the begining of the ipynb file
some are good, but some (Rouge-2) need improvment

my main purpose for this project is learning and would probablity wont be used/improved anymore after putting it in git


## License

This project utilizes the following models and tools:


- **T5 Model**: Licensed under the Apache 2.0 License. For more information, please refer to the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). You can find more details on the [T5 model page](https://huggingface.co/google-t5/t5-small).
            This model wasnt used "as is" but was first fine tuned on SQuAD v2
## Credits

- **Hugging Face Transformers**: [Hugging Face](https://huggingface.co/transformers/)
- **T5 Model**: [T5 Model on Hugging Face](https://huggingface.co/google-t5/t5-small)
- **SQuAD v2 Dataset**: [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)


## License Summary

the  T5 models are provided under the Apache 2.0 License. This license allows for:
- **Use**: For any purpose, including commercial applications.
- **Modification**: To modify and create derivative works.
- **Distribution**: To distribute the original or modified software.

Conditions include:
- **Attribution**: Proper credit must be given to the original authors.
- **State Changes**: Document any significant changes made to the original code.
- **License Copy**: Include a copy of the Apache 2.0 License in redistributed software.

Limitations:
- **No Warranty**: The software is provided "as-is," without any warranties.

For complete details, please refer to the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
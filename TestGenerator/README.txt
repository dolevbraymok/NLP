This project is a work in progress

# Project: Questions and Answers generation by input context
note that the project purpose is for learning only

## Overview
This repository contains the code and resources for Question generation with flan-T5 model, and for question answering using my own fine-tuned T5-small on SQuAD_v2.
 The goal of this project is to create a question-answering system that can handle non-extractive answers and generate various types of questions.

## Model
- **Models**: T5-small and flan-T5
- **Dataset**: SQuAD v2
- **Framework**: Hugging Face Transformers


## License

This project utilizes the following models and tools:

- **FLAN-T5 Model**: Licensed under the Apache 2.0 License. For more information, please refer to the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). You can find more details on the [FLAN-T5 model page](https://huggingface.co/google/flan-t5-base).

- **T5 Model**: Licensed under the Apache 2.0 License. For more information, please refer to the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). You can find more details on the [T5 model page](https://huggingface.co/google-t5/t5-small).
            This model wasnt used "as is" but was first fine tuned on SQuAD v2
## Credits

- **Hugging Face Transformers**: [Hugging Face](https://huggingface.co/transformers/)
- **FLAN-T5 Model**: [FLAN-T5 on Hugging Face](https://huggingface.co/google/flan-t5-base)
- **T5 Model**: [T5 Model on Hugging Face](https://huggingface.co/google-t5/t5-small)
- **SQuAD v2 Dataset**: [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- **Google Colab**: Used for training and fine-tuning the model.

## License Summary

Both the FLAN-T5 and T5 models are provided under the Apache 2.0 License. This license allows for:
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

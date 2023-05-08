# Improving Natural Language Inference by Removing Extreme Examples and Identifying Mislabeled Data

Natural Language Inference (NLI) is a fundamental task in natural language processing, which involves determining the relationship between a premise and a hypothesis. The Stanford Natural Language Inference (SNLI) dataset is a popular benchmark for evaluating NLI models. However, certain examples in the dataset may be too easy or too hard, potentially leading to overfitting or underfitting issues.

In this study, we investigate the impact of removing extreme examples from the SNLI dataset before training an NLI model. We hypothesize that removing the hardest 5% and easiest 45% of examples will result in a more balanced and informative training set, leading to better generalization and performance on the original dataset. Furthermore, we employ GPT-3.5 to identify mislabeled instances within the hardest 5% of examples and analyze the impact of correcting these mislabeled instances on model performance.


## Cartography

We begin by loading the SNLI dataset and filtering out instances with -1 labels. In order to identify the hardest and easiest examples in the dataset, we follow the approach outlined in AI2's Cartography paper (Swayamdipta et al., 2020). We train a model on the SNLI dataset for multiple epochs, recording the model's confidence scores for each example at each epoch. We then compute the variance of these confidence scores across epochs for each example.

Examples with high confidence scores and low variability across epochs are considered easy, as the model consistently classifies them correctly. Conversely, examples with low confidence scores and high variability across epochs are considered hard, as the model struggles to classify them consistently. Using this method, we identify the hardest 5% and easiest 45% of examples in the dataset.

Upon inspecting the hardest examples in the dataset, it becomes evident that many of them are likely mislabeled. For instance, consider the following example:

Premise: Three men playing baseball and the one wearing a red helmet is sliding to the plate.
Hypothesis: Three men are playing basketball.
Gold Label: Entails

In this example, the premise clearly describes a scene involving baseball, while the hypothesis states that the men are playing basketball. This is a clear contradiction, as the two sports are different. However, the gold label is "Entails," which is incorrect given the context.

By examining such instances within the hardest examples, we can identify numerous instances where the gold labels are likely incorrect. This observation suggests that the dataset contains a significant number of mislabeled examples, which can negatively impact the performance of models trained on it. Addressing these mislabeled examples is crucial for improving model performance and ensuring a fair evaluation.

After filtering out these extreme examples, we preprocess the remaining dataset by tokenizing and encoding the premise and hypothesis texts.

We then train a model on this filtered dataset, using the same architecture and hyperparameters as a baseline model trained on the full dataset. Since we filtered out half of the dataset, we trained the model for twice the number of epochs to have the same number of training steps. After training, we evaluate the performance of both models on the original test dataset and compare their results.



## Identifying Mislabeled Examples

To identify mislabeled examples in the dataset, we employed GPT-3.5 to classify the hardest 5% of examples using a zero-shot approach. We crafted a prompt template for GPT-3.5, where the model was asked to assess whether an example was mislabeled or correctly labeled, provide its reasoning, and suggest the correct label if it deemed the example mislabeled. We utilized regular expressions to parse the output of GPT-3.5, extracting the classification, predicted label, and reasoning for each example.

Initially, we attempted to use Alpaca-7B for this task. However, the model did not consistently provide answers in the expected format, and in some cases, it was missing its prediction for the label. Consequently, we opted for GPT-3.5, which proved to be more reliable in providing the required information.


Graph showing the rate of classification mislabeled. Left are harder examples.


We utilized GPT-3.5 to classify the hardest 1000 examples from the filtered dataset. The majority of these examples were classified as mislabeled by the model. We observed that GPT-3.5 tended to overpredict mislabeled examples, resulting in a significant number of false positives. To further investigate the reliability of GPT-3.5's classification, we sampled 20 examples and manually evaluated them. Our analysis revealed that approximately 60% of these examples were incorrectly classified by GPT-3.5.

This result indicates that relying solely on GPT-3.5 for classifying mislabeled data may not be the most effective approach. The model's tendency to over-predict mislabeled examples and produce false positives suggests that additional strategies, such as few-shot learning, chain-of-thought prompting (Wei et al., 2022), or self-reflection (Madaan et al., 2023), might be necessary to improve its accuracy in identifying mislabeled instances.



## Results

We trained two models on the SNLI dataset: one on the filtered dataset, which excluded the hardest 5% and easiest 45% of examples, and the other on the original dataset. The filtered dataset model was trained for twice as many epochs, compensating for the reduced amount of data and ensuring that both models experienced the same number of training steps.

Our evaluation revealed that the model trained on the filtered dataset achieved an accuracy of 0.8719, while the model trained on the original dataset reached an accuracy of 0.8680. This result indicates that the filtered dataset model outperformed the original dataset model by a small margin.

The higher accuracy of the filtered dataset model suggests that addressing the mislabeled examples, as well as focusing on a more challenging subset of the data, can lead to improved model performance. The longer training time for the filtered dataset model also demonstrates the potential benefits of allowing models to train for additional epochs, particularly when working with smaller datasets. Overall, our results highlight the importance of refining the dataset quality and adjusting the training regimen to achieve better performance in natural language understanding tasks.



## Discussion

Our findings demonstrate the potential benefits of removing extreme examples from the training set in NLI tasks. By excluding the hardest 5% and easiest 45% of examples, we can create a more balanced and informative dataset that leads to better generalization and performance on the original dataset.

However, there are some limitations to this study. First, our method for identifying extreme examples is based on external metrics, which may not perfectly capture the true difficulty of each example. Further research is needed to explore more robust methods for identifying extreme examples in NLI tasks.

In our tests with GPT-3.5 as a classifier for mislabeled examples, we found that it tends to over-predict mislabeled instances, resulting in a high rate of false positives. This suggests that relying on GPT-3.5 alone for identifying mislabeled data might not be the most effective approach, and other strategies, such as few-shot learning, chain-of-thought prompting, or self-reflection, could be necessary to improve its accuracy in identifying mislabeled instances (Madaan et al., 2023; Wei et al., 2022).

Second, our results may not generalize to other NLI datasets or tasks, as the SNLI dataset has specific characteristics that may not be present in other benchmarks. Future work should explore the impact of removing extreme examples in other NLI tasks and datasets.


## References

Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., Smith, N. A., & Choi, Y. (2020). Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. In Proceedings of EMNLP. Retrieved from https://arxiv.org/abs/2009.10795

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Welleck, S., Majumder, B. P., Gupta, S., Yazdanbakhsh, A., & Clark, P. (2023). Self-Refine: Iterative Refinement with Self-Feedback. arXiv preprint arXiv:2303.17651. Retrieved from https://arxiv.org/abs/2303.17651 

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv preprint arXiv:2201.11903. Retrieved from https://arxiv.org/abs/2201.11903

Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. In Proceedings of ICLR. Retrieved from https://arxiv.org/abs/2003.10555 

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).

from PIL.Image import Image
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    default_data_collator,
)

from typing import List, Tuple, Union

from .post import post_processing_function
from .prep import get_extractive_features

class QABase:
    def answer(
            self,
            query: str,
            context: Union[Image, Union[str, List[str]]]
    ) -> Tuple[str, bool]: # str : answer, bool : answeralbe or not
        """Return answer when the question is answerable"""
        return NotImplemented

class ODQA(QABase):
    def __init__(
        self,
    ):
        super().__init__()
        model_name_or_path = "kiyoung2/koelectra-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        self.preprocess_function = get_extractive_features(self.tokenizer)
        self.post_process_function = post_processing_function
        
    @property
    def word_embeddings(self):
        """ Return word embeddings """
        prefix = self.model.base_model_prefix
        return getattr(self.model, prefix).embeddings.word_embeddings
                
    def answer(
        self,
        query: str,
        context: str,
    ) -> Tuple[str, bool]: # str : answer, bool : answeralbe or not
        """Return answer when th question is answerable"""
        # model_name_or_path = "./outputs/noanswer1/checkpoint-201792"
        # model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        device = torch.device("cpu")
        self.model.to(device)

        accelerator = Accelerator()

        predict_examples = Dataset.from_dict(
            {
                'answers' : [''],
                'context': [context],
                'id': ['user_input'],
                'question': [query],
            }
        )

        column_names = predict_examples.column_names

        predict_dataset = predict_examples.map(
            self.preprocess_function,
            batched=True,
            remove_columns=column_names,
        )

        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model, collate_fn=default_data_collator, batch_size=16
        )

        all_start_logits = []
        all_end_logits = []
        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    
        # concatenate the numpy array
        start_logits_concat = self.create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
        end_logits_concat = self.create_and_fill_np_array(all_end_logits, predict_dataset, max_len)
        
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        self.post_process_function
        
        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = self.post_process_function(predict_examples, predict_dataset, outputs_numpy)

        pred_answer = prediction[0][0]['prediction_text']
        answerable = True if pred_answer else False

        return (pred_answer, answerable)

    def create_and_fill_np_array(
        self,
        start_or_end_logits, 
        dataset, 
        max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

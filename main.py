import argparse
import dotenv
from datasets import Dataset, load_dataset
import json
from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor,
                          Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer)
from typing import Union, Optional, Any, Dict, List
from dataclasses import dataclass
import torch
import numpy as np
import jiwer

dotenv.load_dotenv()


def load_data_from_hf(dataset_name: str) -> Dataset:
    """Load the dataset from huggingface"""
    dataset = load_dataset("xibe_asr")
    return dataset["train"]


def make_vocab(dataset: Dataset):
    """Get all the character types that appear in the dataset."""
    vocab = set()
    for transcription in dataset["text"]:
        vocab.update(transcription)
    
    return vocab


def prepare_vocab(dataset: Dataset) -> str:
    """Prepare vocab for training."""
    vocab_file = "vocab.json"
    vocab = make_vocab(dataset)

    vocab_dict = {v: k for k, v in enumerate(vocab)}

    # Replace " " (whitespace) with a pipe "|"
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # Add special characters
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f)
    
    return vocab_file


def prepare_dataset(batch: dict) -> dict:
    """Prepare the dataset for the training.
    Add `input_values` and `labels` to the dataset.
    """
    audio = batch["audio"]

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0] # batched output is un-batched

    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = batch["text"]

    return batch


def compute_metrics(pred) -> Dict[str, float]:
        """Compute the evaluation score (CER)."""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids,
                                           group_tokens=False)

        cer = jiwer.cer(
            reference=label_str,
            hypothesis=pred_str
        )
        return {"cer": cer}


Feature = Dict[str, Union[List[int], torch.Tensor]]


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set, it will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self,
                 features: List[Feature]) -> Dict[str, torch.Tensor]:
        """Split inputs and labels since they have to be of different lengths
        and need different padding methods
        """
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        label_texts = [feature["labels"] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor(
            text=label_texts,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels

        return batch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="xibe_asr",
        help="The name of the dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/wav2vec2-xls-r-300m",
        help="The name of the model to use",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="wav2vec2-xls-r-300m-xibe",
        help="The name of the repository to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dataset = load_data_from_hf(args.dataset)

    vocab_file = prepare_vocab(dataset)

    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                tokenizer=tokenizer)
    
    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names)
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    training_args = TrainingArguments(
        output_dir=args.repo_name,
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=True,
    )

    dataset_dict = dataset.train_test_split(test_size=0.1)
    train = dataset_dict["train"]
    valid = dataset_dict["test"]

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor.feature_extractor,
    )
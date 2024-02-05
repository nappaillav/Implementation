# FineTuning LLM 

import numpy as np
import pandas as pd
import os, glob
import warnings
from tqdm import tqdm
import bitsandbytes as bnb

import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


############################# Part-1: DataSet Prep #############################

def generate_prompt(data):
    return f"""
            Analyse the sentiment of the following news headline and convey the sentiment as either 
            positive, negative or neutral. The new is "{data[0]}" = "{data[1]}""".strip()

def generate_prompt_test(data):
    return f"""
            Analyse the sentiment of the following news headline and convey the sentiment as either 
            positive, negative or neutral. The new is "{data[0]}" =""".strip()

# convert your dataset to prompt type input

#convert pandas dataframe to Huggingface dataset
train_data = Dataset.from_pandas(X_train_df)
eval_data = Dataset.from_pandas(X_eval_df)

############################# Part-2: Model #############################
model_name = "llama-2" # TODO 

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Load model weights in 4-bit format
    bnb_4bit_quant_type="nf4", #4-bit NormalFloat(NF4), is a new data type that is information theoretically optimal for normally distributed weights.
    bnb_4bit_compute_dtype=compute_dtype, #Use float16 data type for computations.
    bnb_4bit_use_double_quant=False, #Do not use double quantization (reduces the average memory footprint by quantizing also the quantization 
                                     #constants and saves an additional 0.4 bits per parameter)
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

model.config.use_cache = False
model.config.pretraining_tp = 1 # a value different than 1 will activate the more accurate but slower computation of the linear layers, 
                                # which should better match the original logits.

############################# Part-3: Tokenizer #############################

# Use the tokenizer same as the base model
tokenizer = AutoTokenizer.from_pretrained(model_name
                                          trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token()
tokenizer.padding_side = "right"

############################# Part-4: Training #############################

# PEFT configuration
peft_config = LoraConfig(
    lora_alpha=16, # learning rate for the LoRA update matrices
    lora_dropout=0.1, # dropout probability for the LoRA update matrices 
    r=64,#rank of the LoRA update matrices, lower rank results in smaller update matrices with fewer trainable parameters
    bias="none", #type of bias to use. The possible values are none, additive, and learned.
    task_type="CAUSAL_LM", #type of task that the model is being trained for. The possible values are CAUSAL_LM and MASKED_LM
)

training_arguments = TrainingArguments(
    output_dir="logs", #The directory where the training logs and checkpoints will be saved.
    num_train_epochs=3, #The number of epochs to train the model for
    per_device_train_batch_size=1, #The number of samples in each batch on each device.
    gradient_accumulation_steps=8, # 4 The number of batches to accumulate gradients before updating the model parameters.
    optim="paged_adamw_32bit", #optimizer to use for training the model
    save_steps=0, #number of steps after which to save a checkpoint
    logging_steps=25, #number of steps after which to log the training metrics
    learning_rate=2e-4, #learning rate for the optimizer
    weight_decay=0.001, #weight decay parameter for the optimizer
    fp16=True, #Whether to use 16-bit floating-point precision.
    bf16=False, #Whether to use BFloat16 precision.
    max_grad_norm=0.3, #The maximum gradient norm
    max_steps=-1, #The maximum number of steps to train the model for
    warmup_ratio=0.03,#The proportion of the training steps to use for warming up the learning rate.
    group_by_length=True, #Whether to group the training samples by length.
    lr_scheduler_type="cosine", # type of learning rate scheduler to use
    report_to="tensorboard", # The tools to report the training metrics to
    evaluation_strategy="epoch" #strategy for evaluating the model during training
)

trainer = SFTTrainer(
    model=model, #model to be trained
    train_dataset=train_data, # Hugging face Dataser
    eval_dataset=eval_data, # Hugging face Dataser
    peft_config=peft_config, 
    dataset_text_field="text", # name of the text field in the dataset.
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Whether to pack the training samples
    max_seq_length=1024,
)

############################# Part-5: Evaluation #############################

def predict(test, model, tokenizer):
    # Use the model to predict
    y_pred = []

    # use pipeline
    pipe = pipeline(task="text-generation", 
                    model=model,
                    tokenizer=tokenizer,
                    max_new_token = 1, # number of token to generate[positive, negative, neutral]
                    temperature=0.0 # I'm not sure if this is in need to do with 
                    ) 

    for i in tqdm(range(len(test))):
        prompt = # TODO
        result = pipe(prompt)

        answer = result[0]["generated_text"].split("=")[-1]
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("None")
    return y_pred 

def evaluate(y_true, y_pred):
    labels = ['positive', 'neutral', 'negative']
    
    #Note: sometimes there will be no predictions, so 'none' i
    mapping = {'positive': 2, 'neutral': 1, 'none':1, 'negative': 0}
    
    def map_func(x):
        return mapping.get(x, 1)
    
    #the predicted label and the ones in test set contain labels as 'positive', 'neutral', 'none' or 'negative'
    #map the labels to 0,1 and 2  
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

############################# Part-6: Main Function #############################
from datetime import datetime
import pytz

tz = pytz.timezone("America/New_York")
start_time = datetime.now(tz)
print (f'\nTraining started at {start_time}')

# Train model
trainer.train()

end_time = datetime.now(tz)
duration = end_time - start_time
print (f'Training completed at {end_time}')
print(f'Training duration was {duration}')

# Save trained model to 'trained_model' directory
trainer.model.save_pretrained("trained-model")

y_pred = predict(X_test_df, model, tokenizer)
evaluate(y_true, y_pred)

evaluation = pd.DataFrame({'text': X_test_df["text"], 
                           'y_true':y_true, 
                           'y_pred': y_pred},
                         )
evaluation.to_csv("test_predictions.csv", index=False)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModel
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import json, os
import random
import numpy as np
        

def treinar_com_artigos(dados, artigos):
    auto_tokenizer = None
    model = None
    
    if not os.path.exists(os.path.join("models", "fine-tune-petros")):
        auto_tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')
        model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2')

    else:
        auto_tokenizer = AutoTokenizer.from_pretrained(os.path.join("models", 'fine-tune-petros'))
        model = AutoModel.from_pretrained(os.path.join("models", 'fine-tune-petros'))

    def tokenize_function(examples):
        return auto_tokenizer(examples.text, padding="max_length", truncation=True, max_lenght=128)

    treino = np.array(artigos, dtype=object)[dados['train']].tolist()
    avaliacao = np.array(artigos, dtype=object)[dados['eval']].tolist()
    
    treinamento_dataset = Dataset.from_list(list(map(lambda x:x.text, treino)))
    avaliacao_dataset = Dataset.from_list(list(map(lambda x:x.text, avaliacao)))

    tokenized_treinamento_dataset = treinamento_dataset.map(tokenize_function, batched=True)
    tokenized_avaliacao_dataset = avaliacao_dataset.map(tokenize_function, batched=True)

    args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epochs",
        learning_rate=2e-5,
        per_device_eval_batch_size = 8,
        per_device_train_batch_size = 8,
        num_train_epochs=10,
        weight_decay=0.01           

    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_treinamento_dataset,
        eval_dataset=tokenized_avaliacao_dataset,
        tokenizer = auto_tokenizer
    )

    print("Training...")

    trainer.train()
    model.save_pretrained(os.path.join("models", "fine-tune-petros"))   
    auto_tokenizer.save_pretrained(os.path.join("models", 'fine-tuned-petros'))

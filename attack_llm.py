import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid tensorflow warnings
from transformers import AutoModel


import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List
import torch
import json
from transformers import (
    AutoModel,
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BatchEncoding,
)
import re
import random
import numpy as np
import torch
from nltk import wsd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from spacy.cli import download
from spacy import load
import warnings
from typing import Union, List, Tuple
from datasets import load_dataset, Dataset
import evaluate
import nsga2_llm

from DG_dataset import DGDataset
from typing import Optional

DATA2NAME = {
    "blended_skill_talk": "BST",
    "conv_ai_2": "ConvAI2",
    "empathetic_dialogues": "ED",
    "AlekseyKorshuk/persona-chat": "PC",
}


from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
softmax = nn.Softmax(dim=1)
bce_loss = nn.BCELoss()

# Initialize logging and downloads
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

class ModelConfig:
    def __init__(self, do_sample=True, num_return_sequences=1, temp=1.0, gen_max_len=512, top_p=0.9):
        self.do_sample = do_sample  # Whether to sample or use greedy decoding
        self.num_return_sequences = num_return_sequences  # Number of sequences to generate
        self.temp = temp  # Temperature for generation (controls randomness)
        self.gen_max_len = gen_max_len  # Maximum number of tokens to generate
        self.top_p = top_p  # Nucleus sampling (select tokens with cumulative probability top_p)






class SentenceEncoder:
    def __init__(self, model_name='paraphrase-distilroberta-base-v1', device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        # Ensure sentences are on the correct device
        return self.model.encode(sentences, convert_to_tensor=True,
                                 show_progress_bar = False,
                                 device=self.device)

    def get_sim(self, sentence1, sentence2):
        embeddings = self.encode([sentence1, sentence2])
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cos_sim.item()

    def find_best_match(self, original_sentence, candidate_sentences, find_min=False):
        original_embedding = self.encode(original_sentence)
        candidate_embeddings = self.encode(candidate_sentences)
        best_candidate = None
        best_index = None
        best_sim = float('inf') if find_min else float('-inf')

        for i, candidate_embedding in enumerate(candidate_embeddings):
            sim = util.pytorch_cos_sim(original_embedding, candidate_embedding).item()
            if find_min:
                if sim < best_sim:
                    best_sim = sim
                    best_candidate = candidate_sentences[i]
                    best_index = i
            else:
                if sim > best_sim:
                    best_sim = sim
                    best_candidate = candidate_sentences[i]
                    best_index = i

        return best_candidate, best_index, best_sim

def chatbot_gen(tokenizer, model, prompts: str, stop: Optional[list[str]] = None, system_prompts: str = None, device='cuda', config=None):
    """Run the LLM on the given prompt and input."""
    # Ensure PAD token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.eos_token_id = tokenizer.pad_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id
    
    # Prepare the system and user messages
    if system_prompts is not None:
        messages = [
            {"role": "system", "content": system_prompts},
            {"role": "user", "content": prompts},
        ]
    else:
        messages = [
            {"role": "user", "content": prompts},
        ]

    # Apply chat template and prepare input ids
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # Create batched prompts as BatchEncoding
    batched_prompts = BatchEncoding({"input_ids": input_ids}).to(device)

    input_ids_len: int = batched_prompts["input_ids"].shape[1]

    with torch.inference_mode():
        terminators = tokenizer.eos_token_id

        # Generate tokens
        tokens = model.generate(
            **batched_prompts,
            do_sample=config.do_sample,
            num_return_sequences=config.num_return_sequences,
            temperature=config.temp,
            max_new_tokens=config.gen_max_len,
            top_p=config.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
        )

        # Decode the tokens into text
        texts: list[str] = tokenizer.batch_decode(
            tokens[:, input_ids_len:, ...], skip_special_tokens=False
        )

    return texts[0]

class DGAttackEval(DGDataset):
    def __init__(self, 
        args: argparse.Namespace = None, 
        tokenizer: AutoTokenizer = None, 
        model: AutoModelForSeq2SeqLM = None, 
        #attacker: WordAttacker = None, 
        device: torch.device('cpu') = None, 
        task: str = 'seq2seq', 
        bleu: evaluate.load("bleu") = None, 
        rouge: evaluate.load("rouge") = None,
        meteor: evaluate.load("meteor") = None,
        ):
            
            super(DGAttackEval, self).__init__(
                dataset=args.dataset,
                task=task,
                tokenizer=tokenizer,
                max_source_length=args.max_len,
                max_target_length=args.max_len,
                padding=None,
                ignore_pad_token_for_loss=True,
                preprocessing_num_workers=None,
                overwrite_cache=True,
            )
            self.args = args
            self.model = model
            self.device = args.device
            self.task = task
            self.num_beams = args.num_beams
            self.num_beam_groups = args.num_beam_groups
            self.max_num_samples = args.max_num_samples

            self.bleu = bleu
            self.rouge = rouge
            self.meteor = meteor

            self.sentencoder = SentenceEncoder(device=args.device)
            self.config = ModelConfig(do_sample=False, num_return_sequences=1, temp=0.7, gen_max_len=1024, top_p=0.9)

            self.ori_lens, self.adv_lens = [], []
            self.ori_bleus, self.adv_bleus = [], []
            self.ori_rouges, self.adv_rouges = [], []
            self.ori_meteors, self.adv_meteors = [], []
            self.ori_time, self.adv_time = [], []
            self.cos_sims = []
            self.att_success = 0
            self.total_pairs = 0
            self.sp_token = '<SEP>'

            # self.record = []
            #att_method = args.attack_strategy
            out_dir = args.out_dir
            model_n = args.model_name_or_path.split("/")[-1]
            dataset_n = DATA2NAME.get(args.dataset, args.dataset.split("/")[-1])
            #combined = "combined" if args.use_combined_loss and att_method == 'structure' else "single"
            #max_per = args.tas
            #fitness = args.fitness if att_method == 'structure' else 'performance'
            select_beams = args.select_beams
            max_num_samples = args.max_num_samples
            num_gen = args.num_gen
            num_ind = args.num_ind
            att_method = "NSGA-II_" +  str(num_gen)  + "gen_"  + str(num_ind) + "ind_" 
            file_path = f"{out_dir}/{att_method}_{select_beams}_{model_n}_{dataset_n}_{max_num_samples}.txt"
            self.write_file_path = file_path


    def log_and_save(self, display: str):
        print(display)
        with open(self.write_file_path, 'a') as f:
            f.write(display + "\n")
        #self.write_file.write(display + "\n")   
    
    def get_prediction(self, text: str):
        if self.task == 'seq2seq':
            effective_text = text
        else:
            effective_text = text + self.tokenizer.eos_token

        inputs = self.tokenizer(
            effective_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length-1,
        )
        input_ids = inputs.input_ids.to(args.device)
        self.model = self.model.to(args.device)
        t1 = time.time()
        with torch.no_grad():
            outputs = dialogue(
                self.model,
                input_ids,
                early_stopping=False,
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups,
                use_cache=True,
                max_length=self.max_target_length,
            )
        if self.task == 'seq2seq':
            output = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        else:
            output = self.tokenizer.batch_decode(
                outputs['sequences'][:, input_ids.shape[-1]:],
                skip_special_tokens=True,
            )[0]
        t2 = time.time()
        return output.strip(), t2 - t1


    def eval_metrics(self, output: str, guided_messages: List[str]):
        if not output:
            return

        bleu_res = self.bleu.compute(
            predictions=[output],
            references=[guided_messages],
            smooth=True,
        )
        rouge_res = self.rouge.compute(
            predictions=[output],
            references=[guided_messages],
        )
        meteor_res = self.meteor.compute(
            predictions=[output],
            references=[guided_messages],
        )
        pred_len = bleu_res['translation_length']
        return bleu_res, rouge_res, meteor_res, pred_len

    def prepare_context(self, instance: dict):
        dataset = args.dataset  # Use args.dataset here

        if dataset == 'blended_skill_talk':
            persona = f"<PS>{instance['personas'][1]}"
            num_entries = len(instance["free_messages"])
            if self.task == 'seq2seq':
                if instance['context'] == "wizard_of_wikipedia":
                    additional_context_pieces = f"<CTX>{instance['additional_context']}."
                else:
                    additional_context_pieces = ""
                context = additional_context_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = [sent for sent in instance["previous_utterance"] if sent != '']

        elif dataset == 'conv_ai_2':
            num_entries = len(instance['dialog']) // 2
            persona = f"<PS>{' '.join([''.join(x) for x in instance['user_profile']])}"
            if self.task == 'seq2seq':
                context = ''
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []

        elif dataset == 'empathetic_dialogues':
            num_entries = len(instance['dialog']) // 2
            persona = f"<PS>{instance['prompt']}"
            if self.task == 'seq2seq':
                additional_context_pieces = f"<CTX>{instance['context']}."
                context = additional_context_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []

        elif dataset == 'AlekseyKorshuk/persona-chat':
            num_entries = len(instance['utterances']) // 2
            persona = f"<PS>{' '.join(instance['personality'])}"
            if self.task == 'seq2seq':
                context = ''
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []

        else:
            raise ValueError("Dataset not supported.")
        
        return num_entries, persona, context, prev_utt_pc


    def prepare_entry(self, instance: dict, entry_idx: int, context: str, prev_utt_pc: list):
        dataset = args.dataset  # Use args.dataset here

        if dataset == 'blended_skill_talk':
            free_message = instance['free_messages'][entry_idx]
            guided_message = instance['guided_messages'][entry_idx]
            references = [values[entry_idx] for key, values in instance['suggestions'].items()]

        elif dataset == 'conv_ai_2':
            free_message = instance['dialog'][entry_idx * 2]['text']
            if entry_idx * 2 + 1 >= len(instance['dialog']):
                guided_message = None
            else:
                guided_message = instance['dialog'][entry_idx * 2 + 1]['text']
            references = []

        elif dataset == 'empathetic_dialogues':
            free_message = instance['dialog'][entry_idx * 2]['text']
            if entry_idx * 2 + 1 >= len(instance['dialog']):
                guided_message = None
            else:
                guided_message = instance['dialog'][entry_idx * 2 + 1]['text']
            references = []

        elif dataset == 'AlekseyKorshuk/persona-chat':
            free_message = instance['utterances'][entry_idx * 2]['history'][-1]
            if entry_idx * 2 + 1 >= len(instance['utterances']):
                guided_message = None
            else:
                guided_message = instance['utterances'][entry_idx * 2 + 1]['history'][-1]
            references = instance['utterances'][entry_idx * 2]['candidates']

        else:
            raise ValueError("Dataset not supported.")

        if not prev_utt_pc:
            original_context = context
        else:
            original_context = context + self.sp_token + self.sp_token.join(prev_utt_pc)

        references.append(guided_message)
        return free_message, guided_message, original_context, references


    def generation_step(self, instance: dict):
            # Set up
            num_entries, persona, context, prev_utt_pc = self.prepare_context(instance)
            for entry_idx in range(num_entries):
                free_message, guided_message, original_context, references = self.prepare_entry(
                        instance, entry_idx, context, prev_utt_pc
                    )
                if guided_message is None:
                    continue

                prev_utt_pc += [free_message, guided_message]

                dialogue_his = persona + original_context

                self.log_and_save("\nDialogue history: {}".format(dialogue_his))
                self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(free_message, references[-1]))

                # Determine the model prompt based on model name
                if 'gemma' in args.model_name_or_path.lower():
                    # No system prompts, embed the instruction into the user-facing prompt
                    instruction = "You are roleplaying a user based on the provided persona, chat history, and the current utterance from the other user. The persona is denoted by the <PS> token, and previous dialogue turns are separated by the <SEP> token. Please respond in character as a human and do not reference the fact that you are a model or AI. Generate a concise, relevant response in no more than 20 words or one sentence, ensuring the response aligns with the user's persona and the context."
                    system_prompts = None
            
                elif 'llama' in args.model_name_or_path.lower():
                    system_prompts = """
                    You are roleplaying a user based on the provided persona, chat history and the current utterance from the other user. Generate a concise and relevant response briefly in no more than one sentence which appropriately continues the conversation. Ensure the response aligns with the user's persona.
                    """
                else:
                    system_prompts = """
                    You are roleplaying a user based on the provided persona and chat history. Generate a concise and relevant response briefly in no more than one sentence which appropriately continues the conversation. Ensure the response aligns with the user's persona.
                    """

                # Prepare the prompt for chatbot_gen
                if 'gemma' in args.model_name_or_path.lower():
                    prompts = f"""
                    ### Persona:
                    {persona}

                    ### Context:
                    {original_context}

                    ### Current Utterance:
                    {free_message}

                    ### Instruction:
                    {instruction}
                    """
                else:
                    prompts = f"""
                    ### Persona:
                    {persona}

                    ### Context:
                    {original_context}

                    ### Current Utterance:
                    {free_message}
                    """

                # Generate the original response using chatbot_gen
                t1 = time.time()
                output = chatbot_gen(self.tokenizer, self.model, prompts, system_prompts=system_prompts,device=self.device, config=self.config)
                t2 = time.time()
                time_gap = t2 - t1

                self.log_and_save("G--{}".format(output))
                ori_length = len(output.split())
                bleu_res, rouge_res, meteor_res, pred_len = self.eval_metrics(output, references)
                self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                    ori_length, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
                ))

                self.ori_lens.append(ori_length)
                self.ori_bleus.append(bleu_res['bleu'])
                self.ori_rouges.append(rouge_res['rougeL'])
                self.ori_meteors.append(meteor_res['meteor'])
                self.ori_time.append(time_gap)

                # Attack
        
                self.model = self.model.to(args.device)
                if args.crossover_flag == 1:
                    print("BAT DAU NSGA-II VOI CROSSOVER")
                else:
                    print("BAT DAU NSGA-II")
                
                problem = nsga2_llm.Problem(self.model, self.tokenizer,original_context,persona,
                free_message, guided_message, args.device,args.max_len,self.task,args.acc_metric,self.bleu,self.rouge,self.meteor,output,args.model_name_or_path)

                evolution = nsga2_llm.Evolution(args.crossover_flag, self.write_file_path, problem, num_of_generations=args.num_gen, num_of_individuals=args.num_ind, num_of_tour_particips=2,
                        tournament_prob=0.9, crossover_param=2, mutation_param=5 )

                resulting_front = evolution.evolve()
                result = []
                for individual in resulting_front:
                    result.append((individual.sentence,individual.accuracy, individual.length))
                    #print(individual.sentence, individual.cls_loss, individual.eos_loss)
                data_with_fitness = [(sentence, accuracy, length, length / accuracy) for sentence, accuracy, length in result]

                # Sort based on the fitness score (fourth tuple element), in descending order
                sorted_data = sorted(data_with_fitness, key=lambda x: x[3], reverse=True)
                #sorted_data = sorted(result, key=lambda x: x[1])
                new_free_message = sorted_data[0][0]

                
                if 'gemma' in args.model_name_or_path.lower():
                    adv_prompts = f"""
                    ### Persona:
                    {persona}

                    ### Context:
                    {original_context}

                    ### Current Utterance:
                    {new_free_message}

                    ### Instruction:
                    {instruction}
                    """
                else:
                    adv_prompts = f"""
                    ### Persona:
                    {persona}

                    ### Context:
                    {original_context}

                    ### Current Utterance:
                    {new_free_message}
                    """

                # Generate adversarial response
                t1 = time.time()
                adv_output = chatbot_gen(self.tokenizer, self.model, adv_prompts, system_prompts=system_prompts,device=self.device, config=self.config)
                t2 = time.time()
                time_gap = t2 - t1

                cos_sim = self.sentencoder.get_sim(new_free_message, free_message)

                self.log_and_save(f"U'--{new_free_message} (cosine: {cos_sim})")
                self.log_and_save(f"G'--{adv_output}")
                adv_length = len(adv_output.split())

                # Evaluate adversarial metrics
                adv_bleu_res, adv_rouge_res, adv_meteor_res, adv_pred_len = self.eval_metrics(adv_output, references)
                self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                    adv_length, time_gap, adv_bleu_res['bleu'], adv_rouge_res['rougeL'], adv_meteor_res['meteor'],
                ))

                # Check if the adversarial attack was successful
                success = (
                    (bleu_res['bleu'] > adv_bleu_res['bleu']) or
                    (rouge_res['rougeL'] > adv_rouge_res['rougeL']) or
                    (meteor_res['meteor'] > adv_meteor_res['meteor'])
                ) and float(cos_sim) > 0.7

                if success:
                    self.att_success += 1
                    self.log_and_save("Attack success!")
                else:
                    self.log_and_save("Attack failed!")

                # Append adversarial results
                self.adv_lens.append(adv_pred_len)
                self.adv_bleus.append(adv_bleu_res['bleu'])
                self.adv_rouges.append(adv_rouge_res['rougeL'])
                self.adv_meteors.append(adv_meteor_res['meteor'])
                self.adv_time.append(time_gap)
                self.cos_sims.append(cos_sim)
                self.total_pairs += 1



    def generation(self, test_dataset: Dataset):
        if self.dataset == "empathetic_dialogues":
            test_dataset = self.group_ED(test_dataset)
            
        
        ids = random.sample(range(len(test_dataset)), self.max_num_samples)
        test_dataset = test_dataset.select(ids)
        print("Test dataset: ", test_dataset)
            # print("CHECKPOINT")
            # print(self.task)
        for i, instance in tqdm(enumerate(test_dataset)):
            self.generation_step(instance)

        Ori_len = np.mean(self.ori_lens)
        Adv_len = np.mean(self.adv_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Adv_bleu = np.mean(self.adv_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Adv_rouge = np.mean(self.adv_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Adv_meteor = np.mean(self.adv_meteors)
        self.cos_sims = [float(cos_sim) for cos_sim in self.cos_sims]
        Cos_sims = np.mean(self.cos_sims)
        Ori_t = np.mean(self.ori_time)
        Adv_t = np.mean(self.adv_time)

        # Summarize eval results
        self.log_and_save("\nOriginal output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
        ))
        self.log_and_save("Perturbed [cosine: {:.3f}] output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Cos_sims, Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
        ))
        self.log_and_save("Attack success rate: {:.2f}%".format(100*self.att_success/self.total_pairs))

        


    def generation_step_trans(self, instance: dict, adversarial_users: list, cum_idx: int):
    # Set up
        num_entries, persona, context, prev_utt_pc = self.prepare_context(instance)
        for entry_idx in range(num_entries):
            free_message, guided_message, original_context, references = self.prepare_entry(
                instance, entry_idx, context, prev_utt_pc
            )
            if guided_message is None:
                continue

            prev_utt_pc += [free_message, guided_message]

            dialogue_his = persona + original_context

            self.log_and_save("\nDialogue history: {}".format(dialogue_his))
            self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(free_message, references[-1]))

            # Determine the model prompt based on model name
            if 'gemma' in args.model_name_or_path.lower():
                # No system prompts, embed the instruction into the user-facing prompt
                instruction = "You are roleplaying a user based on the provided persona, chat history, and the current utterance from the other user. The persona is denoted by the <PS> token, and previous dialogue turns are separated by the <SEP> token. Please respond in character as a human and do not reference the fact that you are a model or AI. Generate a concise, relevant response in no more than 20 words or one sentence, ensuring the response aligns with the user's persona and the context."
                system_prompts = None
        
            elif 'llama' in args.model_name_or_path.lower():
                system_prompts = """
                You are roleplaying a user based on the provided persona, chat history and the current utterance from the other user. Generate a concise and relevant response briefly in no more than one sentence which appropriately continues the conversation. Ensure the response aligns with the user's persona.
                """
            else:
                system_prompts = """
                You are roleplaying a user based on the provided persona and chat history. Generate a concise and relevant response briefly in no more than one sentence which appropriately continues the conversation. Ensure the response aligns with the user's persona.
                """

            # Prepare the prompt for chatbot_gen
            if 'gemma' in args.model_name_or_path.lower():
                prompts = f"""
                ### Persona:
                {persona}

                ### Context:
                {original_context}

                ### Current Utterance:
                {free_message}

                ### Instruction:
                {instruction}
                """
            else:
                prompts = f"""
                ### Persona:
                {persona}

                ### Context:
                {original_context}

                ### Current Utterance:
                {free_message}
                """

            # Generate the original response using chatbot_gen
            t1 = time.time()
            output = chatbot_gen(self.tokenizer, self.model, prompts, system_prompts=system_prompts,device=self.device, config=self.config)
            t2 = time.time()
            time_gap = t2 - t1

            self.log_and_save("G--{}".format(output))
            ori_length = len(output.split())
            bleu_res, rouge_res, meteor_res, pred_len = self.eval_metrics(output, references)
            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                ori_length, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
            ))

            self.ori_lens.append(ori_length)
            self.ori_bleus.append(bleu_res['bleu'])
            self.ori_rouges.append(rouge_res['rougeL'])
            self.ori_meteors.append(meteor_res['meteor'])
            self.ori_time.append(time_gap)

            # Process the corresponding adversarial sentence for this turn using cum_idx
            print("CHECK CUMULATIVE ID", cum_idx)

            if cum_idx < len(adversarial_users):
                adv_sentence = adversarial_users[cum_idx][0]  # Adversarial sentence (U')
                cos_sim = adversarial_users[cum_idx][1]       # Cosine similarity
                adv_response = adversarial_users[cum_idx][2]  # Adversarial response (G')

                # Prepare the adversarial prompt
                
                if 'gemma' in args.model_name_or_path.lower():
                    adv_prompts = f"""
                    ### Persona:
                    {persona}

                    ### Context:
                    {original_context}

                    ### Current Utterance:
                    {adv_sentence}

                    ### Instruction:
                    {instruction}
                    """
                else:
                    adv_prompts = f"""
                    ### Persona:
                    {persona}

                    ### Context:
                    {original_context}

                    ### Current Utterance:
                    {adv_sentence}
                    """

                # Generate adversarial response
                t1 = time.time()
                adv_output = chatbot_gen(self.tokenizer, self.model, adv_prompts, system_prompts=system_prompts,device=self.device, config=self.config)
                t2 = time.time()
                time_gap = t2 - t1

                self.log_and_save(f"U'--{adv_sentence} (cosine: {cos_sim})")
                self.log_and_save(f"G'--{adv_output}")
                adv_length = len(adv_output.split())

                # Evaluate adversarial metrics
                adv_bleu_res, adv_rouge_res, adv_meteor_res, adv_pred_len = self.eval_metrics(adv_output, references)
                self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                    adv_length, time_gap, adv_bleu_res['bleu'], adv_rouge_res['rougeL'], adv_meteor_res['meteor'],
                ))

                # Check if the adversarial attack was successful
                success = (
                    (bleu_res['bleu'] > adv_bleu_res['bleu']) or
                    (rouge_res['rougeL'] > adv_rouge_res['rougeL']) or
                    (meteor_res['meteor'] > adv_meteor_res['meteor'])
                ) and float(cos_sim) > 0.7

                if success:
                    self.att_success += 1
                    self.log_and_save("Attack success!")
                else:
                    self.log_and_save("Attack failed!")

                # Append adversarial results
                self.adv_lens.append(adv_length)
                self.adv_bleus.append(adv_bleu_res['bleu'])
                self.adv_rouges.append(adv_rouge_res['rougeL'])
                self.adv_meteors.append(adv_meteor_res['meteor'])
                self.adv_time.append(time_gap)
                self.cos_sims.append(cos_sim)
                self.total_pairs += 1
                cum_idx = cum_idx + 1
            
            
        return cum_idx  # Increment the cumulative index

    def generation_trans(self, test_dataset: Dataset, adversarial_log: str):
        if self.dataset == "empathetic_dialogues":
            test_dataset = self.group_ED(test_dataset)

        ids = random.sample(range(len(test_dataset)), self.max_num_samples)
        test_dataset = test_dataset.select(ids)
        print("Test dataset: ", test_dataset)

        with open(adversarial_log, 'r') as log_file:
            logs = log_file.read()

        # Extract adversarial sentences and corresponding dialogue history from the log
        history_pattern = r"Dialogue history: (.*?)\nU--(.*?)\n\(Ref: \[(.*?)\]"
        adversarial_user_pattern = r"U'--(.*?) \(cosine: ([0-9.]+)\)\nG'--(.*?)\n"

        # Extract histories and adversarial users from the log
        adversarial_users = re.findall(adversarial_user_pattern, logs, re.DOTALL)

        # Initialize cumulative index for adversarial sentences
        cum_idx = 0  # Track dialogue turns across all samples

        # Loop through test dataset and adversarial users
        for i, instance in enumerate(tqdm(test_dataset)):
            cum_idx = self.generation_step_trans(instance, adversarial_users, cum_idx)

        # Summarize evaluation results (as before)
        Ori_len = np.mean(self.ori_lens)
        Adv_len = np.mean(self.adv_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Adv_bleu = np.mean(self.adv_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Adv_rouge = np.mean(self.adv_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Adv_meteor = np.mean(self.adv_meteors)
        self.cos_sims = [float(cos_sim) for cos_sim in self.cos_sims]
        Cos_sims = np.mean(self.cos_sims)
        Ori_t = np.mean(self.ori_time)
        Adv_t = np.mean(self.adv_time)

        # Summarize eval results
        self.log_and_save("\nOriginal output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
        ))
        self.log_and_save("Perturbed [cosine: {:.3f}] output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Cos_sims, Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
        ))
        self.log_and_save("Transferable attack success rate: {:.2f}%".format(100 * self.att_success / self.total_pairs))

    
def main(args: argparse.Namespace):
        random.seed(args.seed)
        model_name_or_path = args.model_name_or_path
        dataset = args.dataset
        max_len = args.max_len
        max_per = args.max_per
        num_beams = args.num_beams
        select_beams = args.select_beams
        num_beam_groups = args.num_beam_groups
        out_dir = args.out_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
        config = AutoConfig.from_pretrained(model_name_or_path,token=args.access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,token=args.access_token)
        if 'gemma' in model_name_or_path.lower() or 'llama' in model_name_or_path.lower():
            task = 'seq2seq'
            if args.transfer==1:
                model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=torch.float16,  
                device_map=args.device,  
                token=args.access_token,
                )   
            else:
                model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=torch.float16,  
                device_map=args.device,  
                token=args.access_token,
                )
            if 'results' not in model_name_or_path.lower():
                tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                tokenizer.add_special_tokens({'mask_token': '<MASK>'})
                model.resize_token_embeddings(len(tokenizer))
        else:
            task = 'seq2seq'
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=torch.float16,  
                device_map='auto',
                token=args.access_token,  
            )


        # Load dataset
        all_datasets = load_dataset(dataset)
        if dataset == "conv_ai_2":
            test_dataset = all_datasets['train']
        elif dataset == "AlekseyKorshuk/persona-chat":
            test_dataset = all_datasets['validation']
        else:
            test_dataset = all_datasets['test']

        # Load evaluation metrics
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")

        # Define DG attack
        dg = DGAttackEval(
            args=args,
            tokenizer=tokenizer,
            model=model,
            device= args.device,
            task=task,
            bleu=bleu,
            rouge=rouge,
            meteor=meteor,
        )

        adv_log = args.transfer_log_dir

        if args.transfer == 1:
            dg.generation_trans(test_dataset,adv_log)
        else:
            dg.generation(test_dataset)

    
if __name__ == "__main__":
    import ssl
    import argparse
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    import nltk
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_samples", type=int, default=5, help="Number of samples to attack")
    parser.add_argument("--max_per", type=int, default=5, help="Number of perturbation iterations per sample")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum length of generated sequence")
    parser.add_argument("--select_beams", type=int, default=2, help="Number of sentence beams to keep for each attack iteration")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for decoding in LLMs")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups for decoding in LLMs")
    parser.add_argument("--acc_metric", type=str, default="combined",
                        choices=["bleu", "rouge", "meteor", "combined"],
                        help="Fitness function for selecting the best candidate")
    parser.add_argument("--model_name_or_path", "-m", type=str, default="result/Bart", help="Path to model")
    parser.add_argument("--dataset", "-d", type=str, default="blended_skill_talk",
                        choices=["blended_skill_talk", "conv_ai_2", "empathetic_dialogues", "AlekseyKorshuk/persona-chat"],
                        help="Dataset to attack")
    parser.add_argument("--out_dir", type=str,
                        default="./results/logging",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--objective", type=str, default="cls", choices=["cls", "eos"], help="Objective")
    parser.add_argument("--num_ind", type=int, default=100, help="Number of Individuals")
    parser.add_argument("--num_gen", type=int, default=50, help="Number of Individuals")
    parser.add_argument("--crossover_flag", type=int, default=0, help="Whether to use Crossover or not")
    parser.add_argument("--device", type=str,default="cuda",help="Determine which GPU to use")
    parser.add_argument("--transfer", type=int, default=0, help="Whether to use transferability or not")
    parser.add_argument("--name", type=str,default="process1",help="Determine which GPU to use")
    parser.add_argument("--transfer_log_dir", type=str,
                        default="./results/logging",
                        help="Directory of transferability log file (txt)")
    parser.add_argument("--access_token", type=str, default="", help="Hugging Face access token for loading models")

    args = parser.parse_args()
    main(args)

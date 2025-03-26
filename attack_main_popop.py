import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid tensorflow warnings
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List
import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
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
import re
from DialogueAPI import dialogue
from DG_dataset import DGDataset
from utils import GrammarChecker

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
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def identify_salient_words(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Get POS tags
    pos_tags = nltk.pos_tag(tokens)
    #print(pos_tags)
    # Define POS tags of interest (e.g., nouns, verbs, adjectives)
    salient_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP','JJ', 'JJR', 'JJS'}

    # Identify salient words based on POS tags
    salient_words = [word for word, tag in pos_tags if tag in salient_pos_tags and word.isalnum() and len(word) > 1  and word.lower() not in immutable_words]
    return salient_words


def identify_salient_words_with_idx(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Get POS tags
    pos_tags = nltk.pos_tag(tokens)
    # print(pos_tags)
    # Define POS tags of interest (e.g., nouns, verbs, adjectives)
    salient_pos_tags = {
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBZ",
        "VBP",
        "JJ",
        "JJR",
        "JJS",
    }

    immutable_words = {'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves'}

    # Identify salient words based on POS tags

    words_count = {}
    salient_words = []

    for word, tag in pos_tags:
        if word not in words_count:
            words_count[word] = 1
        else:
            words_count[word] += 1

        if not (
            tag in salient_pos_tags
            and word.isalnum()
            and len(word) > 1
            and word.lower() not in immutable_words
        ):
            continue

        salient_words.append(
            (
                words_count[word]
                - 1,  # to differentiate the first and second occurrence of the same word
                word,
            )
        )
    return salient_words

def string_replace(text, _from, _to, idx):
    nth = idx + 1
    arr = text.split(_from)
    part1 = _from.join(arr[:nth])
    part2 = _from.join(arr[nth:])
    return part1 + _to + part2

def get_word_antonyms(word: str):
    antonyms_lists = set()
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms_lists.add(l.antonyms()[0].name())
    return list(antonyms_lists)

def constrained_sum_sample_pos(n, total, low=0):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(
        np.random.choice(np.arange(1, total - (low - 1) * n), n - 1, replace=False)
    )

    return [
        a - b + low - 1
        for a, b in zip(dividers + [total - (low - 1) * n], [0] + dividers)
    ]


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
            self.device = device

            self.num_beams = args.num_beams
            self.num_beam_groups = args.num_beam_groups
            self.max_num_samples = args.max_num_samples

            self.bleu = bleu
            self.rouge = rouge
            self.meteor = meteor
            self.acc_metric = args.acc_metric
            self.sentencoder = SentenceEncoder(device=args.device)

            self.ori_lens, self.adv_lens = [], []
            self.ori_bleus, self.adv_bleus = [], []
            self.ori_rouges, self.adv_rouges = [], []
            self.ori_meteors, self.adv_meteors = [], []
            self.ori_time, self.adv_time = [], []
            self.cos_sims = []
            self.att_success = 0
            self.total_pairs = 0
            self.eos_token = self.tokenizer.eos_token
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id


            # self.record = []
            #att_method = args.attack_strategy
            out_dir = args.out_dir
            model_n = args.model_name_or_path.split("/")[-1]
            dataset_n = DATA2NAME.get(args.dataset, args.dataset.split("/")[-1])
            #combined = "combined" if args.use_combined_loss and att_method == 'structure' else "single"
            #max_per = args.max_per
            #fitness = args.fitness if att_method == 'structure' else 'performance'
            select_beams = args.select_beams
            max_num_samples = args.max_num_samples
            print(args.crossover_flag)
            if args.crossover_flag == 1:
                att_method = "POPOP_crossover_" + args.objective
            else:
                att_method = "POPOP_" + args.objective
            num_gen = "5gen"
            num_ind = args.num_ind
            file_path = f"{out_dir}/{att_method}_{num_ind}_{num_gen}_{select_beams}_{model_n}_{dataset_n}_{max_num_samples}.txt"
            self.write_file_path = file_path
            file_path_gen = file_path.split('.txt')[0]
            self.write_file_path_gen = file_path_gen + "_Gen.txt"
            self.berttokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
            self.bertmodel = AutoModelForMaskedLM.from_pretrained('bert-large-uncased').eval().to(args.device)
            self.grammar = GrammarChecker()

    def mask_words(self,original_sentence, idx_words):

    # berttokenizer.mask_token == [MASK]
        return [
            string_replace(
                text=original_sentence, _from=word, _to=self.berttokenizer.mask_token, idx=idx
            )
            for idx, word in idx_words
        ]

    def predict_masked_sentences_for_salient_words(self,sentence, num_sentences=20, top_k=5):
        salient_words = identify_salient_words_with_idx(sentence)
        cur_error = self.grammar.check(sentence)
        # Check if there are no salient words
        if not salient_words:
            print("No salient words identified in the sentence.")
            return [sentence] * num_sentences  # or return an appropriate message or empty list

        random.shuffle(salient_words)

        min_num_sents = 0
        if len(salient_words) < num_sentences:
            min_num_sents = 1

        try:
            num_sent_per_word = constrained_sum_sample_pos(
                len(salient_words), num_sentences, low=min_num_sents
            )
        except ValueError as e:
            print(f"Error: {e}")
            return []

        # Remove word with 0 sentence
        num_sent_per_word = [n for n in num_sent_per_word if n != 0]
        salient_words = [
            w for idx, w in enumerate(salient_words) if idx < len(num_sent_per_word)
        ]

        masked_sentences = self.mask_words(sentence, salient_words)

        inputs = self.berttokenizer(masked_sentences, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.bertmodel(**inputs)
            logits = outputs.logits

        inputs = inputs.to("cpu")
        logits = logits.to("cpu")

        input_ids = inputs["input_ids"]
        mask_token_indices = input_ids == self.berttokenizer.mask_token_id
        assert (mask_token_indices.sum(-1) == 1).all()  # Only one masked word per sentence

        mask_token_logits = logits[mask_token_indices]
        generated_sentences = set()

        for mt_logit, m_sent, n_sent, (_, original_word) in zip(
            mask_token_logits, masked_sentences, num_sent_per_word, salient_words
        ):
            top_predictions = mt_logit.topk(n_sent * 5 + 1).indices.tolist()
            count = 0
            antonyms = get_word_antonyms(original_word)
            for predicted_index in top_predictions:
                predicted_token = self.berttokenizer.decode([predicted_index]).strip()
                if not predicted_token.isalnum() or predicted_token.lower().strip() == original_word.lower().strip():
                    continue
                if predicted_token in antonyms:  # Filter out antonyms
                    continue
                if count >= n_sent:
                    break
                new_sentence = m_sent.replace(self.berttokenizer.mask_token, predicted_token, 1)
                sim = self.sentencoder.get_sim(new_sentence, sentence)
                new_error = self.grammar.check(new_sentence)
                if (sim < 0.80 or sim > 1.0) and (new_error > cur_error):
                    continue

                generated_sentences.add(new_sentence)
                count += 1

        result_list = list(generated_sentences)
        if len(result_list) < num_sentences:
            last_sentence = result_list[-1] if result_list else sentence
            result_list.extend([last_sentence] * (num_sentences - len(result_list)))

        return result_list

    def initialize_population(self, ori_text, num_individuals):
        return self.predict_masked_sentences_for_salient_words(ori_text, num_individuals)

    
    def remove_pad(self, s: torch.Tensor):
        return s[torch.nonzero(s != self.tokenizer.pad_token_id)].squeeze(1)

    def compute_seq_len(self, seq: torch.Tensor):
        if seq.shape[0] == 0: # empty sequence
            return 0
        if seq[0].eq(self.tokenizer.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id)))
        else:
            return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id))) - 1

    def get_average_score(self,generated_sentences: list, reference_sentences: list, acc_metric):
        scores = []

        for gen_sentence, ref_sentence in zip(generated_sentences, reference_sentences):
            # Calculate BLEU score
            bleu_res = self.bleu.compute(
                predictions=[gen_sentence],
                references=[ref_sentence],
                smooth=True,
            )
            #bleu_score = bleu_res['bleu']
            bleu_score = bleu_res['bleu']
            # Calculate ROUGE-L score
            rouge_res = self.rouge.compute(
                predictions=[gen_sentence],
                references=[ref_sentence],
            )
            rouge_score = rouge_res['rougeL']

            # Calculate METEOR score
            meteor_res = self.meteor.compute(
                predictions=[gen_sentence],
                references=[ref_sentence],
            )
            meteor_score_value = meteor_res['meteor']

            # Calculate the average of BLEU, ROUGE, and METEOR
            average_score = (bleu_score + rouge_score + meteor_score_value) / 3.0
            average_score = average_score + 1
            if acc_metric =="combined":
                scores.append(average_score)
            elif acc_metric == "bleu":
                bleu_score = bleu_score + 1
                scores.append(bleu_score)
            elif acc_metric == "rouge":
                rouge_score = rouge_score + 1
                scores.append(rouge_score)
            elif acc_metric == "meteor":
                meteor_score_value = meteor_score_value + 1
                scores.append(meteor_score_value)


        return scores

    def get_prediction_obj(self,sentence: Union[str, List[str]]):
            text = sentence
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1023,
            )
            input_ids = inputs['input_ids'].to(args.device)
            # ['sequences', 'sequences_scores', 'scores', 'beam_indices']
            outputs = dialogue(
                self.model,
                input_ids,
                early_stopping=False,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
                num_beam_groups=1,
                use_cache=True,
                max_length=1024,
            )

            seqs = outputs['sequences'].detach()
    #         if self.task == 'seq2seq':
    #             seqs = outputs['sequences'].detach()
    #         else:
    #             seqs = outputs['sequences'][:, input_ids.shape[-1]:].detach()

            seqs = [self.remove_pad(seq) for seq in seqs]
            out_scores = outputs['scores']
            pred_len = [self.compute_seq_len(seq) for seq in seqs]
            return pred_len, seqs, out_scores

    def compute_batch_score(self, text: List[str]):
            batch_size = len(text)
            num_beams =  1
            batch_size = len(text)
            index_list = [i * 1 for i in range(batch_size + 1)]
            pred_len, seqs, out_scores = self.get_prediction_obj(text)

            scores = [[] for _ in range(batch_size)]
            for out_s in out_scores:
                for i in range(batch_size):
                    current_index = index_list[i]
                    scores[i].append(out_s[current_index: current_index + 1])
            scores = [torch.cat(s) for s in scores]
            scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
            return scores, seqs, pred_len


    def compute_score(self, text: Union[str, List[str]], batch_size: int = None):
            total_size = len(text)
            if batch_size is None:
                batch_size = len(text)

            if batch_size < total_size:
                scores, seqs, pred_len = [], [], []
                for start in range(0, total_size, batch_size):
                    end = min(start + batch_size, total_size)
                    score, seq, p_len = self.compute_batch_score(text[start: end])
                    pred_len.extend(p_len)
                    seqs.extend(seq)
                    scores.extend(score)
            else:
                scores, seqs, pred_len = self.compute_batch_score(text)
            return scores, seqs, pred_len
    
    def get_target_p(self, scores: list, pred_len: list, label: list):
            targets = []
            for i, s in enumerate(scores): # s: T X V
                if pred_len[i] == 0:
                    targets.append(torch.tensor(0.0).to(self.device))
                else:
                    # if self.pad_token_id != self.eos_token_id:
                    s[:, self.pad_token_id] = 1e-12
                    softmax_v = softmax(s) # T X V
                    target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(label[:softmax_v.size(0)])])
                    target_p = target_p[:pred_len[i]]
                    targets.append(torch.sum(target_p))
            return torch.stack(targets).detach().cpu().numpy()


    def objective_length(self,pop,context,guided_mess):
        pred_length = []
        for sentence in pop:
            text = context + self.eos_token + sentence
            scores, seqs, p_len = self.compute_score([text])
            pred_length.append(p_len[0])
        return pred_length
    
    def objective_accuracy(self,pop,context,guided_mess):
        pred_acc = []
        for sentence in pop:
            text = context + self.eos_token + sentence
            scores, seqs, p_len = self.compute_score([text])
            # label = self.tokenizer(guided_mess, truncation=True, max_length=1024, return_tensors='pt')
            # label = label['input_ids'][0] # (T, )
            # res = self.get_target_p(scores, p_len, label) # numpy array (N, )
            generated_responses = [self.tokenizer.batch_decode([seq], skip_special_tokens=True)[0].strip() for seq in seqs]
            accuracy = self.get_average_score(generated_responses, [guided_mess],self.acc_metric)
            pred_acc.append(accuracy[0])
        return pred_acc

    import random
    def mutation(self,pop):
        new_pop = []  # This will store the new mutated population

        for sentence in pop:
            # Assume predict_masked_sentences_for_salient_words is a function that provides possible mutations
            mutated_sentences = self.predict_masked_sentences_for_salient_words(sentence,10)

            if mutated_sentences:
                # Choose a random mutation from the options
                mutated_sentence = random.choice(mutated_sentences)
                #mutated_sentence = mutated_sentences[0]
            else:
                # If no mutations are available, keep the original sentence
                mutated_sentence = sentence

            # Add the mutated or original sentence to the new population
            new_pop.append(mutated_sentence)

        return new_pop

    def tournament_selection(self, pop, pop_fitness, selection_size, tournament_size=4):
        if selection_size is None:
            selection_size = len(pop)

        # Partition the population into non-overlapping tournaments
        def partition(pop):
            num_tournaments = int(len(pop) / tournament_size)
            index = np.arange(len(pop))
            np.random.shuffle(index)
            return [index[tournament_size*i:tournament_size*(i+1)] for i in range(num_tournaments)]

        offspring = []
        selected_indices = []
        while len(offspring) < selection_size:
            tournaments = partition(pop)
            for tournament in tournaments:
                tournament_inds = [pop[i] for i in tournament]
                tournament_fitness = [pop_fitness[i] for i in tournament]
                indices = np.argsort(tournament_fitness)
                offspring.append(tournament_inds[indices[-1]])

        return np.array(offspring[:selection_size])
    def crossover(self, pop):

        new_pop = []  # This will store the new population after crossover

        # Iterate through the population two items at a time
        for i in range(0, len(pop) - 1, 2):
            sentence_1 = pop[i]
            sentence_2 = pop[i + 1]

            # Calculate the split points for each sentence
            split_point_1 = len(sentence_1.split()) // 2
            split_point_2 = len(sentence_2.split()) // 2

            # Split the sentences into two parts each
            part1_1 = sentence_1.split()[:split_point_1]
            part1_2 = sentence_1.split()[split_point_1:]
            part2_1 = sentence_2.split()[:split_point_2]
            part2_2 = sentence_2.split()[split_point_2:]

            # Create new sentences by crossover
            new_sentence_1 = ' '.join(part1_1 + part2_2)
            new_sentence_2 = ' '.join(part2_1 + part1_2)

            # Add the new sentences to the new population
            new_pop.append(new_sentence_1)
            new_pop.append(new_sentence_2)

        # If the original population has an odd number of elements, add the last one unchanged
        if len(pop) % 2 != 0:
            new_pop.append(pop[-1])

        return new_pop


    def POPOP(
        self,
        context,
        sentence,
        guided_messages,
        objective,
        selection_func,
        num_individuals,
        max_evaluations,
        seed,
    ):
        random.seed(seed)     # python random generator
        np.random.seed(seed)  # numpy random generator

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        pop = self.initialize_population(sentence, num_individuals)
        pop_fitness = objective(pop, context, guided_messages)
        num_evaluations = num_individuals

        selection_size = num_individuals
        best_fitness = []

        # Initialize the tqdm progress bar
        with tqdm(total=max_evaluations, desc="Evaluations", unit="eval") as pbar:
            while num_evaluations < max_evaluations:
                best_fitness.append([num_evaluations, np.max(pop_fitness)])

                if args.crossover_flag == 1:
                    offspring = self.crossover(pop)
                offspring = self.mutation(pop)
                offspring_fitness = objective(offspring, context, guided_messages)

                num_evaluations += num_individuals

                # Update the tqdm progress bar
                pbar.update(num_individuals)

                pop_off = np.concatenate((pop, offspring))
                pop_off_fitness = np.concatenate((pop_fitness, offspring_fitness))

                # tournament selection will have a constant tournament_size of 4
                pop = selection_func(pop_off, pop_off_fitness, selection_size)
                pop_fitness = objective(pop, context, guided_messages)

            best_fitness.append([num_evaluations, np.max(pop_fitness)])

        return pop, pop_fitness, best_fitness

    def log_and_save(self, display: str):
        print(display)
        with open(self.write_file_path, 'a') as f:
            f.write(display + "\n")
        #self.write_file.write(display + "\n")   

    def log_and_save_gen(self,display: str):
        print(display)
        with open(self.write_file_path_gen, 'a') as f:
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


    def generation_step(self, instance: dict):
        # Set up
        num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)
        for entry_idx in range(num_entries):
            free_message, guided_message, original_context, references = self.prepare_entry(
                instance,
                entry_idx,
                context,
                prev_utt_pc,
                total_entries,
            )
            if guided_message is None:
                continue

            prev_utt_pc += [
                free_message,
                guided_message,
            ]
            self.log_and_save("\nDialogue history: {}".format(original_context))
            self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(free_message, references[-1]))
            # Original generation
            eos_token = self.tokenizer.eos_token
            text = original_context + eos_token + free_message
            output, time_gap = self.get_prediction(text)
            self.log_and_save("G--{}".format(output))

            if not output:
                continue
            #print("OUTPUT:",output)
            #print("REF:",references)
            bleu_res, rouge_res, meteor_res, pred_len = self.eval_metrics(output, references)
            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                pred_len, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
            ))
            self.ori_lens.append(pred_len)
            self.ori_bleus.append(bleu_res['bleu'])
            self.ori_rouges.append(rouge_res['rougeL'])
            self.ori_meteors.append(meteor_res['meteor'])
            self.ori_time.append(time_gap)

            # Attack
            #success, adv_his = self.attacker.run_attack(text, guided_message)
            #new_text = adv_his[-1][0]
            #new_free_message = new_text.split(self.sp_token)[1].strip()
            #cos_sim = self.attacker.sent_encoder.get_sim(new_free_message, free_message)
            self.model = self.model.to(args.device)
            if args.crossover_flag == 1:
                print("BAT DAU POPOP VOI CROSSOVER")
            else:
                print("BAT DAU POPOP")
            if args.objective == "acc":
                pop, pop_fitness, best_fitness = self.POPOP(original_context, free_message, guided_message, self.objective_accuracy, self.tournament_selection ,num_individuals= args.num_ind, max_evaluations= args.num_gen * args.num_ind, seed = args.seed)
            else:
                pop, pop_fitness, best_fitness = self.POPOP(original_context, free_message, guided_message, self.objective_length, self.tournament_selection ,num_individuals= args.num_ind, max_evaluations= args.num_gen * args.num_ind, seed = args.seed)

            pop_with_fitness = list(zip(pop_fitness, pop))

            if args.objective == "acc":
                sorted_pop_with_fitness = sorted(pop_with_fitness, key=lambda x: x[0])
            else:
                sorted_pop_with_fitness = sorted(pop_with_fitness, key=lambda x: x[0],reverse=True)

            best_individual = sorted_pop_with_fitness[0][1]
            best_fitness_value = sorted_pop_with_fitness[0][0]
            print("Pop:", pop)
            print("Candidate:", best_individual)

            # problem = Problem(self.model, self.tokenizer,original_context, free_message, guided_message)

            # evolution = Evolution(problem, num_of_generations=5, num_of_individuals=18, num_of_tour_particips=2,
            #           tournament_prob=0.9, crossover_param=2, mutation_param=5)

            # resulting_front = evolution.evolve()
            # result = []
            # for individual in resulting_front:
            #     result.append((individual.sentence,individual.eos_loss, individual.cls_loss))
            #     #print(individual.sentence, individual.cls_loss, individual.eos_loss)
            #sorted_data = sorted(result, key=lambda x: x[1])
            #new_free_message = sorted_data[0][0]

            new_text = original_context + self.tokenizer.eos_token + best_individual
            cos_sim = self.sentencoder.get_sim(best_individual, free_message)
            output, time_gap = self.get_prediction(new_text)
            if not output:
                continue

            self.log_and_save("U'--{} (cosine: {:.3f})".format(best_individual, cos_sim))
            self.log_and_save("G'--{}".format(output))
            adv_bleu_res, adv_rouge_res, adv_meteor_res, adv_pred_len = self.eval_metrics(output, references)

            # ASR
            success = (
                (bleu_res['bleu'] > adv_bleu_res['bleu']) or
                (rouge_res['rougeL'] > adv_rouge_res['rougeL']) or
                (meteor_res['meteor'] > adv_meteor_res['meteor'])
                #) and cos_sim > 0.01
                ) and cos_sim > 0.7
            if success:
                self.att_success += 1
            else:
                self.log_and_save("Attack failed!")

            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                adv_pred_len, time_gap, adv_bleu_res['bleu'], adv_rouge_res['rougeL'], adv_meteor_res['meteor'],
            ))
            self.adv_lens.append(adv_pred_len)
            self.adv_bleus.append(adv_bleu_res['bleu'])
            self.adv_rouges.append(adv_rouge_res['rougeL'])
            self.adv_meteors.append(adv_meteor_res['meteor'])
            self.adv_time.append(time_gap)
            self.cos_sims.append(cos_sim)
            self.total_pairs += 1

  

    def adv_load_metrics_and_find_last_entry(self,log_file_path):
        metrics = {
            "adv_lens": [],
            "adv_bleus": [],
            "adv_rouges": [],
            "adv_meteors": [],
            "adv_times": [],
            "cos_sims": [],
            "total_pairs": 0,
            "att_success": 0,
        }
        last_dialogue = None
        capture_next_metrics = False  # Flag to determine if next metrics should be captured
        attack_failed_pattern = re.compile(r'Attack failed!')
        attack_failed = 0
        with open(log_file_path, "r") as file:
            for line in file:
                if "Dialogue history:" in line:
                    last_dialogue = line.strip().split(": ")[1]

                if "U'--" in line:
                    # Capture cosine similarity
                    cos_sim_match = re.search(r'cosine: ([0-9.]+)', line)
                    if cos_sim_match:
                        metrics['cos_sims'].append(float(cos_sim_match.group(1)))
                        if float(cos_sim_match.group(1)) < 0.7:
                            attack_failed +=1
                    capture_next_metrics = True  # Set flag to capture next metrics

                elif attack_failed_pattern.search(line):
                    #metrics['att_success'] += 1  # Counting successful attacks inversely by 'Attack failed!'
                    attack_failed += 1

                elif capture_next_metrics:
                    metric_match = re.search(r'\(length: (\d+), latency: ([0-9.]+), BLEU: ([0-9.]+), ROUGE: ([0-9.]+), METEOR: ([0-9.]+)\)', line)
                    if metric_match:
                        metrics['adv_lens'].append(float(metric_match.group(1)))
                        metrics['adv_times'].append(float(metric_match.group(2)))
                        metrics['adv_bleus'].append(float(metric_match.group(3)))
                        metrics['adv_rouges'].append(float(metric_match.group(4)))
                        metrics['adv_meteors'].append(float(metric_match.group(5)))
                        metrics['total_pairs'] += 1  # Increment counter for each adv sample processed
                        capture_next_metrics = False  # Reset flag after capturing
        metrics['att_success'] = metrics['total_pairs'] - attack_failed  # Adjusting successful attack count
        return last_dialogue, metrics

    import re

    def ori_load_metrics_and_find_last_entry(self, log_file_path):
        metrics = {
            "ori_lens": [],
            "ori_bleus": [],
            "ori_rouges": [],
            "ori_meteors": [],
            "ori_times": [],
            "total_pairs": 0
        }
        last_dialogue = None
        capture_next_metrics = False
        with open(log_file_path, "r") as file:
            for line in file:
                if "Dialogue history:" in line:
                    last_dialogue = line.strip().split(": ")[1]

                if "U--" in line:
                    capture_next_metrics = True
                elif capture_next_metrics:
                    metric_match = re.search(r'\(length: (\d+), latency: ([0-9.]+), BLEU: ([0-9.]+), ROUGE: ([0-9.]+), METEOR: ([0-9.]+)\)', line)
                    if metric_match:
                        metrics['ori_lens'].append(int(metric_match.group(1)))
                        metrics['ori_times'].append(float(metric_match.group(2)))
                        metrics['ori_bleus'].append(float(metric_match.group(3)))
                        metrics['ori_rouges'].append(float(metric_match.group(4)))
                        metrics['ori_meteors'].append(float(metric_match.group(5)))
                        metrics['total_pairs'] += 1  # Increment each time an original sentence metrics are captured
                        capture_next_metrics = False

        return last_dialogue, metrics

    def find_start_index(self,test_dataset, last_dialogue):
        for i, instance in tqdm(enumerate(test_dataset)):
            num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)
            for entry_idx in range(num_entries):
                free_message, guided_message, original_context, references = self.prepare_entry(
                        instance,
                        entry_idx,
                        context,
                        prev_utt_pc,
                        total_entries,
                )
                if guided_message is None:
                    continue

                prev_utt_pc += [
                    free_message,
                    guided_message,
                ]

                if original_context == last_dialogue:
                    return i, original_context
                    # Log for debugging
                #print(f"Checking context: {original_context} against last_dialogue: {last_dialogue}")
                print("\nDialogue history: {}".format(original_context))
                print("\nLast Dialogue: {}".format(last_dialogue))
        return -1, None  # If no match found
    
    def generation(self, test_dataset: Dataset):
        if self.dataset == "empathetic_dialogues":
            test_dataset = self.group_ED(test_dataset)

        # Sample test dataset
        last_dialogue = None
        if args.resume:
            start_index = 0
            ids = random.sample(range(len(test_dataset)), self.max_num_samples)
            test_dataset = test_dataset.select(ids)
            last_dialogue, adv_metrics = self.adv_load_metrics_and_find_last_entry(self.args.resume_log_dir)
            last_dialogue, ori_metrics = self.ori_load_metrics_and_find_last_entry(self.args.resume_log_dir)
            # Extend current metrics with the loaded ones
            self.ori_lens.extend(adv_metrics['adv_lens'])
            self.adv_time.extend(adv_metrics['adv_times'])
            self.adv_bleus.extend(adv_metrics['adv_bleus'])
            self.adv_rouges.extend(adv_metrics['adv_rouges'])
            self.adv_meteors.extend(adv_metrics['adv_meteors'])
            self.cos_sims.extend(adv_metrics['cos_sims'])
            self.total_pairs += adv_metrics['total_pairs']

            self.ori_lens.append(ori_metrics['ori_lens'])
            self.ori_bleus.append(ori_metrics['ori_bleus'])
            self.ori_rouges.append(ori_metrics['ori_rouges'])
            self.ori_meteors.append(ori_metrics['ori_meteors'])
            self.ori_time.append(ori_metrics['ori_times'])

            # Update log file to a new file to avoid overlap
            #new_log_filename = os.path.splitext(self.args.resume_log_dir)[0] + "_continued.txt"
            #self.write_file = open(new_log_filename, "w")
            print(f"Resuming from: {last_dialogue}, logging to new file")

            # ids = random.sample(range(len(test_dataset)), self.max_num_samples)
            # test_dataset = test_dataset.select(ids)
            print("Test dataset: ", test_dataset)
            print("LENGTH DATASET:", len(test_dataset))

            if last_dialogue:
            # Iterate through dataset to find where this history matches
                index, found_context = self.find_start_index(test_dataset, last_dialogue)
                if index != -1:
                    print(f"Resuming from index: {index}, context: {found_context}")
                    start_index = index
                else:
                    print("No matching context found. Please check the 'last_dialogue' or dataset processing.")

            # for i, instance in tqdm(enumerate(test_dataset)):
            #     self.generation_step(instance)
            #print("Test dataset: ", start_index)
            #print(test_dataset[index])

            for i, instance in tqdm(enumerate(test_dataset)):
                if i > start_index:
                    self.generation_step(instance)

        else:
            ids = random.sample(range(len(test_dataset)), self.max_num_samples)
            test_dataset = test_dataset.select(ids)
            print("Test dataset: ", test_dataset)
            for i, instance in tqdm(enumerate(test_dataset)):
                self.generation_step(instance)
            #total_samples = len(test_dataset)

            # Check if the maximum number of samples equals the dataset size
            # if self.max_num_samples == total_samples:
            #     print("Using the full dataset without sampling.")
            # else:
            #     # Sample the dataset if the number of max samples is less than the total
            #     ids = random.sample(range(total_samples), min(self.max_num_samples, total_samples))
            #     test_dataset = test_dataset.select(ids)

            # print("Test dataset: ", test_dataset)
            # for i, instance in tqdm(enumerate(test_dataset)):
            #     self.generation_step(instance)

        Ori_len = np.mean(self.ori_lens)
        Adv_len = np.mean(self.adv_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Adv_bleu = np.mean(self.adv_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Adv_rouge = np.mean(self.adv_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Adv_meteor = np.mean(self.adv_meteors)
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
    
def main(args: argparse.Namespace):
    random.seed(args.seed)
        #random.seed(seed)     # python random generator
    np.random.seed(args.seed)  # numpy random generator

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_name_or_path = args.model_name_or_path
    dataset = args.dataset
    max_len = args.max_len
    max_per = args.max_per
    num_beams = args.num_beams
    select_beams = args.select_beams
            #fitness = args.fitness
    num_beam_groups = args.num_beam_groups
    #     att_method = args.attack_strategy
    #     cls_weight = args.cls_weight
    #     eos_weight = args.eos_weight
    #     delta = args.delta
    #     use_combined_loss = args.use_combined_loss
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    config = AutoConfig.from_pretrained(model_name_or_path, num_beams=num_beams, num_beam_groups=num_beam_groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gpt' in model_name_or_path.lower():
        task = 'clm'
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        if 'results' not in model_name_or_path.lower():
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            tokenizer.add_special_tokens({'mask_token': '<MASK>'})
            model.resize_token_embeddings(len(tokenizer))
    else:
        task = 'seq2seq'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

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
        #attacker=attacker,
        device=args.device,
        task=task,
        bleu=bleu,
        rouge=rouge,
        meteor=meteor,
    )
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
    parser.add_argument("--model_name_or_path", "-m", type=str, default="results/Bart", help="Path to model")
    parser.add_argument("--dataset", "-d", type=str, default="blended_skill_talk",
                        choices=["blended_skill_talk", "conv_ai_2", "empathetic_dialogues", "AlekseyKorshuk/persona-chat"],
                        help="Dataset to attack")
    parser.add_argument("--out_dir", type=str,
                        default="./results/logging",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--objective", type=str, default="acc", choices=["acc", "len"], help="Objective")
    parser.add_argument("--num_ind", type=int, default=100, help="Number of Individuals")
    parser.add_argument("--num_gen", type=int, default=50, help="Number of Individuals")
    parser.add_argument("--crossover_flag", type=int, default=0, help="Whether to use Crossover or not")
    parser.add_argument("--device", type=str,default="cuda",help="Determine which GPU to use")
    parser.add_argument("--resume", action="store_true", help="Resume from the last processed entry")
    parser.add_argument("--resume_log_dir", type=str,
                        default="/kaggle/working/results/logging",
                        help="Output directory")
    parser.add_argument("--attack_strategy", "-a", type=str,
                        default='structure',
                        choices=[
                            'structure', #DGSlow
                            'word',
                            'pwws',
                            'bae',
                            'fd', # white-box attack
                            'hotflip', # white-box attack
                            ],
                        help="Attack strategy")
    args = parser.parse_args()
    main(args)

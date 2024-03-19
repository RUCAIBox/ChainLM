import json
import argparse
from typing import List, Literal, Optional, Tuple, TypedDict
from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
general_public_messages = []
scientist_messages = []
mathematician_messages = []
judge_messages = []

def init_debators():
# 3 debators
    general_public_messages.clear()
    scientist_messages.clear()
    mathematician_messages.clear()
    judge_messages.clear()
    general_public_messages.append({"role": "system", "content": "You are the general public. Given a problem, you should give one step of your Chain-of-Thought answer. For each step, we will have a debating and the judge will decide the final answer for this step. You need to give the next step based on the previous steps until the judge gives the final answer. It must be noted that you can only give one step at a time."})
    scientist_messages.append({"role": "system", "content": "You are the scientist. Given a problem and one solving step, you should judge whether the step and discussion are correct. If it is not correct, you should give your reason and your opinion of the correct step."})
    mathematician_messages.append({"role": "system", "content": "You are the mathematician. Given a problem and one solving step, you should judge whether the step and discussion are correct. If it is not correct, you should give your reason and your opinion of the correct step."})
    judge_messages.append({"role": "system", "content": "You are the judge. Given a problem and the debating process of one solving step, you should judge which opinion is correct and give the answer of the very step. If you can conclude the final answer directly, repeat the final answer with 'Debate ended.' in the end. If there are choices in the question, give the right choice."})

def add_messages(role, content, forwho):
    message = {"role": "assistant", "content": role + " says: " + content} 
    if "general_public" in forwho:
        general_public_messages.append(message)
    if "scientist" in forwho:
        scientist_messages.append(message)
    if "mathematician" in forwho:
        mathematician_messages.append(message)
    if "judge" in forwho:
        judge_messages.append(message)

def load_model(
        model_path: str,
        device: str,
):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif "cuda" in device:
        kwargs = {"torch_dtype": torch.float16}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except ValueError:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as e:
        print(e)
        return None

    return model, tokenizer

def chat_completion(
    args,
    tokenizer,
    model,
    dialogs: List[Dialog],
    temperature: float = 0.7,
    max_gen_len: Optional[int] = 512,
    ) -> List[ChatPrediction]:

    if max_gen_len is None:
        max_gen_len = model.params.max_seq_len - 1
    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:
        unsafe_requests.append(
            any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
        )
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)

    generation_tokens = model.generate(
        torch.as_tensor(prompt_tokens).to(args.device),
        temperature=temperature,
        max_new_tokens=max_gen_len,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    return [
        {
            "generation": {
                "role": "assistant",
                "content": tokenizer.decode(t).split("[/INST]")[-1].split("</s>")[0].strip(),
            }
        }
        for t in generation_tokens
    ]
    
def get_input(question_file):
    # Load questions file
    question_jsons = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    return question_jsons


def run_debating(args, test_data, prompt):
    # Evaluate the model for answers
    model, tokenizer = load_model(
        args.model_path, args.device
    )
    if "cuda" in args.device or args.device == "mps":
        model.to(args.device)
    
    for i, line in enumerate(tqdm(test_data)):
        final_answer = ""
        init_debators()
        test = json.loads(line)
        problem = test["question"]
        if prompt != "":
            examples = {"role": "user", "content": prompt}
            general_public_messages.append(examples)
            scientist_messages.append(examples)
            mathematician_messages.append(examples)
            judge_messages.append(examples)
        question = {"role": "user", "content": "The problem is : "+problem+" Your first reasoning step is: " }
        general_public_messages.append(question)
        step_1 = chat_completion(args, tokenizer, model, [general_public_messages])
        question = {"role": "user", "content": "The problem is : "+problem+" The first reasoning step is: " + step_1[0]["generation"]["content"] }
        scientist_messages.append(question)
        mathematician_messages.append(question)
        judge_messages.append(question)
        cnt = 0
        while final_answer == "":
            if cnt >= 3:
                judge_messages.append({"role": "user", "content": "Debata ended. Please conclude the final answer. If there are choices in the question, give the right choice."})
                jud_ans = chat_completion(args, tokenizer, model, [judge_messages])
                judge_messages.append({"role": "user", "content": jud_ans[0]["generation"]["content"]})
                final_answer = jud_ans[0]["generation"]["content"]
                break
            scientist_messages.append({"role": "user", "content": "Give your opinion."})
            sci_ans = chat_completion(args, tokenizer, model, [scientist_messages])
            print(sci_ans[0]["generation"]["content"])
            add_messages("scientist", sci_ans[0]["generation"]["content"],["scientist", "mathematician", "judge", "general_public"])
            mathematician_messages.append({"role": "user", "content": "Give your opinion."})
            mat_ans = chat_completion(args, tokenizer, model, [mathematician_messages])
            print(mat_ans[0]["generation"]["content"])
            add_messages("mathematician", mat_ans[0]["generation"]["content"], ["scientist", "mathematician", "judge", "general_public"])
            judge_messages.append({"role": "user", "content": "Judge which opinion is correct in the above debating and give the answer of the very step. Or if you can conclude the final answer directly, give the final answer with \"Debate ended.\" in the end. ATTENTION: If there are choices in the question, give the right choice."})
            jud_ans = chat_completion(args, tokenizer, model, [judge_messages])
            print(jud_ans[0]["generation"]["content"])
            add_messages("judge", jud_ans[0]["generation"]["content"], ["scientist", "mathematician", "judge", "general_public"])
            if "Debate ended" in jud_ans[0]["generation"]["content"] or "debate ended" in jud_ans[0]["generation"]["content"]:
                final_answer = jud_ans[0]["generation"]["content"]
            else:
                cnt += 1
                next_prompt = "The result of this step of the debate is: "+ jud_ans[0]["generation"]["content"] + " According to the above content, the next reasoning step is: " 
                add_messages("judge", next_prompt, ["general_public"])
                next_step = chat_completion(args, tokenizer, model, [general_public_messages])
                print(next_step[0]["generation"]["content"])
                scientist_messages.append({"role": "user", "content": next_prompt+next_step[0]["generation"]["content"]})
                mathematician_messages.append({"role": "user", "content": next_prompt+next_step[0]["generation"]["content"]})
                judge_messages.append({"role": "user", "content": next_prompt+next_step[0]["generation"]["content"]})
        print("The final answer is: " + final_answer)
        dump_jsonl({"question": problem, "generation": final_answer, "ground_truth": test["answer"], "judge":judge_messages, "general_public":general_public_messages, "scientist":scientist_messages,"mathematican":mathematician_messages}, args.answer_file, append=True)

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=""
    )
    parser.add_argument( 
        "--device",
        type=str,
        default="cuda:0",
        help="The device type",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=""
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=""
    )
    args = parser.parse_args()
    if args.prompt == "":
        prompt = ""
    else:
        with open(args.prompt, "r") as f:
            prompt = f.read()
    input = get_input(args.test_file)
    run_debating(
        args,
        input, 
        prompt
    )

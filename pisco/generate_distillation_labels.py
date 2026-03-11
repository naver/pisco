import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

system_prompt = "You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible."
user_prompt = "Background:\n<docs>\n\nQuestion: <question>"


if __name__ == "__main__":
    with torch.no_grad():
        ds = datasets.load_from_disk(
            "/beegfs/scratch/user/hdejean/data_splare/spladev3_8n_qds.hf"
        )

        top_k = 5
        batch_size = 64
        max_new_tokens = 128

        model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # ok for MSMarco data.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to("cuda")
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def build_text_prompt(elt):
            query = elt["q"]
            docs = elt["docs"][: min(len(elt["docs"]), top_k)]
            docs = " ".join([f"Document {i}: {doc}" for i, doc in enumerate(docs)])

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt.replace("<question>", query).replace(
                        "<docs>", docs
                    ),
                },
            ]

            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

        def run_batched_inference(dataset):
            outputs = []
            for start in tqdm(range(0, len(dataset), batch_size)):
                end = min(start + batch_size, len(dataset))
                prompts = [build_text_prompt(dataset[i]) for i in range(start, end)]

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to(model.device)

                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)

                for j in range(out.shape[0]):
                    input_width = inputs["input_ids"].shape[1]
                    new_tokens = out[j, input_width:]
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    outputs.append(text)

            return outputs

        distillation_labels = run_batched_inference(ds)
        ds2 = ds.add_column("mistral_label", distillation_labels).rename_column("q", "query")
        ds2 = datasets.DatasetDict({"train": ds2})

        save_path = "/beegfs/scratch/user/mlouis/calmar/pisco/datasets/spladev3_8n_qds_with_mistral_labels.hf"
        ds2.save_to_disk(save_path)

        print("Saved dataset to:", save_path)

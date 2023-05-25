from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn


class generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.generator = AutoModelForCausalLM.from_pretrained(args.generator)
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator)
        self.beam = args.beam

    def get_tokens(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.generator.generate(inputs.input_ids, min_length=inputs.input_ids.shape[1]+1, max_length=inputs.input_ids.shape[1]+1, do_sample=False, num_beams=self.beam, num_return_sequences=self.beam)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
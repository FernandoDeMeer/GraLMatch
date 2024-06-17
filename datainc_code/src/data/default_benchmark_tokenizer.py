from src.data.tokenizer import DataIncTokenizer


class DefaultBenchmarkTokenizer(DataIncTokenizer):
    def __init__(self, model_name, do_lower_case, max_seq_length):
        super().__init__(model_name, do_lower_case, max_seq_length)

    def generate_sample(self, l_txt: list, r_txt: list, label: int = 0):
        seq = super().build_sequence(l_txt, r_txt, label)
        return seq

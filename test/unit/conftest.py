class FakeRegister:
    def __init__(self, name="n_pi"):
        self.name = name
        self.init_val = None
        self.reset_calls = 0
        self.set_calls = []

    def reset(self):
        self.reset_calls += 1

    def set_to(self, *args, **kwargs):
        self.set_calls.append((args, kwargs))

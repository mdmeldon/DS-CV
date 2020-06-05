from catalyst.dl.callbacks import AUCCallback as CatalystAUCCallback

class AUCCallback(CatalystAUCCallback):

    def on_batch_end(self, state):
        logits = state.output[self.output_key].detach().float()
        targets = state.input[self.input_key].detach().float()
        probabilities = self.activation_fn(logits)

        for i in range(self.num_classes):
            target = (targets == i).float()
            self.meters[i].add(probabilities[:, i], target)

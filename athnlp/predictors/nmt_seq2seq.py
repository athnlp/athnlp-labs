from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import matplotlib.pyplot as plt
import numpy as np

@Predictor.register('nmt_seq2seq')
class NmTSeq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models that visualizes attention scores
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source" : source})

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        # outputs["attentions"] shape: (source_tokens_length, predicted_tokens_length)
        self.plot_heatmap(instance["source_tokens"].tokens,
                          outputs["predicted_tokens"],
                          outputs["attentions"])
        return sanitize(outputs)

    def plot_heatmap(self, source, target, scores):
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(scores, cmap='viridis')

        ax.set_xticklabels(source, minor=False, rotation='vertical')
        ax.set_yticklabels(target, minor=False)

        # put the major ticks at the middle of each cell
        # and the x-ticks on top
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(scores.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(scores.shape[1]) + 0.5, minor=False)
        ax.invert_yaxis()

        plt.colorbar(heatmap)
        plt.show()

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)
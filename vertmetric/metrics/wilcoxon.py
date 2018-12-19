
import logging

from vertmetric.metrics import metric
from vertmetric.utils import general as gen
import scipy.stats
from rouge import Rouge as RougeLib


class Wilcoxon (metric.Metric):
    def __init__(self):
        super(Wilcoxon, self).__init__()
        self.logger = logging.getLogger('vert')

    def score(self):
        self.logger.info("Calculating Wilcoxon scores.")

        r_scores_baseline = RougeLib().get_scores(
            self.baseline,
            self.targets,
            avg=False
        )
        r_scores_model = RougeLib().get_scores(
            self.model,
            self.targets,
            avg=False
        )
        self.logger.info("Done: calculating ROUGE scores.")

        baseline_rouge_1 = []
        baseline_rouge_2 = []
        baseline_rouge_l = []
        for score in r_scores_baseline:
            baseline_rouge_1.append(score['rouge-1']['f'])
            baseline_rouge_2.append(score['rouge-2']['f'])
            baseline_rouge_l.append(score['rouge-l']['f'])

        model_rouge_1 = []
        model_rouge_2 = []
        model_rouge_l = []
        for score in r_scores_model:
            model_rouge_1.append(score['rouge-1']['f'])
            model_rouge_2.append(score['rouge-2']['f'])
            model_rouge_l.append(score['rouge-l']['f'])

        _, wilcoxon_rouge1 = scipy.stats.wilcoxon(baseline_rouge_1, model_rouge_1)
        _, wilcoxon_rouge2 = scipy.stats.wilcoxon(baseline_rouge_2, model_rouge_2)
        _, wilcoxon_rougel = scipy.stats.wilcoxon(baseline_rouge_l, model_rouge_l)

        return {
            'wilcoxon-rouge-1': wilcoxon_rouge1,
            'wilcoxon-rouge-2': wilcoxon_rouge2,
            'wilcoxon-rouge-l': wilcoxon_rougel,
        }

    def load_files(self, model_f, target_f, baseline_f):
        self.model = list()
        self.targets = list()
        self.baseline = list()

        with open(model_f, 'r') as gen_f:
            for line in gen_f:
                line = line.strip('\n')
                self.model.append(line)

        with open(target_f, 'r') as tgt_f:
            for line in tgt_f:
                line = line.strip('\n')
                self.targets.append(line)

        with open(baseline_f, 'r') as bas_f:
            for line in bas_f:
                line = line.strip('\n')
                self.baseline.append(line)

    @classmethod
    def save_report_to_file(cls, report, out_dir='./', filename=''):
        """
        Args:
            report (dict): metrics calculated to be dumped to JSON
            filename (str): optional specification.
        Returns:
            None
        """
        if filename == '':
            filename = gen.generate_filename('rouge')
        super(Wilcoxon, cls).save_report_to_file(report, out_dir, filename)

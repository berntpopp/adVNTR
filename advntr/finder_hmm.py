"""Finder HMM construction with explicit run rates and legacy cache compatibility."""
import logging
import os

from advntr import settings
from advntr.hmm_utils import get_read_matcher_model, get_read_matcher_model_enhanced
from advntr.profiler import time_usage
from advntr.run_context import runtime_value
from hmm.hmm import Model


class FinderHMM(object):
    @time_usage
    def build_vntr_matcher_hmm(self, copies, flanking_region_size=100):
        patterns = self.reference_vntr.get_repeat_segments()
        sorted_unique_repeat_units = sorted(list(set(patterns)))
        for i, ru in enumerate(sorted_unique_repeat_units):
            logging.info("RU{} {}".format(i+1, ru))
        left_flanking_region = self.reference_vntr.left_flanking_region[-flanking_region_size:]
        right_flanking_region = self.reference_vntr.right_flanking_region[:flanking_region_size]

        if runtime_value(self, 'enhanced_hmm'):
            vntr_matcher = get_read_matcher_model_enhanced(left_flanking_region, right_flanking_region,
                                                           patterns, copies, None, self.is_frameshift_mode,
                                                           maximum_error_rate=runtime_value(self, 'maximum_error_rate'))
        else:
            vntr_matcher = get_read_matcher_model(left_flanking_region, right_flanking_region, patterns, copies)
        return vntr_matcher

    def get_vntr_matcher_hmm(self, read_length):
        """Try to load trained HMM for this VNTR
        If there was no trained HMM, it will build one and store it for later usage
        """
        logging.info('Using read length %s' % read_length)
        copies = self.get_copies_for_hmm(read_length)

        if self.run_context is not None:
            return self.build_vntr_matcher_hmm(copies, read_length)

        base_name = str(self.reference_vntr.id) + '_' + str(read_length) + '.json'
        stored_hmm_file = settings.TRAINED_HMMS_DIR + base_name
        if runtime_value(self, 'trained_hmms') and os.path.isfile(stored_hmm_file):
            model = Model()
            model = model.from_json(stored_hmm_file)
            return model

        flanking_region_size = read_length
        vntr_matcher = self.build_vntr_matcher_hmm(copies, flanking_region_size)

        if runtime_value(self, 'trained_hmms'):
            json_str = vntr_matcher.to_json()
            with open(stored_hmm_file, 'w') as outfile:
                outfile.write(json_str)
        return vntr_matcher

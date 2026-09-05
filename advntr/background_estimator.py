"""Task 8c: Background model estimator and null distribution fitting.

Estimates the frozen frameshift background model (p0) from calibration sinks.
"""
import json
import math
import os
import re

from advntr.exact_tail import exact_indel_tail_log
from advntr.frameshift_opportunities import parse_components

SINK_SCHEMA = 'advntr.frameshift.calibration'
SINK_VERSION = 1

#: The frozen estimator. Pre-registered 2026-09-03; not tunable, and read from here by
#: every code path so a change is a one-line diff a reviewer cannot miss.
HYPERPARAMETERS = {
    'kprot': 4,                       # MIN_SUPPORTING_READ_COUNT + 1
    'phi': 2.0,
    'min_events': 10,                 # MIN_EVENTS
    'dispersion_threshold': 3.0,      # X^2/df above which the Bernoulli null is refuted
    'dispersion_min_events': 20,      # screen eligibility: sum_i k_is
    'dispersion_min_samples': 15,     # screen eligibility: contributing control samples
    'floor_target': 0.001,            # settings.INDEL_MUTATION_MIN_PVALUE
    #: NOT a hyperparameter of the frozen estimator. Always True for any fit that is
    #: emitted; set False only by the 8b 5.2 component ablation, which needs a run with
    #: the multiplicity floor removed. `advntr/frameshift_background.py` refuses `p0 = 0`,
    #: so "no floor" is realised as `ABLATION_EPSILON`, and the ablation's numbers must be
    #: read with that substitution stated.
    'apply_floor': True,
}

#: The stand-in for "no floor" in the ablation. A true zero is inadmissible in the v1
#: schema, so the ablation reports what the smallest loadable value does instead.
ABLATION_EPSILON = 1e-12

#: `advntr/settings.py:43`. Read here rather than imported so the fitter states its own
#: dependency on the number; asserted against the shipped value by `fit_background.py`.
SETTINGS_MIN_SUPPORTING_READ_COUNT = 3

#: The largest `p0` `advntr/frameshift_background.py` will load. See ruling 5.
MAX_PROBABILITY = 1.0 - 1e-12

#: The family of `State` strings the shipped grammar can produce, one component.
#: `advntr/hmm_utils.py:637-647` (repeat submodel), `:366-376` (prefix), `:432-443`
#: (suffix) name every `I`/`D` state `'%s%s_%s' % (kind, index, hmm_name)` with
#: `hmm_name` an integer, `'prefix'` or `'suffix'`; `advntr/mutation_keys.py:153-157`
#: appends the emitted base for a repeat-unit insertion and `:168`/`:204` the `_LEN<n>`
#: suffix; `advntr/vntr_finder.py:422` builds a FLANK insertion candidate straight off
#: the state name, so a flank insertion carries no base character.
GRAMMAR_COMPONENT = re.compile(
    r'^(?:D[1-9]\d*_(?:[1-9]\d*|prefix|suffix)'
    r'|I(?:0|[1-9]\d*)_(?:(?:[1-9]\d*_[ACGT])|prefix|suffix)_LEN[1-9]\d*)$')


class FitterError(Exception):
    """The fit cannot proceed. Always names the file and the problem."""


def write_json(path, document, compact=False):
    """Deterministic JSON: sorted keys, and no trailing whitespace."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    separators = (',', ':') if compact else (',', ': ')
    with open(path, 'w') as handle:
        handle.write(json.dumps(document, sort_keys=True, indent=None if compact else 1,
                                separators=separators))
        handle.write('\n')


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def class_of(state):
    """The state's pattern index, exactly as `advntr/vntr_finder.py:474` derives it."""
    head = state.split('&')[0] if '&' in state else state
    fields = head.split('_')
    if len(fields) < 2:
        return None
    return fields[1]


def is_compound(state):
    return '&' in state

def solve_floor(n_median, kprot, target):
    """The `p` solving `P(K >= kprot | n_median, p) = target`, to machine precision.

    Bisection in `log p` on the shipped `exact_indel_tail_log`, which is monotone
    increasing in `p`. Log space rather than linear because the floors span 3e-5 to
    3e-3 and a linear bisection would spend every iteration in the wrong decade.
    """
    if not isinstance(kprot, int) or kprot < 1:
        raise FitterError('kprot must be a positive integer, got %r' % (kprot,))
    if not 0.0 < target < 1.0:
        raise FitterError('floor target must lie in (0, 1), got %r' % (target,))
    n_median = int(n_median)
    if n_median < kprot:
        raise FitterError(
            'cannot solve a floor at N=%d for kprot=%d: %d events are impossible in %d '
            'trials, so the equation has no solution' % (n_median, kprot, kprot, n_median))
    goal = math.log(target)
    low, high = math.log(1e-300), math.log(MAX_PROBABILITY)
    if exact_indel_tail_log(kprot, n_median, math.exp(high)) < goal:
        raise FitterError('no floor solves P(K>=%d | N=%d, p)=%r below p=1'
                          % (kprot, n_median, target))
    for _ in range(400):
        middle = (low + high) / 2.0
        if exact_indel_tail_log(kprot, n_median, math.exp(middle)) < goal:
            low = middle
        else:
            high = middle
        if high - low < 1e-15:
            break
    solution = math.exp(high)
    while solution < MAX_PROBABILITY and exact_indel_tail_log(kprot, n_median, solution) < goal:
        solution *= 1.0 + 1e-14
    return min(max(solution, 1e-300), MAX_PROBABILITY)

class BackgroundFit(object):
    """The fitted background: one `p0` per enumerated state, plus every diagnostic."""

    def __init__(self, probabilities, default_probability, diagnostics, class_rates,
                 class_floors, class_n_median, mixed_class_states, hyperparameters,
                 control_count, degenerate_classes, clamped_states):
        self.probabilities = probabilities
        self.default_probability = default_probability
        self.diagnostics = diagnostics
        self.class_rates = class_rates
        self.class_floors = class_floors
        self.class_n_median = class_n_median
        self.mixed_class_states = mixed_class_states
        self.hyperparameters = hyperparameters
        self.control_count = control_count
        self.degenerate_classes = degenerate_classes
        self.clamped_states = clamped_states

    @property
    def screened_states(self):
        return sorted(state for state, entry in self.diagnostics.items()
                      if entry['tier'] == 'envelope')


def fit_background(control_observations, states, hyperparameters):
    """The pre-registered estimator, over CONTROLS of the calibration partition only.

    `control_observations` must already be the control subset: carriers are used for the
    discrimination diagnostic and for sensitivity in CV, never for a rate (8b 1.4,
    "including carriers puts the pathogenic rate into the null"). The caller's obligation
    is visible in `test_carriers_never_enter_the_rate_even_when_present`.

    Untruncated: every control contributes its `N` for every enumerated state, including
    the controls where `k = 0`. A control whose `N_is == 0` for a state does not
    contribute to that state at all -- it demonstrated no trial, so it carries no
    information about that state's rate.
    """
    states = sorted(set(states))
    kprot = hyperparameters['kprot']
    phi = hyperparameters['phi']
    min_events = hyperparameters['min_events']
    threshold = hyperparameters['dispersion_threshold']
    screen_min_k = hyperparameters['dispersion_min_events']
    screen_min_samples = hyperparameters['dispersion_min_samples']
    target = hyperparameters['floor_target']

    per_state = {}
    class_k = {}
    class_n = {}
    class_denominators = {}
    mixed = []
    for state in states:
        klass = class_of(state)
        components = parse_components(state)
        if components is not None and len(set(part[2] for part in components)) > 1:
            mixed.append(state)
        sum_k = 0
        mass = 0
        contributing = 0
        envelope = 0.0
        samples = []
        for observation in control_observations:
            support, opportunities = observation.evidence(state)
            if opportunities <= 0:
                continue
            contributing += 1
            sum_k += support
            mass += opportunities
            samples.append((support, opportunities))
            rate = support / float(opportunities)
            if rate > envelope:
                envelope = rate
            class_denominators.setdefault(klass, []).append(opportunities)
        class_k[klass] = class_k.get(klass, 0) + sum_k
        class_n[klass] = class_n.get(klass, 0) + mass
        per_state[state] = {'state': state, 'class': klass, 'sum_k': sum_k,
                            'opportunity_mass': mass,
                            'contributing_samples': contributing,
                            'envelope': envelope, 'samples': samples,
                            'compound': is_compound(state),
                            'components': None if components is None else len(components)}

    class_rates = {}
    for klass in class_n:
        class_rates[klass] = (class_k[klass] / float(class_n[klass])
                              if class_n[klass] else 0.0)

    apply_floor = hyperparameters.get('apply_floor', True)
    class_n_median = {}
    class_n_median_raw = {}
    class_floors = {}
    degenerate = []
    clamped_medians = []
    for klass in sorted(set(list(class_n) + list(class_denominators))):
        denominators = class_denominators.get(klass, [])
        if denominators:
            raw = median(denominators)
            centre = int(round(raw))
            if centre < kprot:
                # The solver has no root below `kprot`: `kprot` events are impossible in
                # fewer than `kprot` trials. Raising the median to `kprot` gives the
                # largest floor the solver can return, which is the conservative
                # direction. Recorded so a class where this fires is visible.
                centre = kprot
                clamped_medians.append(klass)
        else:
            raw = None
            centre = kprot
            degenerate.append(klass)
        class_n_median[klass] = centre
        class_n_median_raw[klass] = raw
        class_floors[klass] = (solve_floor(centre, kprot, target) if apply_floor
                               else ABLATION_EPSILON)

    default_probability = (max(class_floors.values()) if class_floors
                           else (solve_floor(kprot, kprot, target) if apply_floor
                                 else ABLATION_EPSILON))
    default_probability = min(default_probability, MAX_PROBABILITY)

    probabilities = {}
    clamped = []
    for state in states:
        entry = per_state[state]
        klass = entry['class']
        pooled = (entry['sum_k'] / float(entry['opportunity_mass'])
                  if entry['opportunity_mass'] else 0.0)
        entry['pooled_rate'] = pooled
        eligible = (entry['sum_k'] >= screen_min_k
                    and entry['contributing_samples'] >= screen_min_samples)
        entry['dispersion_eligible'] = eligible
        entry['dispersion_df'] = max(entry['contributing_samples'] - 1, 0)
        entry['dispersion'] = None
        # The dispersion is COMPUTED for every state that reaches the event threshold,
        # because pre-registered prediction 4 is about "states with sum k >= 20" and
        # says nothing about the sample count. It is only ever ACTED ON -- i.e. allowed
        # to move a state onto the envelope rate -- when `eligible` is true, which is
        # the pre-registered rule verbatim. Reporting a number more widely than the
        # estimator uses it does not change the estimator.
        if entry['sum_k'] >= screen_min_k and entry['dispersion_df'] > 0:
            entry['dispersion'] = _pearson_dispersion(entry['samples'], pooled,
                                                      entry['dispersion_df'])
        screened = (eligible and entry['dispersion'] is not None
                    and entry['dispersion'] > threshold)
        if screened:
            entry['tier'] = 'envelope'
            rate = entry['envelope']
        elif entry['sum_k'] >= min_events:
            entry['tier'] = 'own'
            rate = pooled
        else:
            entry['tier'] = 'class'
            rate = class_rates.get(klass, 0.0)
        entry['rate'] = rate
        floor = class_floors.get(klass, default_probability)
        entry['floor'] = floor
        scaled = phi * rate
        value = max(scaled, floor)
        entry['binding'] = 'rate' if scaled >= floor else 'floor'
        if value >= 1.0:
            value = MAX_PROBABILITY
            entry['clamped'] = True
            clamped.append(state)
        else:
            entry['clamped'] = False
        if value <= 0.0:
            raise FitterError(
                'state %s would be emitted at p0=%r, which the shipped loader refuses; '
                'this can only happen if its class floor is zero' % (state, value))
        entry['p0'] = value
        probabilities[state] = value
        del entry['samples']

    fit = BackgroundFit(probabilities, default_probability, per_state, class_rates,
                        class_floors, class_n_median, sorted(mixed),
                        dict(hyperparameters), len(control_observations),
                        sorted(degenerate), sorted(clamped))
    fit.class_n_median_raw = class_n_median_raw
    fit.class_n_median_clamped_to_kprot = sorted(clamped_medians)
    return fit


def _pearson_dispersion(samples, rate, df):
    """`X^2/df` with `X^2 = sum_i (k_i - N_i r)^2 / (N_i r (1 - r))`, 8b 1.2(c)."""
    if df <= 0 or not 0.0 < rate < 1.0:
        return None
    total = 0.0
    for support, opportunities in samples:
        variance = opportunities * rate * (1.0 - rate)
        if variance <= 0.0:
            continue
        residual = support - opportunities * rate
        total += residual * residual / variance
    return total / df

def grammar_states(repeat_unit_lengths, flank_length=None, insert_lengths=8):
    """Every SIMPLE `State` the shipped grammar can produce for this model.

    8b 4.2, as corrected by the controller's Ruling 18: enumerate all simple states, not
    only the observed ones, so an unobserved simple state gets its class rate and class
    floor rather than the conservative default. Only unobserved COMPOUNDS fall to
    `default_probability`.

    The admissible position range is read off the code that names the states, not
    guessed:

    - repeat submodel of `L` match columns (`advntr/hmm_utils.py:637-647`): insert states
      `I0..IL` (`for i in range(len(matches) + 1)`), match and delete states `M1..ML`,
      `D1..DL` (`for i in range(1, len(matches) + 1)`). So there is no `D0`.
    - prefix matcher (`:366-376`) and suffix matcher (`:432-443`) have exactly the same
      shape over their own pattern: `I0..IL`, `D1..DL`.
    - the flank pattern is the flanking region truncated to the READ LENGTH, not to the
      100-base default: `advntr/vntr_finder.py:117-118` sets `flanking_region_size =
      read_length` before `build_vntr_matcher_hmm`, whose own default of 100 (`:88`) is
      only used by callers that pass nothing. Measured on the public capture: the largest
      `suffix` position in any span signature is 151, and `read_length` is 151.
    - a repeat-unit insertion candidate carries the emitted base
      (`advntr/mutation_keys.py:153-157`) and a `_LEN<n>` suffix (`:168`), while a FLANK
      insertion candidate is built straight off the state name with only the `_LEN`
      suffix (`advntr/vntr_finder.py:422`) -- so `I0_prefix_LEN1`, never
      `I0_prefix_A_LEN1`.
    """
    states = set()
    for pattern in sorted(repeat_unit_lengths):
        length = repeat_unit_lengths[pattern]
        for position in range(1, length + 1):
            states.add('D%d_%s' % (position, pattern))
        for position in range(0, length + 1):
            for base in ('A', 'C', 'G', 'T'):
                for run in range(1, insert_lengths + 1):
                    states.add('I%d_%s_%s_LEN%d' % (position, pattern, base, run))
    if flank_length:
        for flank in ('prefix', 'suffix'):
            for position in range(1, flank_length + 1):
                states.add('D%d_%s' % (position, flank))
            for position in range(0, flank_length + 1):
                for run in range(1, insert_lengths + 1):
                    states.add('I%d_%s_LEN%d' % (position, flank, run))
    return states


def validate_emitted_keys(states, observed_keys, grammar_states=None):
    """Refuse any key a byte-exact `states.get(state, default)` could never match.

    `advntr/frameshift_background.py`'s `probability_for` is a bare dict lookup with no
    normalisation, so a key with surrounding whitespace -- or any key that is not a
    byte-exact emitted `State` -- is accepted, never matches, and SILENTLY scores that
    state against `default_probability`. With ~21,000 generated keys a systematic defect
    would be silent across the whole table, which is why this runs here as well as in the
    loader: the fitter must not need the loader to catch its own bug.

    Returns the key count by origin: `observed` (seen in a sink of the calibration
    partition) or `grammar` (generated by the shipped naming rules). A key that is
    neither is refused.
    """
    origins = {'observed': 0, 'grammar': 0}
    stripped = {}
    for key in states:
        if not isinstance(key, str) and not isinstance(key, unicode):  # noqa: F821
            raise FitterError('states key %r is not a string' % (key,))
        stripped.setdefault(key.strip(), []).append(key)
    collisions = sorted(group for group in stripped.values() if len(group) > 1)
    if collisions:
        raise FitterError('states keys collide once stripped: %s' % (collisions,))
    for key in sorted(states):
        if key == '':
            raise FitterError('a states key is the empty string')
        if key != key.strip():
            raise FitterError('states key %r has leading or trailing whitespace' % (key,))
        for component in key.split('&'):
            if component != component.strip():
                raise FitterError('states key %r has whitespace around its %r component'
                                  % (key, component))
        if any(ord(char) >= 128 for char in key):
            raise FitterError('states key %r is not pure ASCII' % (key,))
        if parse_components(key) is None:
            raise FitterError('states key %r is not a form the shipped grammar can '
                              'produce (parse_components rejected it)' % (key,))
        if key in observed_keys:
            origins['observed'] += 1
        elif not is_compound(key) and GRAMMAR_COMPONENT.match(key) and (
                grammar_states is None or key in grammar_states):
            origins['grammar'] += 1
        else:
            raise FitterError(
                'states key %r was neither observed in a sink nor generated by the '
                'shipped grammar; an invented key can never byte-match an emitted State '
                'and would silently score against default_probability' % (key,))
    return origins
def _hyperparameter_phrase(hyperparameters):
    return ('phi=%s, kprot=%s, MIN_EVENTS=%s, dispersion screen X2/df>%s at sum k>=%s '
            'and >=%s contributing control samples, floor solves P(K>=kprot|N_median,p)=%s'
            % (hyperparameters['phi'], hyperparameters['kprot'],
               hyperparameters['min_events'], hyperparameters['dispersion_threshold'],
               hyperparameters['dispersion_min_events'],
               hyperparameters['dispersion_min_samples'],
               hyperparameters['floor_target']))


#: The three disclaimers the brief mandates, verbatim, for the SIMULATED cohort profile.
SIMULATED_DISCLAIMERS = ('calibrated on simulated reads',
                         'not validated for non-simulated data',
                         'not a production default')


def default_disclaimers(source_cohort):
    """The mandated three for the simulated cohort; a truthful analogue otherwise.

    The brief's wording exists so a cohort profile cannot be mistaken for something it
    is not. Pasting it onto a profile built from a DIFFERENT source would make the
    artifact the brief exists to make unmistakable into the thing that misleads, so the
    wording follows the source rather than being hard-coded.
    """
    if source_cohort == 'SIMULATED':
        return list(SIMULATED_DISCLAIMERS)
    return ['calibrated on %s' % source_cohort, 'not validated for any data',
            'not a production default']


def provenance_line(profile_name, profile_version, fit, context):
    """The one compact line the caller logs at run time. Free text; names no sample.

    The v1 schema has no metadata field and refuses unknown ones, so this line is the
    only thing that travels with the artifact into a run log -- which is what makes a
    result attributable to a named null. The full structured description lives in the
    sidecar. `context['source_cohort']` is REQUIRED and is written verbatim: for the
    calibration profile it is the literal string `SIMULATED`, which is the wording the
    brief mandates, and there is deliberately no default, so a profile cannot acquire
    that claim by omission.
    """
    if not context.get('source_cohort'):
        raise FitterError(
            'provenance needs an explicit source_cohort: the artifact must state what '
            'it was calibrated on, and "SIMULATED" must never be reachable by default')
    depth = context.get('depth_median')
    depth_phrase = ('median VNTR depth %s; ' % depth) if depth is not None else ''
    controls = context.get('controls', context.get('samples'))
    disclaimers = (context.get('disclaimers')
                   or default_disclaimers(context['source_cohort']))
    return (
        '%s%s v%s: frameshift background for adVNTR. Source cohort: %s. Estimated on %s '
        'control samples of the %s partition, out of %s samples; carriers excluded from '
        'every rate. Capture design: %s; %s%s state-specific rates over %s classes, '
        'default_probability %.6g. Estimator: p0 = max(phi * rate, class floor); %s. '
        'This artifact was %s.'
        % (context.get('provenance_banner', ''), profile_name, profile_version,
           context['source_cohort'], controls,
           context.get('partition', 'calibration'),
           context.get('samples', 'an unstated number of'),
           context.get('design', 'unstated'), depth_phrase, len(fit.probabilities),
           len(fit.class_floors), fit.default_probability,
           _hyperparameter_phrase(fit.hyperparameters), '; '.join(disclaimers)))


def artifact_document(fit, provenance, observed=None, grammar_states=None):
    """A v1 `advntr.frameshift.background` document and nothing else.

    `advntr/frameshift_background.py:KNOWN_FIELDS` refuses an unknown top-level field, so
    everything the v1 schema has no room for goes in the sidecar instead.
    """
    if observed is None:
        for key in fit.probabilities:
            if not isinstance(key, (str, unicode)):  # noqa: F821
                raise FitterError('states key %r is not a string' % (key,))
            if key.strip() != key:
                raise FitterError('states key %r has leading or trailing whitespace' % (key,))
            if any(ord(char) >= 128 for char in key):
                raise FitterError('states key %r is not pure ASCII' % (key,))
            if parse_components(key) is None:
                raise FitterError('states key %r is not a form the shipped grammar can '
                                  'produce (parse_components rejected it)' % (key,))
    else:
        validate_emitted_keys(fit.probabilities, observed, grammar_states=grammar_states)
    return {'schema': 'advntr.frameshift.background', 'version': 1,
            'provenance': provenance,
            'default_probability': fit.default_probability,
            'states': dict(fit.probabilities)}


def sidecar_document(profile_name, profile_version, fit, context, notes):
    """Everything the v1 artifact cannot carry, in machine-readable form."""
    source_cohort = context.get('source_cohort', 'SIMULATED')
    if source_cohort == 'SIMULATED':
        source_note = (
            'The reads this profile was calibrated on are SIMULATED. Not "in-silico", '
            'not "modeled": simulated. No claim about non-simulated data follows from '
            'anything in this file.')
    else:
        source_note = 'Source cohort: %s. Capture design: %s.' % (
            source_cohort, context.get('design', 'unstated'))
    return {
        'schema': 'advntr-bench.frameshift.background.sidecar',
        'version': 1,
        'profile_name': profile_name,
        'profile_version': profile_version,
        'source_cohort': source_cohort,
        'source_cohort_note': source_note,
        'partition': context.get('partition'),
        'capture_design': context.get('design'),
        'depth_distribution': context.get('depth'),
        'estimation_population': {
            'samples_in_partition': context.get('samples'),
            'controls_used_for_rates': context.get('controls'),
            'carriers_present_but_excluded': context.get('carriers'),
            'excluded': context.get('excluded'),
            'truncation': ('untruncated inventory: every control contributes its N for '
                           'every enumerated state, including the controls where k = 0; '
                           'a control with N == 0 for a state does not contribute to '
                           'that state'),
        },
        'hyperparameters': fit.hyperparameters,
        'class_rates': fit.class_rates,
        'class_floors': fit.class_floors,
        'class_n_median': fit.class_n_median,
        'default_probability': fit.default_probability,
        'state_count': len(fit.probabilities),
        'screened_states': fit.screened_states,
        'clamped_states': fit.clamped_states,
        'mixed_class_states': fit.mixed_class_states,
        'degenerate_classes': fit.degenerate_classes,
        'adVNTR_provenance': context.get('provenance_pins'),
        'disclaimers': [
            'calibrated on simulated reads',
            'not validated for non-simulated data',
            'not a production default',
        ],
        'not_a_production_default': True,
        'notes': notes,
    }


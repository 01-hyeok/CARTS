"""CPU-only constraints for the isolated Weather336 score-kernel investigation."""
TARGETS = [('Weather_336', 'V0', 'P100', 'FULL'),
           ('Weather_336', 'V1', 'P100', 'FULL')]
BACKENDS = ['cuda12_lt', 'split64', 'cuda11_blas']

def actual_batches(count):
    assert count >= 32
    return [(0, 32), ((count - 1) // 32 * 32, count)]

def acceptable(row):
    return (row.get('channels') == 21 and row.get('bank_count') == 36216
            and row.get('query_exact') is True and row.get('keys_exact') is True
            and row.get('scores_exact') is True and row.get('ordered_top500_exact') is True
            and row.get('mask_exact') is True and not row.get('error'))

def select_backend(rows):
    # Predetermined priority, exact past-only arithmetic comparison, no metric tuning.
    selected = {}
    for model in ['V0', 'V1']:
        for size in [7, 28]:
            passing = [name for name in BACKENDS if any(
                r.get('model') == model and r.get('batch') == size
                and r.get('backend') == name and acceptable(r) for r in rows)]
            if not passing:
                return None
            selected[model + ':' + str(size)] = passing[0]
    return selected

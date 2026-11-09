"""Narrow recovery contract. Exact past-only evidence is mandatory."""

PRIORITY = [('lt11_tf32', 'canonical5')]


def applies(cell, training, heads, batch):
    return (cell, training, heads, batch) == ('ETTh1_96', 'FULL', 5, 1)


def choose(proof):
    assert proof['status'] == 'PASS_EXACT_ALL_CHANNEL_CANDIDATE'
    assert proof['checkpoint_unchanged'] is True
    eligible = []
    flags = ['raw_exact', 'encoder_exact', 'query_exact', 'scores_exact',
             'full_ordered_exact', 'controlled_ordered_exact']
    for pair in PRIORITY:
        rows = [r for r in proof['rows'] if (r.get('projection'), r.get('norm')) == pair]
        if len(rows) != 2 or {r['split'] for r in rows} != {'val', 'test'}:
            continue
        if all(not r.get('reference_only', False) and not r.get('error')
               and all(r.get(k) is True for k in flags)
               and r['channels'] == 7 and r['bank_count'] == 8449
               for r in rows):
            eligible.append(pair)
    assert eligible, 'No exact, full-channel, non-reference production path is proven'
    return eligible[0]

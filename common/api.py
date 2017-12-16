"""
This module implements tools to help with api identification,
it is usefull for defining input for pipeline and easy communicate
afterwards.
"""


def describe_fields(dataset, experiment, file='fields.json'):
    import os
    exp_dir = experiment.lower().replace(' ', '_')
    directory = os.path.join('./apidef', exp_dir)
    os.makedirs(directory, exist_ok=True)
    types = dataset.dtypes
    cols = dataset.columns
    struct = _new_struct()
    for i, typ in enumerate(types):
        col = cols[i]
        struct['fields'].append(_new_field(col, typ))
    _save(struct, os.path.join(directory, file))


def _new_struct():
    return {
        'fields': []
    }


def _new_field(name, value):
    return {'name': name, 'dtype': value.name}


def _save(dic, dest):
    import json
    encoder = json.encoder.JSONEncoder()
    s = encoder.encode(dic)
    with open(dest, 'w') as target:
        target.write(s)

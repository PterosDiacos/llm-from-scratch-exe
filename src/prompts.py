def alpaca_style(d):
    prompt = ('Below is an instruction that describes a task. '
        'Write a response that appropriately completes the request.'
        '\n\n### Instruction:\n{0[instruction]}' + 
        ('\n\n### Input:\n{0[input]}' if d['input'] else '')
    ).format(d)
    resp = '\n\n### Response:\n{0[output]}'.format(d)
    return prompt, resp
